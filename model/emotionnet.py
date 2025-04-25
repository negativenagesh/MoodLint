import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from timm import create_model
import multiprocessing
import gc
import time
from torch.cuda.amp import autocast, GradScaler

# Fix CUDA reinitialization issue
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Free up memory before starting
gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

class EmotionDataset(Dataset):
    """Dataset for emotion recognition with pre-cropped face images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Directly load the image (already a face)
            image = Image.open(img_path).convert("RGB")
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            else:
                # Create a dummy image if transform fails
                image = torch.zeros((3, 224, 224), device='cpu')
                
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor
            dummy_image = torch.zeros((3, 224, 224), device='cpu')
            return dummy_image, label

def load_dataset(dataset_dir, emotions):
    """Memory-optimized dataset loading"""
    image_paths = []
    labels = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Collect all valid images
    for idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(dataset_dir, emotion)
        if not os.path.isdir(emotion_dir):
            raise ValueError(f"Directory not found: {emotion_dir}")
        
        print(f"Loading images from {emotion_dir}...")
        emotion_count = 0
        
        for img_name in os.listdir(emotion_dir):
            # Skip non-image files
            if os.path.splitext(img_name)[1].lower() not in valid_extensions:
                continue
                
            img_path = os.path.join(emotion_dir, img_name)
            if not os.path.isfile(img_path):
                continue
                
            # Just check if the file exists and has a valid extension
            image_paths.append(img_path)
            labels.append(idx)
            emotion_count += 1
        
        print(f"Loaded {emotion_count} images for emotion '{emotion}'")
    
    # Verify we have data
    if not image_paths:
        raise ValueError("No valid images found in the dataset")
    
    # Split data into train, validation, and test sets with stratification
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Force garbage collection
    gc.collect()
    
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

class EmotionNet(nn.Module):
    """Memory-optimized model for emotion detection"""
    def __init__(self, num_classes=5, backbone='efficientnet_b2', pretrained=True):
        super(EmotionNet, self).__init__()
        
        # Load the base model
        try:
            self.base_model = create_model(
                backbone, 
                pretrained=pretrained,
                num_classes=0,
                global_pool='avg'
            )
            print(f"Loaded pretrained {backbone} weights")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}. Using random initialization.")
            self.base_model = create_model(backbone, pretrained=False, num_classes=0)
        
        # Get the number of features
        if backbone.startswith('efficientnet_b0'):
            num_features = 1280
        elif backbone.startswith('efficientnet_b1'):
            num_features = 1280
        elif backbone.startswith('efficientnet_b2'):
            num_features = 1408
        elif backbone.startswith('efficientnet_b3'):
            num_features = 1536
        elif backbone.startswith('efficientnet_b4'):
            num_features = 1792
        elif backbone.startswith('efficientnet_b5'):
            num_features = 2048
        elif backbone.startswith('efficientnet_b6'):
            num_features = 2304
        elif backbone.startswith('efficientnet_b7'):
            num_features = 2560
        else:
            num_features = 1280
        
        # Feature enhancement layers - More memory efficient
        self.feature_enhancer = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(1024)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(768)
        )
            
        # Emotion classifier head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Extract base features
        base_features = self.base_model(x)
        
        # Enhance features
        enhanced_features = self.feature_enhancer(base_features)
        
        # Apply attention
        attention_weights = torch.sigmoid(self.attention(enhanced_features))
        attended_features = enhanced_features * attention_weights
        
        # Apply feature transformation
        transformed_features = self.feature_transform(attended_features)
        
        # Add residual connection if shapes match
        if transformed_features.shape == base_features.shape:
            transformed_features = transformed_features + base_features
            
        # Apply classifier
        output = self.classifier(transformed_features)
        
        return output

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=10, early_stopping_patience=5):
    """GPU-optimized training with mixed precision"""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Initialize the scaler for mixed precision training
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print batch progress every 10 batches
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {i}/{len(train_loader)}")
                
            # Clear GPU memory after batch if needed
            if i % 20 == 0:
                torch.cuda.empty_cache()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # Use mixed precision for inference too
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Epoch timing
        time_elapsed = time.time() - start_time
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {time_elapsed:.1f}s - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping and model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Just save the state_dict to reduce memory usage
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, "best_emotionnet_model.pth")
            print(f"Checkpoint saved (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load the best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, data_loader, criterion, class_names):
    """Memory-efficient evaluation"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    # Process in batches to avoid memory overload
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Use mixed precision for inference
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            # Move predictions and labels to CPU to save GPU memory
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = 100 * np.mean(all_predictions == all_labels)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Calculate ROC AUC score
    try:
        # One-hot encode the predictions and labels for ROC AUC calculation
        n_classes = len(class_names)
        y_true_one_hot = np.eye(n_classes)[all_labels]
        y_pred_one_hot = np.eye(n_classes)[all_predictions]
        auc = roc_auc_score(y_true_one_hot, y_pred_one_hot, average='weighted')
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        auc = 0.0
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                               target_names=class_names, digits=4))
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'auc': auc
    }

def plot_results(train_losses, val_losses, train_accs, val_accs):
    """Plot training results without using too much memory"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=100)  # Lower DPI to save memory
    plt.close()  # Close the figure to free memory

def visualize_predictions(model, data_loader, class_names, num_images=8):
    """Memory-efficient visualization of predictions"""
    model.eval()
    images_so_far = 0
    
    # Create a figure
    plt.figure(figsize=(15, 12))
    
    # Process only a few batches to save memory
    with torch.no_grad():
        for inputs, labels in data_loader:
            if images_so_far >= num_images:
                break
                
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Use mixed precision for inference
            with autocast():
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(min(inputs.size()[0], num_images - images_so_far)):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'True: {class_names[labels[j]]}\nPred: {class_names[preds[j]]}',
                           color=('green' if preds[j] == labels[j] else 'red'))
                
                # Convert tensor to numpy and un-normalize (on CPU)
                img = inputs.cpu()[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                plt.imshow(img)
                
                if images_so_far >= num_images:
                    break
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png', dpi=100)  # Lower DPI to save memory
    plt.close()  # Close the figure to free memory
    
    # Free up memory
    torch.cuda.empty_cache()
    gc.collect()

def main():
    """GPU-optimized main function"""
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Define emotion classes
    emotions = ['angry', 'exhausted', 'frustration', 'happy', 'sad']
    
    # Path to the expression dataset
    dataset_dir = '/home/vu-lab03-pc24/MoodLint/expression'
    
    # Initial memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load the dataset
    try:
        print("Loading dataset...")
        train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = load_dataset(
            dataset_dir, emotions
        )
        
        print(f"Dataset statistics:")
        print(f"Number of training images: {len(train_paths)}")
        print(f"Number of validation images: {len(val_paths)}")
        print(f"Number of test images: {len(test_paths)}")
        
        # Class balance information
        train_class_counts = np.bincount([label for label in train_labels])
        val_class_counts = np.bincount([label for label in val_labels])
        test_class_counts = np.bincount([label for label in test_labels])
        
        print("\nTraining set class distribution:")
        for i, emotion in enumerate(emotions):
            print(f"  {emotion}: {train_class_counts[i]} images")
            
        print("\nValidation set class distribution:")
        for i, emotion in enumerate(emotions):
            print(f"  {emotion}: {val_class_counts[i]} images")
            
        print("\nTest set class distribution:")
        for i, emotion in enumerate(emotions):
            print(f"  {emotion}: {test_class_counts[i]} images")
        
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Memory cleanup
    gc.collect()
    
    # Create datasets and dataloaders
    print("\nCreating data loaders...")
    
    # Smaller batch size to save GPU memory
    batch_size = 16 if torch.cuda.is_available() else 32
    
    # Process datasets one by one to save memory
    print("Creating training dataset...")
    train_dataset = EmotionDataset(
        train_paths, train_labels, 
        transform=data_transforms['train']
    )
    
    # Clear cache between datasets
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Creating validation dataset...")
    val_dataset = EmotionDataset(
        val_paths, val_labels, 
        transform=data_transforms['val']
    )
    
    # Clear cache between datasets
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Creating test dataset...")
    test_dataset = EmotionDataset(
        test_paths, test_labels, 
        transform=data_transforms['val']
    )
    
    # Clear cache again
    gc.collect()
    torch.cuda.empty_cache()
    
    # No workers for CUDA or more for CPU
    num_workers = 0 if torch.cuda.is_available() else 2
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    # Release dataset memory
    train_paths, val_paths, test_paths = None, None, None
    train_labels, val_labels, test_labels = None, None, None
    
    # Free up memory again
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create the model
    print("\nInitializing EmotionNet model...")
    model = EmotionNet(
        num_classes=len(emotions), 
        backbone='efficientnet_b2',
        pretrained=True
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=3e-4, 
        weight_decay=1e-4
    )
    
    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nTraining EmotionNet model with mixed precision...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer,
        num_epochs=25, early_stopping_patience=7
    )
    
    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Plot training curves
    print("\nPlotting training results...")
    plot_results(train_losses, val_losses, train_accs, val_accs)
    
    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Evaluate the model on the test set
    print("\nEvaluating model on test set...")
    test_metrics = evaluate_model(model, test_loader, criterion, emotions)
    
    # Print test results
    print(f"\nEmotionNet Test Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        test_metrics['confusion_matrix'], 
        annot=True, fmt='d', cmap='Blues',
        xticklabels=emotions, 
        yticklabels=emotions
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()  # Close to free memory
    
    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Visualize some predictions
    print("\nVisualizing model predictions...")
    visualize_predictions(model, test_loader, emotions, num_images=8)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': emotions,
        'test_accuracy': test_metrics['accuracy'],
    }, "emotionnet_final_model.pth")
    
    print("\nTraining and evaluation complete!")
    print(f"Final model saved as 'emotionnet_final_model.pth'")
    
    # Final memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()