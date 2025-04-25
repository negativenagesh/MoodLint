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
import multiprocessing
import gc
import time
from torch.cuda.amp import autocast, GradScaler
from timm import create_model

# Fix CUDA reinitialization issue
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Free up memory before starting
gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define optimized data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
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
            # Directly load the pre-cropped face image
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
                
            # Just add file path without loading to save memory
            image_paths.append(img_path)
            labels.append(idx)
            emotion_count += 1
        
        print(f"Loaded {emotion_count} images for emotion '{emotion}'")
    
    # Verify we have data
    if not image_paths:
        raise ValueError("No valid images found in the dataset")
    
    # Split data with stratification for better class balance
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Force garbage collection
    gc.collect()
    
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

class EfficientNetB7FER(nn.Module):
    """Advanced EfficientNetB7 model for facial expression recognition"""
    def __init__(self, num_classes=5):
        super(EfficientNetB7FER, self).__init__()
        
        # Load EfficientNetB7 backbone
        try:
            self.backbone = create_model(
                'efficientnet_b7', 
                pretrained=True,
                num_classes=0,
                global_pool='avg'
            )
            print("Loaded pretrained EfficientNetB7 weights")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}. Using random initialization.")
            self.backbone = create_model('efficientnet_b7', pretrained=False, num_classes=0)
        
        # EfficientNetB7 feature dimension is 2560
        backbone_features = 2560
        
        # Attention module
        self.attention = nn.Sequential(
            nn.BatchNorm1d(backbone_features),
            nn.Linear(backbone_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Feature enhancement module
        self.feature_enhancer = nn.Sequential(
            nn.BatchNorm1d(backbone_features),
            nn.Linear(backbone_features, 1536),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(1536)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(1536, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(768),
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(384),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(192),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply attention
        att_weights = self.attention(features)
        attended_features = features * att_weights
        
        # Enhance features
        enhanced_features = self.feature_enhancer(attended_features)
        
        # Classification
        output = self.classifier(enhanced_features)
        
        return output

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, 
                num_epochs=60, early_stopping_patience=5):
    """GPU-optimized training with mixed precision and test evaluation"""
    train_losses = []
    val_losses = []
    test_losses = []
    train_accs = []
    val_accs = []
    test_accs = []
    
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
            optimizer.zero_grad(set_to_none=True)
            
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
            
            # Print batch progress 
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
        
        # Test phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # Use mixed precision for inference too
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = 100 * test_correct / test_total
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Epoch timing
        time_elapsed = time.time() - start_time
        
        # Print epoch results including test accuracy
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {time_elapsed:.1f}s - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
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
                'test_acc': test_acc,
            }, "best_efficientb7_fer_model.pth")
            print(f"Checkpoint saved (val_loss: {val_loss:.4f}, test_acc: {test_acc:.2f}%)")
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
    
    return train_losses, val_losses, test_losses, train_accs, val_accs, test_accs

def evaluate_model(model, data_loader, criterion, class_names):
    """Memory-efficient evaluation"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    # Process in batches to avoid memory overload
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Use mixed precision for inference
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Move predictions and labels to CPU to save GPU memory
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
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
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
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
        'auc': auc,
        'probabilities': all_probs
    }

def visualize_predictions(model, test_loader, class_names, num_images=8):
    """Visualize model predictions on test images"""
    model.eval()
    images_so_far = 0
    
    fig = plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with autocast():
                outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                if images_so_far == num_images:
                    return
                    
                images_so_far += 1
                ax = plt.subplot(num_images//4, 4, images_so_far)
                ax.axis('off')
                
                # If prediction is correct, green text, else red
                title_color = 'green' if preds[j] == labels[j] else 'red'
                ax.set_title(f'True: {class_names[labels[j]]}\nPred: {class_names[preds[j]]}', 
                             color=title_color)
                
                # Convert tensor to numpy and denormalize
                inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                
                plt.imshow(inp)
                
                if images_so_far == num_images:
                    break
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()

def main():
    """Main function for training and evaluating EfficientNetB7 FER model"""
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Define emotion classes for our dataset
    emotions = ['angry', 'exhausted', 'frustration', 'happy', 'sad']
    
    # Path to the dataset with pre-cropped face images
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
    
    # Smaller batch size due to model size
    batch_size = 8 if torch.cuda.is_available() else 16
    
    # Process datasets one by one to save memory
    print("Creating training dataset...")
    train_dataset = EmotionDataset(
        train_paths, train_labels, 
        transform=data_transforms['train']
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Creating validation dataset...")
    val_dataset = EmotionDataset(
        val_paths, val_labels, 
        transform=data_transforms['val']
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Creating test dataset...")
    test_dataset = EmotionDataset(
        test_paths, test_labels, 
        transform=data_transforms['val']
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # No workers for CUDA to avoid memory issues
    num_workers = 0 if torch.cuda.is_available() else 2
    
    # Create the data loaders with pin_memory for faster GPU transfer
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
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create the model
    print("\nInitializing EfficientNetB7 FER model...")
    model = EfficientNetB7FER(num_classes=len(emotions)).to(device)
    
    # Print model summary
    print(f"Model architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define weighted loss function to handle class imbalance
    class_weights = torch.FloatTensor([1/count for count in train_class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define optimizer with weight decay - AdamW works better for EfficientNetB7
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4,
        weight_decay=1e-5
    )
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nTraining EfficientNetB7 FER model with mixed precision...")
    train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = train_model(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer,
        num_epochs=50, early_stopping_patience=7
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Plot training curves
    print("\nPlotting training results...")
    plt.figure(figsize=(15, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training, Validation, and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('efficientb7_fer_training_curves.png', dpi=100)
    plt.close()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Evaluate the model on the test set for final detailed metrics
    print("\nEvaluating model on test set...")
    test_metrics = evaluate_model(model, test_loader, criterion, emotions)
    
    # Print test results
    print(f"\nEfficientNetB7 FER Model Test Results:")
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
    plt.savefig('efficientb7_fer_confusion_matrix.png', dpi=100)
    plt.close()
    
    # Visualize some predictions
    visualize_predictions(model, test_loader, emotions)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': emotions,
        'test_accuracy': test_metrics['accuracy'],
    }, "efficientb7_fer_final_model.pth")
    
    print("\nTraining and evaluation complete!")
    print(f"Final model saved as 'efficientb7_fer_final_model.pth'")
    
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()