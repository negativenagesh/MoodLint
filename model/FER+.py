import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from facenet_pytorch import MTCNN
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

# Initialize a single MTCNN instance for the main process
mtcnn_global = None
if torch.cuda.is_available():
    try:
        mtcnn_global = MTCNN(
            keep_all=False,
            device=device,
            select_largest=True,
            post_process=False,
            image_size=224
        )
        print("Global MTCNN initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize global MTCNN: {e}")

# Define FER+ optimized data transforms - these match Microsoft's approach
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Global face cache with a size limit
face_cache = {}
MAX_CACHE_SIZE = 500

def extract_face(image_path, mtcnn=None):
    """Extract face from image using MTCNN"""
    if image_path in face_cache:
        return face_cache[image_path]
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        if mtcnn is not None:
            try:
                boxes, probs = mtcnn.detect(image)
                
                if boxes is not None:
                    box = boxes[0]
                    margin = 20
                    x1, y1, x2, y2 = [int(b) for b in box]
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(image.width, x2 + margin)
                    y2 = min(image.height, y2 + margin)
                    
                    face_img = image.crop((x1, y1, x2, y2))
                else:
                    face_img = image
            except Exception as e:
                print(f"Face detection error in {image_path}: {e}")
                face_img = image
        else:
            face_img = image
        
        # Manage cache size
        if len(face_cache) >= MAX_CACHE_SIZE:
            # Remove a random key to keep memory usage bounded
            face_cache.pop(next(iter(face_cache)))
            
        face_cache[image_path] = face_img
        return face_img
    except Exception as e:
        print(f"Error in extract_face for {image_path}: {e}")
        return None

class EmotionDataset(Dataset):
    """Dataset for emotion recognition with FER+ optimizations"""
    
    def __init__(self, image_paths, labels, transform=None, extract_faces=True, mtcnn_detector=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.extract_faces = extract_faces
        self.mtcnn = mtcnn_detector
        
        # Pre-process faces in batches to avoid memory overload
        if self.extract_faces and self.mtcnn is not None:
            print(f"Pre-processing faces in batches...")
            batch_size = 30  # Process 30 images at a time
            for i in range(0, len(image_paths), batch_size):
                print(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
                batch_end = min(i + batch_size, len(image_paths))
                batch_paths = image_paths[i:batch_end]
                
                for path in batch_paths:
                    if path not in face_cache:
                        extract_face(path, self.mtcnn)
                
                # Force garbage collection after each batch
                gc.collect()
                torch.cuda.empty_cache()
                
            print(f"Completed face extraction")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Get face from cache or extract it
            if self.extract_faces:
                if img_path in face_cache:
                    face_img = face_cache[img_path]
                else:
                    face_img = extract_face(img_path, None)
            else:
                face_img = Image.open(img_path).convert("RGB")
            
            # Apply transformations
            if self.transform and face_img is not None:
                face_img = self.transform(face_img)
            else:
                # Create a dummy image
                face_img = torch.zeros((3, 224, 224), device='cpu')
                
            return face_img, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor
            dummy_image = torch.zeros((3, 224, 224), device='cpu')
            return dummy_image, label

def load_dataset(dataset_dir, emotions, extract_faces=True):
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

class FERPlusNet(nn.Module):
    """FER+ optimized model architecture based on Microsoft research"""
    def __init__(self, num_classes=5):
        super(FERPlusNet, self).__init__()
        
        # Use Microsoft recommended VGG-based architecture for FER+
        # Load base VGG model (without classifier)
        self.base_model = models.vgg13_bn(pretrained=True)
        self.base_model.features = self.base_model.features[:-1]  # Remove last max pooling
        
        # Feature extraction layers
        self.features = self.base_model.features
        
        # Spatial Attention Module (similar to Microsoft implementation)
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier as per Microsoft FER+ recommendations
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Apply attention
        attention_mask = self.attention(features)
        attended_features = features * attention_mask
        
        # Global pooling
        pooled_features = self.global_pool(attended_features).view(x.size(0), -1)
        
        # Classification
        output = self.classifier(pooled_features)
        
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
                'test_acc': test_acc,  # Save test accuracy in checkpoint
            }, "best_ferplus_model.pth")
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

def main():
    """GPU-optimized main function with FER+ approach"""
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Define emotion classes for our dataset
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
            dataset_dir, emotions, extract_faces=False
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
        transform=data_transforms['train'],
        extract_faces=True,
        mtcnn_detector=mtcnn_global
    )
    
    # Clear cache between datasets
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Creating validation dataset...")
    val_dataset = EmotionDataset(
        val_paths, val_labels, 
        transform=data_transforms['val'],
        extract_faces=True,
        mtcnn_detector=mtcnn_global
    )
    
    # Clear cache between datasets
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Creating test dataset...")
    test_dataset = EmotionDataset(
        test_paths, test_labels, 
        transform=data_transforms['val'],
        extract_faces=True,
        mtcnn_detector=mtcnn_global
    )
    
    # Clear cache again
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
    
    # Free up memory again
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create the model
    print("\nInitializing FER+ model...")
    model = FERPlusNet(num_classes=len(emotions)).to(device)
    
    # Print model summary
    print(f"Model architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define weighted loss function to handle class imbalance (Microsoft FER+ approach)
    class_weights = torch.FloatTensor([1/count for count in train_class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define optimizer with FER+ recommended parameters
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nTraining FER+ model with mixed precision...")
    train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = train_model(
        model, train_loader, val_loader, test_loader,  # Added test_loader
        criterion, optimizer,
        num_epochs=90, early_stopping_patience=7
    )
    
    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Plot training curves including test accuracy
    print("\nPlotting training results...")
    plt.figure(figsize=(15, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')  # Added test loss
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.plot(test_accs, label='Test Accuracy')  # Added test accuracy
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training, Validation, and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ferplus_training_curves.png', dpi=100)
    plt.close()
    
    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Evaluate the model on the test set for final detailed metrics
    print("\nEvaluating model on test set...")
    test_metrics = evaluate_model(model, test_loader, criterion, emotions)
    
    # Print test results
    print(f"\nFER+ Model Test Results:")
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
    plt.savefig('ferplus_confusion_matrix.png', dpi=100)
    plt.close()
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': emotions,
        'test_accuracy': test_metrics['accuracy'],
    }, "ferplus_final_model.pth")
    
    print("\nTraining and evaluation complete!")
    print(f"Final model saved as 'ferplus_final_model.pth'")
    
    # Final memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()