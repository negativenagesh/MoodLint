import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm import create_model
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np
import seaborn as sns

# Set environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear GPU memory before starting
torch.cuda.empty_cache()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

class EmotionDataset(Dataset):
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
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image and label to avoid crashing; this will be filtered later
            dummy_image = torch.zeros((3, 600, 600))  # Match expected input size
            return dummy_image, label

def load_dataset(dataset_dir, emotions):
    image_paths = []
    labels = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}  # Supported image extensions
    
    for idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(dataset_dir, emotion)
        if not os.path.isdir(emotion_dir):
            raise ValueError(f"Directory not found: {emotion_dir}")
        
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            # Check if the file has a valid image extension
            if os.path.splitext(img_name)[1].lower() not in valid_extensions:
                print(f"Skipping non-image file: {img_path}")
                continue
            # Try opening the image to verify it's valid
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify image integrity
                image_paths.append(img_path)
                labels.append(idx)
            except Exception as e:
                print(f"Skipping invalid image {img_path}: {e}")
                continue
        
        if not image_paths:
            raise ValueError(f"No valid images found in {emotion_dir}")
    
    if not image_paths:
        raise ValueError("No valid images found in the dataset")
    
    # Split the data with stratification
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(EmotionClassifier, self).__init__()
        try:
            self.base_model = create_model('efficientnet_b7', pretrained=True, num_classes=0)
            print("Loaded pretrained EfficientNet-B7 weights.")
        except RuntimeError as e:
            print(f"Failed to load pretrained weights: {e}. Using random initialization.")
            self.base_model = create_model('efficientnet_b7', pretrained=False, num_classes=0)
        
        self.classifier = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x

def evaluate_model(model, data_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    valid_samples = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Skip dummy images (all zeros)
            if inputs.sum() == 0:
                continue
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            valid_samples += 1
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    if valid_samples == 0:
        raise ValueError("No valid samples in data loader")
    
    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    loss = running_loss / valid_samples
    accuracy = np.mean(all_preds == all_labels) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm
    }
    
    return metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        valid_batches = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Skip dummy images
            if inputs.sum() == 0:
                continue
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            valid_batches += 1
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / max(valid_batches, 1)
        train_acc = 100 * correct / max(total, 1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        valid_batches = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.sum() == 0:
                    continue
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                valid_batches += 1
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / max(valid_batches, 1)
        val_acc = 100 * correct / max(total, 1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if scheduler:
            scheduler.step(val_loss)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, "best_emotion_model_checkpoint.pth")
            print(f"Checkpoint saved (val_loss: {val_loss:.4f})")

        torch.cuda.empty_cache()
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return train_losses, val_losses, train_accs, val_accs

def plot_results(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label="Train Loss")
    plt.plot(epochs, val_losses, 'r-', label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epochs")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label="Train Acc")
    plt.plot(epochs, val_accs, 'r-', label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy vs Epochs")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    dataset_dir = '/home/vu-lab03-pc24/MoodLint/expression'
    emotions = ['angry', 'exhausted', 'frustration', 'happy', 'sad']
    
    try:
        train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = load_dataset(dataset_dir, emotions)
        print(f"Number of training images: {len(train_paths)}")
        print(f"Number of validation images: {len(val_paths)}")
        print(f"Number of test images: {len(test_paths)}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    train_dataset = EmotionDataset(train_paths, train_labels, data_transforms['train'])
    val_dataset = EmotionDataset(val_paths, val_labels, data_transforms['val'])
    test_dataset = EmotionDataset(test_paths, test_labels, data_transforms['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    
    model = EmotionClassifier(num_classes=len(emotions)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        scheduler=scheduler, num_epochs=15
    )
    
    plot_results(train_losses, val_losses, train_accs, val_accs)
    
    test_metrics = evaluate_model(model, test_loader, criterion)
    
    print(f"\nEmotion Detection Test Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    torch.save(model.state_dict(), "efficientnet_b7_emotion_no_early_stopping.pth")
    print("Model saved as 'efficientnet_b7_emotion_no_early_stopping.pth'")
    
    torch.cuda.empty_cache()