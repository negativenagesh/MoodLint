import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import datetime

# Set up output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset class
class ExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Include all available classes in the dataset
        self.classes = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []

        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if os.path.exists(class_dir):
                print(f"Loading {cls} class from {class_dir}")
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[cls])
            else:
                print(f"Warning: Directory {class_dir} not found")
        
        print(f"Loaded {len(self.images)} images across {len(self.classes)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Enhanced data transforms with more augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def prepare_data():
    # Load full dataset with correct path
    print("Loading dataset...")
    dataset_path = '/home/vu-lab03-pc24/MoodLint/Expressions'
    full_dataset = ExpressionDataset(root_dir=dataset_path, transform=None)
    
    # Count samples per class
    class_counts = {}
    for label in full_dataset.labels:
        class_name = full_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")

    # Split dataset into train, validation, and test sets with better stratification
    indices = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=full_dataset.labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[full_dataset.labels[i] for i in temp_idx], random_state=42)

    # Create subsets with appropriate transforms
    train_dataset = Subset(ExpressionDataset(root_dir=dataset_path, transform=train_transforms), train_idx)
    val_dataset = Subset(ExpressionDataset(root_dir=dataset_path, transform=val_test_transforms), val_idx)
    test_dataset = Subset(ExpressionDataset(root_dir=dataset_path, transform=val_test_transforms), test_idx)
    
    print(f"Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")

    # Create data loaders with appropriate batch sizes
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, full_dataset.classes

# Define improved model architecture
class ExpressionRecognitionModel(nn.Module):
    def __init__(self, num_classes=5):  # Updated for all classes
        super(ExpressionRecognitionModel, self).__init__()
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Custom classifier with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

        # Attention mechanism for focusing on important facial features
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features from the backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # [batch_size, 2048, 7, 7]

        # Apply attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # Apply classifier
        x = self.classifier(x)
        return x

# Training and validation functions
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Update learning rate at each step
        scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# Implementation of mixup augmentation
def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_model(train_loader, val_loader, test_loader, class_names):
    num_classes = len(class_names)
    print(f"Training model for {num_classes} classes: {class_names}")
    
    # Initialize model
    model = ExpressionRecognitionModel(num_classes=num_classes).to(device)

    # Define loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer with decoupled weight decay
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': 1e-4}
    ], weight_decay=1e-2)

    # Learning rate scheduler with warmup
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-4, 5e-4],
        steps_per_epoch=len(train_loader),
        epochs=50,
        pct_start=0.2,  # Warm up for 20% of training
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    num_epochs = 50
    best_val_acc = 0.0
    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []
    
    # Create timestamp for run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(OUTPUT_DIR, f"best_model_{timestamp}.pth")
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        # Apply mixup training for every other epoch
        use_mixup = (epoch % 2 == 0)

        # Training with or without mixup
        if use_mixup:
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Apply mixup
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # Update learning rate
                lr_scheduler.step()

                running_loss += loss.item()

                # For accuracy calculation with mixup
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (lam * (predicted == targets_a).sum().item() +
                            (1 - lam) * (predicted == targets_b).sum().item())

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
        else:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                    optimizer, lr_scheduler, device, scaler)

        # Validation
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

        # Test - evaluate test set for each epoch
        test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        epoch_time = time.time() - epoch_start
        
        print(f'Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.1f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")

    # Load best model and evaluate on test set
    print(f"Loading best model from {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    print(f'Best Model Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # Plot training metrics
    plot_training_metrics(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, timestamp)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds, class_names, timestamp)
    
    # Evaluate class-wise performance
    evaluate_class_performance(test_labels, test_preds, class_names)
    
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    print(f'Final Test Accuracy: {test_acc:.2f}%')
    
    return model, test_acc

def plot_training_metrics(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, timestamp):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    metrics_path = os.path.join(OUTPUT_DIR, f"training_metrics_{timestamp}.png")
    plt.savefig(metrics_path)
    print(f"Training metrics plot saved to {metrics_path}")
    plt.close()

def plot_confusion_matrix(test_labels, test_preds, class_names, timestamp):
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()

def evaluate_class_performance(test_labels, test_preds, class_names):
    print("\nClass-wise Performance:")
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_indices = [j for j, label in enumerate(test_labels) if label == i]
        if class_indices:
            class_preds = [test_preds[j] for j in class_indices]
            class_true = [test_labels[j] for j in class_indices]
            acc = accuracy_score(class_true, class_preds) * 100
            class_accuracy[class_name] = acc
            print(f"{class_name} Class Accuracy: {acc:.2f}%")
        else:
            print(f"{class_name} Class: No samples in test set")

def main():
    print("=" * 50)
    print("Facial Expression Recognition Training")
    print("=" * 50)
    
    # Prepare data
    train_loader, val_loader, test_loader, class_names = prepare_data()
    
    # Train model
    model, accuracy = train_model(train_loader, val_loader, test_loader, class_names)
    
    print("=" * 50)
    print(f"Training completed with final test accuracy: {accuracy:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()