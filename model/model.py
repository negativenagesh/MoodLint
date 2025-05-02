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

# Create output directories for saving models and plots (using relative paths)
os.makedirs('./output', exist_ok=True)
os.makedirs('./output/model_checkpoints', exist_ok=True)
os.makedirs('./output/plots', exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset class
class ExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Include 5 classes: angry, happy, neutral, sad, surprise
        self.classes = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []

        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if os.path.exists(class_dir):  # Check if directory exists
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[cls])
            else:
                print(f"Warning: Directory {class_dir} not found")

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

# Correct path to your dataset - using the actual path where data exists
dataset_path = "/home/vu-lab03-pc24/MoodLint/Expressions"
print(f"Looking for dataset at: {os.path.abspath(dataset_path)}")

try:
    # Load full dataset
    full_dataset = ExpressionDataset(root_dir=dataset_path, transform=None)

    print(f"Total images found: {len(full_dataset)}")
    print(f"Classes found: {full_dataset.classes}")

    # Split dataset into train, validation, and test sets with better stratification
    indices = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=full_dataset.labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[full_dataset.labels[i] for i in temp_idx], random_state=42)

    # Create subsets with appropriate transforms
    train_dataset = Subset(ExpressionDataset(root_dir=dataset_path, transform=train_transforms), train_idx)
    val_dataset = Subset(ExpressionDataset(root_dir=dataset_path, transform=val_test_transforms), val_idx)
    test_dataset = Subset(ExpressionDataset(root_dir=dataset_path, transform=val_test_transforms), test_idx)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Adjust num_workers based on your system
    num_workers = 2 if os.name != 'nt' else 0  # Use 0 for Windows

    # Create data loaders with appropriate batch sizes
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=True)

except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    print("Please ensure the dataset is organized as follows:")
    print("/home/vu-lab03-pc24/MoodLint/Expressions/")
    print("├── Angry/")
    print("├── Happy/")
    print("├── Neutral/")
    print("├── Sad/")
    print("└── Surprise/")
    exit(1)

# Define improved model architecture
class ExpressionRecognitionModel(nn.Module):
    def __init__(self, num_classes=5):  # Updated to 5 classes
        super(ExpressionRecognitionModel, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')

        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

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

# Initialize model
model = ExpressionRecognitionModel(num_classes=5).to(device)  # Updated to 5 classes

# Define loss function with label smoothing for better generalization
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer with decoupled weight decay
optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-5},
    {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': 1e-4}
], weight_decay=1e-2)

# Learning rate scheduler with warmup
def get_lr_scheduler(optimizer):
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-4, 5e-4],
        steps_per_epoch=len(train_loader),
        epochs=70,
        pct_start=0.2,  # Warm up for 20% of training
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    return lr_scheduler

scheduler = get_lr_scheduler(optimizer)

# Mixed precision training with compatible syntax
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Training and validation functions
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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

# Training loop
num_epochs = 70
best_val_acc = 0.0
train_losses, val_losses, test_losses = [], [], []
train_accs, val_accs, test_accs = [], [], []

for epoch in range(num_epochs):
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

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Update learning rate
            scheduler.step()

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
                                                optimizer, scheduler, device, scaler)

    # Validation
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    # Test - added test evaluation for each epoch
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), './output/model_checkpoints/best_model.pth')
        print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    # Save model checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f'./output/model_checkpoints/model_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc
        }, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")
        
        # Plot and save training progress every 10 epochs
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
        plt.savefig(f'./output/plots/training_metrics_epoch_{epoch+1}.png')
        plt.close()

# Load best model and evaluate on test set
model.load_state_dict(torch.load('./output/model_checkpoints/best_model.pth'))
test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
print(f'Best Model Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# Plot final training metrics
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
plt.savefig('./output/plots/final_training_metrics.png')
plt.close()

# Plot confusion matrix for test set
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise'], 
            yticklabels=['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise'])
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('./output/plots/confusion_matrix_test.png')
plt.close()

print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
print(f'Final Test Accuracy: {test_acc:.2f}%')

# Evaluate class-wise performance
class_accuracy = {}
for i, class_name in enumerate(['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']):
    class_indices = [j for j, label in enumerate(test_labels) if label == i]
    if class_indices:
        class_preds = [test_preds[j] for j in class_indices]
        class_true = [test_labels[j] for j in class_indices]
        acc = accuracy_score(class_true, class_preds) * 100
        class_accuracy[class_name] = acc
        print(f"{class_name} Class Accuracy: {acc:.2f}%")

# Save final model
torch.save(model.state_dict(), './output/model_checkpoints/final_model.pth')
print("Final model saved to ./output/model_checkpoints/final_model.pth")