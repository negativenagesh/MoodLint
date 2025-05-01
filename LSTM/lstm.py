import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Configuration parameters (optimized for Kaggle's 16GB GPU)
IMG_SIZE = 128
BATCH_SIZE = 128  # Increased batch size for 16GB GPU
EPOCHS = 30
SEQUENCE_LENGTH = 5
NUM_CLASSES = 5  # angry, happy, sad, surprise, neutral
LEARNING_RATE = 0.0005
FEATURE_DIM = 512  # ResNet18 feature dimension

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9} GB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9} GB")

# Dataset paths (for Kaggle) - MODIFIED: more flexible path detection
BASE_PATH = "/kaggle/input"
DATA_PATH = None

# Check for standard dataset locations in Kaggle
possible_paths = [
    "/kaggle/input/expressions",
    "/kaggle/input/facial-expressions", 
    "/kaggle/input/fer2013",
    "/kaggle/input/facial-expression-dataset"
]

# Find the first valid path
for path in possible_paths:
    if os.path.exists(path):
        DATA_PATH = path
        print(f"Found dataset at: {DATA_PATH}")
        break

# If not found in standard locations, search for any folder with expression-related content
if DATA_PATH is None:
    print("Searching for dataset in input directory...")
    for root, dirs, _ in os.walk(BASE_PATH):
        for dir_name in dirs:
            if any(keyword in dir_name.lower() for keyword in ['express', 'emotion', 'facial', 'face']):
                possible_path = os.path.join(root, dir_name)
                print(f"Found potential dataset directory: {possible_path}")
                DATA_PATH = possible_path
                break
        if DATA_PATH:
            break

# If still not found, use the base path
if DATA_PATH is None:
    DATA_PATH = BASE_PATH
    print(f"Using base path as data path: {DATA_PATH}")

# Output directory
OUTPUT_PATH = "/kaggle/working/model_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Class mapping
CLASS_NAMES = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']
CLASS_MAP = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}

# Image transformations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Added slight rotation for augmentation
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Added color jitter for augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset class with enhanced error handling and flexible directory structure support
class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None, recursive_search=True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Print directory contents for debugging
        print(f"Contents of {root_dir}: {os.listdir(root_dir)}")
        
        # Determine directory structure type
        if self._is_flat_directory():
            print("Detected flat directory structure with labels in filenames")
            self._load_flat_directory()
        else:
            # Check for hierarchical directory structure
            found_classes = False
            # Try direct class directories
            for class_name in CLASS_NAMES:
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    found_classes = True
                    print(f"Found class directory: {class_name}")
                    class_idx = CLASS_MAP[class_name]
                    self._add_images_from_dir(class_dir, class_idx)
            
            # If no standard classes found and recursive search enabled, look for class directories recursively
            if not found_classes and recursive_search:
                print("Standard class directories not found, searching recursively...")
                self._recursive_search(root_dir)
        
        # If still no samples, look for any image files and use dummy labels
        if len(self.samples) == 0:
            print("No structured dataset found, collecting all images with dummy labels...")
            self._collect_all_images()
        
        print(f"Total samples found: {len(self.samples)}")
    
    def _is_flat_directory(self):
        """Check if dataset has a flat directory structure with labels in filenames"""
        # Look for a few files to check naming pattern
        files = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]
        if len(files) > 0:
            # Check if filenames contain emotion labels
            sample_files = files[:min(10, len(files))]
            emotion_keywords = [c.lower() for c in CLASS_NAMES]
            has_emotion_in_name = any(any(keyword in f.lower() for keyword in emotion_keywords) for f in sample_files)
            if has_emotion_in_name:
                return True
        return False
    
    def _load_flat_directory(self):
        """Load dataset with flat structure (labels in filenames)"""
        files = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f)) 
                and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for file in files:
            filepath = os.path.join(self.root_dir, file)
            # Try to detect class from filename
            detected_class = None
            for class_name in CLASS_NAMES:
                if class_name.lower() in file.lower():
                    detected_class = class_name
                    break
            
            # If class detected, add to samples
            if detected_class:
                class_idx = CLASS_MAP[detected_class]
                self.samples.append((filepath, class_idx))
            
        print(f"Loaded {len(self.samples)} images from flat directory")
    
    def _add_images_from_dir(self, directory, class_idx):
        """Add all valid images from a directory with known class index"""
        file_count = 0
        for img_name in os.listdir(directory):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append((os.path.join(directory, img_name), class_idx))
                file_count += 1
        print(f"Found {file_count} files in {os.path.basename(directory)}")
    
    def _recursive_search(self, directory):
        """Recursively search for class directories"""
        for item in os.listdir(directory):
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                # Check if directory name matches a class
                dir_name = os.path.basename(path)
                for class_name in CLASS_NAMES:
                    if class_name.lower() in dir_name.lower():
                        print(f"Found class directory: {path}")
                        self._add_images_from_dir(path, CLASS_MAP[class_name])
                        break
                else:
                    # Not a class directory, search deeper
                    self._recursive_search(path)
    
    def _collect_all_images(self):
        """Collect all images in the directory structure with dummy labels"""
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Assign a random class (0) for demonstration
                    self.samples.append((os.path.join(root, file), 0))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # Open image and convert to RGB to ensure 3 channels
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image and the label if there's an error
            if self.transform:
                dummy = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            else:
                dummy = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            return dummy, label

# CNN Feature Extractor - using ResNet18
class FeatureExtractor(nn.Module):
    def __init__(self, output_dim=FEATURE_DIM):
        super(FeatureExtractor, self).__init__()
        # Load pre-trained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add new layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

# LSTM Model
class MoodLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.3, num_classes=NUM_CLASSES):
        super(MoodLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0  # Only used if num_layers > 1
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout and final layer
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        
        return out

# Sequence Dataset class
class SequenceDataset(Dataset):
    def __init__(self, features, labels, seq_length=SEQUENCE_LENGTH):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        
        # Create sequences and corresponding labels
        self.sequences = []
        self.sequence_labels = []
        
        for i in range(len(features) - seq_length):
            self.sequences.append(features[i:i+seq_length])
            self.sequence_labels.append(labels[i+seq_length])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), self.sequence_labels[idx]

# Function to extract features from all images - with caching
def extract_features(dataset, feature_extractor, batch_size=64, cache_file=None):
    # Check if cache exists
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data['features'], data['labels']
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    feature_extractor.eval()
    feature_extractor.to(device)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device, non_blocking=True)
            with autocast():  # Use mixed precision
                features = feature_extractor(images).cpu().numpy()
            
            all_features.extend(features)
            all_labels.extend(labels.numpy())
    
    # Cache the features
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez(cache_file, features=np.array(all_features), labels=np.array(all_labels))
    
    return np.array(all_features), np.array(all_labels)

# Training function - with mixed precision
def train(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(train_loader, desc="Training", leave=False):
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Zero the parameter gradients
        optimizer.zero_grad(set_to_none=True)  # Slightly faster than setting to zero
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(sequences)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc="Validating", leave=False):
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(sequences)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# Main execution function
def main():
    print("Starting facial expression classification training pipeline...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    
    # Check CUDA availability and memory
    if torch.cuda.is_available():
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Maximum GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("Loading datasets...")
    
    # Load the facial expression dataset with error handling and recursive search
    try:
        full_dataset = FacialExpressionDataset(DATA_PATH, transform=val_transform, recursive_search=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Checking if dataset directory exists...")
        if os.path.exists(DATA_PATH):
            print(f"Directory exists. Contents: {os.listdir(DATA_PATH)}")
        else:
            print(f"Directory does not exist: {DATA_PATH}")
            return
    
    # Check if dataset is empty
    if len(full_dataset) == 0:
        print(f"ERROR: No images found in {DATA_PATH}")
        print("Please verify that the directory structure is:")
        for class_name in CLASS_NAMES:
            print(f"  - {DATA_PATH}/{class_name}/")
        print("Each folder should contain image files (.jpg, .jpeg, or .png)")
        
        # Simulate a small dataset for testing if in debug mode
        print("Creating a simulated dataset for testing...")
        # Create 50 dummy samples, 10 for each class
        dummy_samples = []
        for class_idx in range(NUM_CLASSES):
            dummy_tensor = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            for _ in range(10):
                dummy_samples.append((dummy_tensor, class_idx))
        
        class DummyDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        full_dataset = DummyDataset(dummy_samples)
        print(f"Created dummy dataset with {len(full_dataset)} samples")
    
    print(f"Found {len(full_dataset)} images across {len(set([label for _, label in full_dataset.samples]))} classes")
    
    # Class distribution analysis
    class_counts = {}
    for _, label in full_dataset:
        class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"Unknown-{label}"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images ({count/len(full_dataset)*100:.2f}%)")
    
    # Split the dataset into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2,
        stratify=[full_dataset[i][1] for i in range(len(full_dataset))],
        random_state=SEED
    )
    
    # Create train and validation subsets with appropriate transforms
    train_dataset = Subset(FacialExpressionDataset(DATA_PATH, transform=train_transform, recursive_search=True), train_indices)
    val_dataset = Subset(FacialExpressionDataset(DATA_PATH, transform=val_transform, recursive_search=True), val_indices)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create feature extractor model
    print("Creating feature extractor...")
    feature_extractor = FeatureExtractor(output_dim=FEATURE_DIM)
    
    # First, train the feature extractor as a classifier to learn facial expressions
    print("Training feature extractor...")
    
    # Add a classification head for initial training
    classifier = nn.Sequential(
        feature_extractor,
        nn.Linear(FEATURE_DIM, NUM_CLASSES)
    ).to(device)
    
    # Train the classifier with mixed precision
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for better stability
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    scaler = GradScaler()  # For mixed precision training
    
    best_val_loss = float('inf')
    cnn_train_losses, cnn_val_losses = [], []
    cnn_train_accs, cnn_val_accs = [], []
    
    # Training loop for feature extractor
    for epoch in range(10):  # Reduced epochs for testing
        print(f"Epoch {epoch+1}/10")
        
        # Train for one epoch
        train_loss, train_acc = train(classifier, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc, _, _ = validate(classifier, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(feature_extractor.state_dict(), os.path.join(OUTPUT_PATH, 'feature_extractor_best.pth'))
            # Also save the full model for easier loading
            torch.save(feature_extractor, os.path.join(OUTPUT_PATH, 'feature_extractor_full.pth'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Record metrics
        cnn_train_losses.append(train_loss)
        cnn_val_losses.append(val_loss)
        cnn_train_accs.append(train_acc)
        cnn_val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Plot CNN training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(cnn_train_accs, label='Train Accuracy')
    plt.plot(cnn_val_accs, label='Validation Accuracy')
    plt.title('CNN Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(cnn_train_losses, label='Train Loss')
    plt.plot(cnn_val_losses, label='Validation Loss')
    plt.title('CNN Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'cnn_training_history.png'))
    plt.close()
    
    # Load the best feature extractor model
    try:
        feature_extractor.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, 'feature_extractor_best.pth')))
        print("Loaded best feature extractor model")
    except Exception as e:
        print(f"Error loading best model: {e}. Continuing with last model.")
    
    # Extract features from all images with caching
    print("Extracting features from images...")
    train_cache = os.path.join(OUTPUT_PATH, 'train_features.npz')
    val_cache = os.path.join(OUTPUT_PATH, 'val_features.npz')
    
    train_features, train_labels = extract_features(
        train_dataset, feature_extractor, batch_size=BATCH_SIZE, cache_file=train_cache
    )
    
    val_features, val_labels = extract_features(
        val_dataset, feature_extractor, batch_size=BATCH_SIZE, cache_file=val_cache
    )
    
    print(f"Extracted {len(train_features)} training features and {len(val_features)} validation features")
    
    # Create sequence datasets
    train_seq_dataset = SequenceDataset(train_features, train_labels, SEQUENCE_LENGTH)
    val_seq_dataset = SequenceDataset(val_features, val_labels, SEQUENCE_LENGTH)
    
    print(f"Created {len(train_seq_dataset)} training sequences and {len(val_seq_dataset)} validation sequences")
    
    # Create data loaders for sequences
    train_seq_loader = DataLoader(
        train_seq_dataset, 
        batch_size=BATCH_SIZE * 2,  # Larger batch size for sequence training
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        drop_last=True  # Prevent issues with small batches
    )
    
    val_seq_loader = DataLoader(
        val_seq_dataset, 
        batch_size=BATCH_SIZE * 2, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    # Create LSTM model
    print("Creating LSTM model...")
    lstm_model = MoodLSTM(
        input_dim=FEATURE_DIM,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        num_classes=NUM_CLASSES
    ).to(device)
    
    # Training setup for LSTM
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    scaler = GradScaler()  # For mixed precision training
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    lstm_train_losses, lstm_val_losses = [], []
    lstm_train_accs, lstm_val_accs = [], []
    early_stopping_counter = 0
    early_stopping_patience = 5
    
    # Training loop for LSTM
    print("Training LSTM model...")
    num_epochs = min(10, EPOCHS)  # Reduced epochs for testing
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train(lstm_model, train_seq_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(lstm_model, val_seq_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(lstm_model.state_dict(), os.path.join(OUTPUT_PATH, 'lstm_model_best.pth'))
            # Also save full model for easier loading
            torch.save(lstm_model, os.path.join(OUTPUT_PATH, 'lstm_model_full.pth'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Record metrics
        lstm_train_losses.append(train_loss)
        lstm_val_losses.append(val_loss)
        lstm_train_accs.append(train_acc)
        lstm_val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Check GPU memory
        if torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Plot LSTM training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(lstm_train_accs, label='Train Accuracy')
    plt.plot(lstm_val_accs, label='Validation Accuracy')
    plt.title('LSTM Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(lstm_train_losses, label='Train Loss')
    plt.plot(lstm_val_losses, label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'lstm_training_history.png'))
    plt.close()
    
    # Try to load the best LSTM model
    try:
        lstm_model.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, 'lstm_model_best.pth')))
        print("Loaded best LSTM model")
    except Exception as e:
        print(f"Error loading best LSTM model: {e}. Continuing with last model.")
    
    # Final evaluation
    _, _, final_preds, final_labels = validate(lstm_model, val_seq_loader, criterion, device)
    
    # Generate classification report
    print("Classification Report:")
    report = classification_report(final_labels, final_preds, target_names=CLASS_NAMES)
    print(report)
    
    # Save the report to a file
    with open(os.path.join(OUTPUT_PATH, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix.png'))
    plt.close()
    
    print("Training completed! Models saved.")
    
    # Save the complete models for inference
    torch.save(feature_extractor, os.path.join(OUTPUT_PATH, 'feature_extractor_full.pth'))
    torch.save(lstm_model, os.path.join(OUTPUT_PATH, 'lstm_model_full.pth'))

# Function for predicting on new sequences
def predict_next_mood(image_paths, feature_extractor, lstm_model, sequence_length=SEQUENCE_LENGTH):
    """
    Predict the next mood based on a sequence of facial expression images
    
    Args:
        image_paths: List of paths to facial expression images (in sequence order)
        feature_extractor: The trained feature extractor model
        lstm_model: The trained LSTM model
        sequence_length: Length of sequence expected by the LSTM model
        
    Returns:
        predicted_class: The predicted mood class
        probability: The probability of the prediction
    """
    # Check if we have enough images
    if len(image_paths) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} images, but only {len(image_paths)} provided")
    
    # Use only the most recent images if more are provided
    if len(image_paths) > sequence_length:
        image_paths = image_paths[-sequence_length:]
    
    # Set models to evaluation mode
    feature_extractor.eval()
    lstm_model.eval()
    
    # Move models to device
    feature_extractor.to(device)
    lstm_model.to(device)
    
    # Load and preprocess images
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Extract features
    features = []
    with torch.no_grad():
        for img_path in image_paths:
            try:
                # Load and preprocess image and convert to RGB
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Extract features with mixed precision
                with autocast():
                    feature = feature_extractor(img_tensor).cpu().numpy()[0]
                features.append(feature)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # Use zeros as features for problematic images
                features.append(np.zeros(FEATURE_DIM))
    
    # Create sequence tensor
    sequence = torch.tensor(np.array([features]), dtype=torch.float32).to(device)
    
    # Make prediction
    with torch.no_grad():
        with autocast():
            prediction = lstm_model(sequence)
        probabilities = F.softmax(prediction, dim=1)
        
        pred_class = torch.argmax(probabilities, dim=1).item()
        probability = probabilities[0][pred_class].item()
    
    return CLASS_NAMES[pred_class], probability

# Example usage of the prediction function with robust error handling
def test_prediction():
    # Check if models exist before loading
    model_files = [
        os.path.join(OUTPUT_PATH, 'feature_extractor_full.pth'),
        os.path.join(OUTPUT_PATH, 'lstm_model_full.pth')
    ]
    
    # Verify models exist
    missing_models = [f for f in model_files if not os.path.exists(f)]
    if missing_models:
        print(f"Warning: The following model files are missing: {missing_models}")
        print("Cannot run prediction test without trained models.")
        print("This is expected if the training process didn't complete.")
        return
    
    try:
        # Load models with weights_only=True to avoid security warning
        print("Loading feature extractor model...")
        feature_extractor = torch.load(model_files[0], weights_only=True)
        print("Loading LSTM model...")
        lstm_model = torch.load(model_files[1], weights_only=True)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Try to find test images in the dataset
    test_images = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_PATH, class_name)
        if os.path.isdir(class_dir):
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                # Take up to 5 files from this class
                test_images.extend(files[:min(5, len(files))])
                if len(test_images) >= 5:
                    test_images = test_images[:5]  # Take only 5 images
                    break
    
    # If not enough images found in standard structure, search recursively
    if len(test_images) < 5:
        print(f"Warning: Could only find {len(test_images)} test images in standard directories")
        print("Searching for images recursively...")
        
        for root, _, files in os.walk(DATA_PATH):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(root, file))
                    if len(test_images) >= 5:
                        break
            if len(test_images) >= 5:
                break
    
    # If still not enough images, create dummy images
    if len(test_images) < 5:
        print(f"Warning: Could only find {len(test_images)} test images, need 5 for a sequence")
        print("Creating dummy test images...")
        
        dummy_dir = os.path.join(OUTPUT_PATH, 'dummy_images')
        os.makedirs(dummy_dir, exist_ok=True)
        
        for i in range(5 - len(test_images)):
            # Create a black image
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            img_path = os.path.join(dummy_dir, f'dummy_{i}.jpg')
            img.save(img_path)
            test_images.append(img_path)
    
    print(f"Using test images: {test_images}")
    
    # Predict next mood
    try:
        predicted_mood, confidence = predict_next_mood(
            test_images, 
            feature_extractor, 
            lstm_model
        )
        
        print(f"Predicted next mood: {predicted_mood} with confidence {confidence:.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
    
    # Only test prediction if models were successfully created
    model_files_exist = os.path.exists(os.path.join(OUTPUT_PATH, 'feature_extractor_full.pth'))
    
    # Test prediction with sample images from the dataset
    if model_files_exist:
        print("\nTesting prediction function...")
        test_prediction()
    else:
        print("\nSkipping prediction test as model files were not created")