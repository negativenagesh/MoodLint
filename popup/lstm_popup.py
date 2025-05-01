#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
import traceback
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import signal
import torchvision.models.resnet

# Set a timeout handler to prevent hanging
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

# Check if we have enough arguments
if len(sys.argv) < 2:
    print(json.dumps({"error": "Missing image directory. Usage: lstm_popup.py <image_directory>"}), flush=True)
    sys.exit(1)

# Get the image directory from arguments
image_dir = sys.argv[1]
if not os.path.exists(image_dir):
    print(json.dumps({"error": f"Image directory not found: {image_dir}"}), flush=True)
    sys.exit(1)

# Configuration parameters
IMG_SIZE = 128
FEATURE_DIM = 512
SEQUENCE_LENGTH = 5
CLASS_NAMES = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Device configuration - force CPU for compatibility
device = torch.device("cpu")
print(json.dumps({"info": f"Using device: {device}"}), flush=True)

# Define the model classes BEFORE loading - using exact same architecture as training
class FeatureExtractor(nn.Module):
    def __init__(self, output_dim=FEATURE_DIM):
        super(FeatureExtractor, self).__init__()
        # Load pre-trained ResNet18 - use IMAGENET1K_V1 to match training
        try:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(json.dumps({"warning": f"Could not load with IMAGENET1K_V1: {str(e)}. Trying DEFAULT."}), flush=True)
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
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

class MoodLSTM(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=128, num_layers=1, dropout=0.3, num_classes=5):
        super(MoodLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # LSTM expects input shape (batch_size, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout and pass through the fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        return out

# NOW register safe globals AFTER the classes are defined
try:
    import torch.serialization
    
    # Add all potentially needed classes to safe globals
    torch.serialization.add_safe_globals([
        FeatureExtractor,  
        MoodLSTM,          
        torchvision.models.resnet.ResNet,
        nn.Module,
        nn.Sequential,
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.LSTM,
        nn.Flatten
    ])
    print(json.dumps({"info": "Added model classes to safe globals globally"}), flush=True)
except (ImportError, AttributeError) as e:
    print(json.dumps({"warning": f"Could not add global safe globals: {str(e)}"}), flush=True)

# Rest of your functions
def predict_next_mood(image_paths, feature_extractor, lstm_model, sequence_length=SEQUENCE_LENGTH):
    """
    Predict the next mood based on a sequence of facial expression images
    """
    # [function body remains the same]

def load_model_with_fallback(model_path, model_class, device):
    """
    Load a model with multiple fallback strategies
    """
    # [function body remains the same]

def main():
    # [function body remains the same]

# Single if __name__ block at the end
    if __name__ == "__main__":
        main()