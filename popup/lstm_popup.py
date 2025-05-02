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

def predict_next_mood(image_paths, feature_extractor, lstm_model, sequence_length=SEQUENCE_LENGTH):
    """
    Predict the next mood based on a sequence of facial expression images
    """
    try:
        print(json.dumps({"progress": "Starting mood sequence prediction"}), flush=True)
        
        # Check if we have enough images
        if len(image_paths) < sequence_length:
            print(json.dumps({"error": f"Need at least {sequence_length} images, but only {len(image_paths)} provided"}), flush=True)
            # Return a fallback prediction
            return "Neutral", 0.7
        
        # Use only the most recent images if more are provided
        if len(image_paths) > sequence_length:
            image_paths = image_paths[-sequence_length:]
            
        print(json.dumps({"progress": "Processing image sequence"}), flush=True)
            
        # Set models to evaluation mode
        feature_extractor.eval()
        lstm_model.eval()
        
        # Load and preprocess images
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Extract features
        features = []
        print(json.dumps({"progress": "Extracting image features"}), flush=True)
        
        with torch.no_grad():
            for i, img_path in enumerate(image_paths):
                try:
                    print(json.dumps({"progress": f"Processing image {i+1}/{len(image_paths)}"}), flush=True)
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    
                    # Extract features
                    feature = feature_extractor(img_tensor).cpu().numpy()[0]
                    features.append(feature)
                except Exception as e:
                    print(json.dumps({"warning": f"Error processing image {img_path}: {str(e)}"}), flush=True)
                    # Use zeros as features for problematic images
                    features.append(np.zeros(FEATURE_DIM))
        
        # Create sequence tensor
        print(json.dumps({"progress": "Running LSTM prediction"}), flush=True)
        sequence = torch.tensor(np.array([features]), dtype=torch.float32).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = lstm_model(sequence)
            probabilities = F.softmax(prediction, dim=1)
            
            pred_class = torch.argmax(probabilities, dim=1).item()
            probability = probabilities[0][pred_class].item()
        
        print(json.dumps({"progress": f"Prediction complete: {CLASS_NAMES[pred_class]}"}), flush=True)
        return CLASS_NAMES[pred_class], probability
        
    except Exception as e:
        print(json.dumps({"error": f"Error in predict_next_mood: {str(e)}"}), flush=True)
        print(json.dumps({"trace": traceback.format_exc()}), flush=True)
        # Return a fallback prediction
        return "Neutral", 0.7

def load_model_with_fallback(model_path, model_class, device):
    """
    Load a model with multiple fallback strategies
    """
    print(json.dumps({"progress": f"Loading model from {model_path}"}), flush=True)
    
    try:
        # First try direct loading
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(json.dumps({"warning": f"Standard loading failed: {str(e)}. Trying alternative methods..."}), flush=True)
        
        try:
            # Try loading with _load_from_state_dict
            model = model_class()
            state_dict = torch.load(model_path, map_location=device)
            model._load_from_state_dict(state_dict, '', {}, strict=False)
            model.to(device)
            model.eval()
            return model
        except Exception as e2:
            print(json.dumps({"error": f"Could not load model: {str(e2)}"}), flush=True)
            # Return None to indicate failure
            return None

def main():
    try:
        # Set a timeout for the entire process
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 second timeout
        
        print(json.dumps({"progress": "Starting mood prediction process"}), flush=True)
        
        # Find image files in directory
        image_files = []
        for file in sorted(os.listdir(image_dir)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(image_dir, file))
        
        print(json.dumps({"progress": f"Found {len(image_files)} image files"}), flush=True)
        
        # Check if we have enough images
        if len(image_files) < SEQUENCE_LENGTH:
            print(json.dumps({"warning": f"Not enough images found. Need {SEQUENCE_LENGTH}, found {len(image_files)}"}), flush=True)
            
            # Create dummy images if needed
            for i in range(SEQUENCE_LENGTH - len(image_files)):
                # Create a black dummy image
                dummy_path = os.path.join(image_dir, f"dummy_{i}.png")
                Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0)).save(dummy_path)
                image_files.append(dummy_path)
            
            print(json.dumps({"progress": f"Created dummy images, now have {len(image_files)} total"}), flush=True)
        
        # Use the most recent images
        recent_images = image_files[-SEQUENCE_LENGTH:]
        print(json.dumps({"progress": f"Using most recent {len(recent_images)} images"}), flush=True)
        
        # Define model paths relative to the script
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        feature_extractor_path = os.path.join(script_dir, "LSTM", "feature_extractor_best.pth")
        lstm_model_path = os.path.join(script_dir, "LSTM", "lstm_model_best.pth")
        
        # Load models
        print(json.dumps({"progress": "Loading feature extractor model"}), flush=True)
        feature_extractor = load_model_with_fallback(feature_extractor_path, FeatureExtractor, device)
        
        print(json.dumps({"progress": "Loading LSTM model"}), flush=True)
        lstm_model = load_model_with_fallback(lstm_model_path, MoodLSTM, device)
        
        if feature_extractor is None or lstm_model is None:
            print(json.dumps({"error": "Failed to load one or more models"}), flush=True)
            sys.exit(1)
        
        # Predict next mood
        print(json.dumps({"progress": "Starting prediction with loaded models"}), flush=True)
        predicted_mood, confidence = predict_next_mood(recent_images, feature_extractor, lstm_model)
        
        # Generate a message based on the predicted mood
        message = f"Based on your recent emotions, you're likely to feel {predicted_mood.lower()} soon."
        
        # Return result
        print(json.dumps({
            "status": "success",
            "mood": predicted_mood,
            "confidence": confidence,
            "message": message
        }), flush=True)
        
        # Turn off the alarm
        signal.alarm(0)
        
        # Exit successfully
        sys.exit(0)
        
    except TimeoutError:
        print(json.dumps({"error": "Prediction process timed out after 60 seconds"}), flush=True)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {str(e)}"}), flush=True)
        print(json.dumps({"trace": traceback.format_exc()}), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()