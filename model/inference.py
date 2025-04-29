import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# Define the same model architecture used in training
class ExpressionRecognitionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(ExpressionRecognitionModel, self).__init__()
        # Load pre-trained ResNet50 backbone
        self.backbone = models.resnet50()

        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Attention mechanism for focusing on important facial features
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
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

def preprocess_image(image_path):
    """Preprocess the input image for model inference"""
    try:
        # Define transforms - match the preprocessing used in training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and transform image
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            input_tensor = transform(img)
            # Add batch dimension
            input_tensor = input_tensor.unsqueeze(0)
            return input_tensor
    except Exception as e:
        print(json.dumps({"error": f"Error preprocessing image: {str(e)}"}), flush=True)
        return None

def load_model(model_path):
    """Load the model from the checkpoint"""
    try:
        # Set device
        device = torch.device("cpu")  # For compatibility, use CPU
        
        # Initialize model
        model = ExpressionRecognitionModel(num_classes=5)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.eval()
        
        return model, device
    except Exception as e:
        print(json.dumps({"error": f"Error loading model: {str(e)}"}), flush=True)
        return None, None

def predict_mood(model, image_tensor, device):
    """Run the model to predict mood from image"""
    try:
        # Transfer tensor to device
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Map index to emotion label - use correct labels from training
            emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
            predicted_emotion = emotion_labels[predicted.item()]
            
            return predicted_emotion, confidence.item()
    
    except Exception as e:
        print(json.dumps({"error": f"Error during prediction: {str(e)}"}), flush=True)
        return None, 0.0

def main():
    """Main function to process the image and detect mood"""
    try:
        # Get arguments
        if len(sys.argv) < 2:
            print(json.dumps({"error": "Missing image path argument"}), flush=True)
            return
            
        image_path = sys.argv[1]
        
        # Get model path from command line or use default
        if len(sys.argv) >= 3:
            model_path = sys.argv[2]
        else:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "model_epoch_40.pth")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(json.dumps({"error": f"Model file not found at {model_path}"}), flush=True)
            return
            
        # Load and preprocess image
        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            return
            
        # Load model
        model, device = load_model(model_path)
        if model is None:
            return
            
        # Make prediction
        mood, confidence = predict_mood(model, image_tensor, device)
        if mood is None:
            return
            
        # Return prediction result as JSON
        print(json.dumps({"mood": mood, "confidence": confidence}), flush=True)
        
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {str(e)}"}), flush=True)

if __name__ == "__main__":
    main()