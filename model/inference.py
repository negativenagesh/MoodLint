import os
import sys
import json

# Check for required packages first
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, models
except ImportError as e:
    missing_package = str(e).split("'")[1] if "'" in str(e) else str(e).split(" ")[-1]
    print(json.dumps({
        "error": f"Missing Python package: {missing_package}. Please install using: pip install {missing_package}"
    }), flush=True)
    sys.exit(1)

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
        # Verify file exists
        if not os.path.exists(image_path):
            print(json.dumps({"error": f"Image file not found: {image_path}"}), flush=True)
            return None
            
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
        # Verify file exists
        if not os.path.exists(model_path):
            # Try to find the model in alternative locations
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alt_paths = [
                os.path.join(base_dir, "model-training-outputs", "model_checkpoints", "final_model.pth"),
                os.path.join(base_dir, "model_checkpoints", "final_model.pth"),
                os.path.join(base_dir, "model", "final_model.pth"),
                os.path.join(base_dir, "final_model.pth")
            ]
            
            found = False
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    found = True
                    print(json.dumps({"info": f"Found model at alternate location: {model_path}"}), flush=True)
                    break
                    
            if not found:
                # If empty file, create a stub for debugging
                print(json.dumps({"error": f"Model file not found: {model_path}"}), flush=True)
                print(json.dumps({"error": "Creating fallback mode with random predictions"}), flush=True)
                return None, None
            
        # Set device - use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(json.dumps({"info": f"Using device: {device}"}), flush=True)
        
        # Initialize model
        model = ExpressionRecognitionModel(num_classes=5)
        
        # Load weights
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            # Set to evaluation mode
            model.eval()
            
            # Move model to device
            model = model.to(device)
            
            return model, device
            
        except Exception as load_error:
            print(json.dumps({"error": f"Error loading model weights: {str(load_error)}"}), flush=True)
            return None, device  # Return device for fallback mode
            
    except Exception as e:
        print(json.dumps({"error": f"Error in load_model: {str(e)}"}), flush=True)
        return None, None

def predict_mood(model, image_tensor, device):
    """Run the model to predict mood from image"""
    try:
        # Check if we're in fallback mode
        if model is None:
            print(json.dumps({"warning": "Using fallback prediction mode"}), flush=True)
            
            # Return a fallback prediction (random but with fixed seed for consistency)
            import random
            random.seed(42)  # Fixed seed for reproducibility
            emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
            confidence = random.uniform(0.6, 0.8)  # Reasonable confidence
            emotion_idx = random.randint(0, len(emotions)-1)
            predicted_emotion = emotions[emotion_idx]
            
            # Create probabilities
            all_probabilities = {}
            for i, emotion in enumerate(emotions):
                if i == emotion_idx:
                    all_probabilities[emotion] = confidence
                else:
                    all_probabilities[emotion] = (1.0 - confidence) / (len(emotions) - 1)
                    
            return predicted_emotion, confidence, all_probabilities
            
        # Transfer tensor to device
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get confidence scores for all classes
            probs_array = probabilities[0].cpu().numpy()
            
            # Get top prediction
            confidence, predicted = torch.max(probabilities, 1)
            
            # Map index to emotion label - use correct labels from training
            emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
            predicted_emotion = emotion_labels[predicted.item()]
            
            # Create probability dictionary for all emotions
            all_probabilities = {emotion: float(probs_array[i]) for i, emotion in enumerate(emotion_labels)}
            
            return predicted_emotion, confidence.item(), all_probabilities
    
    except Exception as e:
        print(json.dumps({"error": f"Error during prediction: {str(e)}"}), flush=True)
        return None, 0.0, {}

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
            # Use the specified model path as default
            model_path = "/home/subrahmanya/projects/MoodLint/model-training-outputs/model_checkpoints/final_model.pth"
        
        # Load and preprocess image
        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            # Create a fallback response
            print(json.dumps({
                "mood": "Neutral", 
                "agent_mood": "happy", 
                "confidence": 0.7,
                "all_probabilities": {
                    "Angry": 0.05, "Happy": 0.15, "Neutral": 0.7, "Sad": 0.05, "Surprise": 0.05
                },
                "fallback": True,
                "model_path": model_path,
                "device": "fallback"
            }), flush=True)
            return
            
        # Load model
        model, device = load_model(model_path)
            
        # Make prediction
        mood, confidence, all_probabilities = predict_mood(model, image_tensor, device)
        if mood is None:
            # Create a fallback response
            print(json.dumps({
                "mood": "Neutral", 
                "agent_mood": "happy", 
                "confidence": 0.7,
                "all_probabilities": {
                    "Angry": 0.05, "Happy": 0.15, "Neutral": 0.7, "Sad": 0.05, "Surprise": 0.05
                },
                "fallback": True,
                "model_path": model_path,
                "device": str(device) if device else "unknown"
            }), flush=True)
            return
        
        # Map model output to agent-compatible mood
        mood_mapping = {
            "Angry": "angry",
            "Happy": "happy",
            "Sad": "sad",
            "Neutral": "happy",  # Map neutral to happy for agent compatibility
            "Surprise": "happy"  # Map surprise to happy for agent compatibility
        }
        
        agent_mood = mood_mapping.get(mood, "happy").lower()
        
        # Return prediction result as JSON with all information needed for agent selection
        result = {
            "mood": mood,  # Original model output
            "agent_mood": agent_mood,  # Mapped to agent format
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "model_path": model_path,
            "device": str(device)
        }
        
        print(json.dumps(result), flush=True)
        
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {str(e)}"}), flush=True)
        # Fall back to a default response so the agent popup will still work
        print(json.dumps({
            "mood": "Neutral", 
            "agent_mood": "happy", 
            "confidence": 0.7,
            "all_probabilities": {
                "Angry": 0.05, "Happy": 0.15, "Neutral": 0.7, "Sad": 0.05, "Surprise": 0.05
            },
            "fallback": True,
            "error_info": str(e)
        }), flush=True)

if __name__ == "__main__":
    main()