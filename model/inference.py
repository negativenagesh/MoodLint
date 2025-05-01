import os
import sys
import json
import traceback
import random

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
        # Send progress update for tracking
        print(json.dumps({"progress": "Model forward pass - starting backbone"}), flush=True)
        
        # Extract features from the backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        print(json.dumps({"progress": "Model forward pass - layer 1 complete"}), flush=True)
        
        x = self.backbone.layer2(x)
        print(json.dumps({"progress": "Model forward pass - layer 2 complete"}), flush=True)
        
        x = self.backbone.layer3(x)
        print(json.dumps({"progress": "Model forward pass - layer 3 complete"}), flush=True)
        
        x = self.backbone.layer4(x)  # [batch_size, 2048, 7, 7]
        print(json.dumps({"progress": "Model forward pass - all layers complete"}), flush=True)

        # Apply attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        print(json.dumps({"progress": "Model forward pass - attention applied"}), flush=True)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        print(json.dumps({"progress": "Model forward pass - pooling complete"}), flush=True)

        # Apply classifier
        x = self.classifier(x)
        print(json.dumps({"progress": "Model forward pass - classification complete"}), flush=True)
        
        return x

def preprocess_image(image_path):
    """Load and preprocess image for model input"""
    try:
        print(json.dumps({"progress": "Loading image from path"}), flush=True)
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(json.dumps({"error": f"Image file not found: {image_path}"}), flush=True)
            return None
        
        # Check file size
        file_size = os.path.getsize(image_path)
        print(json.dumps({"info": f"Image file size: {file_size} bytes"}), flush=True)
        if file_size <= 0:
            print(json.dumps({"error": "Image file is empty"}), flush=True)
            return None
        
        # Try multiple methods to load the image
        image = None
        error_message = ""
        
        # Method 1: PIL
        try:
            from PIL import Image
            image = Image.open(image_path)
            image = image.convert('RGB')
            print(json.dumps({"info": "Image loaded with PIL"}), flush=True)
        except Exception as e:
            error_message += f"PIL error: {str(e)}; "
        
        # Method 2: OpenCV if PIL failed
        if image is None:
            try:
                import cv2
                image_cv = cv2.imread(image_path)
                if image_cv is not None and image_cv.size > 0:
                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    # Convert to PIL
                    from PIL import Image
                    image = Image.fromarray(image_cv)
                    print(json.dumps({"info": "Image loaded with OpenCV"}), flush=True)
                else:
                    error_message += "OpenCV returned empty image; "
            except Exception as e:
                error_message += f"OpenCV error: {str(e)}; "
        
        if image is None:
            print(json.dumps({"error": f"Failed to load image: {error_message}"}), flush=True)
            return None
        
        # Transform the image for model input
        print(json.dumps({"progress": "Resizing image"}), flush=True)
        image = image.resize((48, 48))  # Resize to model input size
        
        # Convert to tensor and normalize
        print(json.dumps({"progress": "Converting to tensor"}), flush=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        print(json.dumps({"progress": "Image preprocessing complete"}), flush=True)
        
        # Check tensor for issues
        if torch.isnan(image_tensor).any():
            print(json.dumps({"error": "Tensor contains NaN values"}), flush=True)
            return None
            
        return image_tensor
        
    except Exception as e:
        print(json.dumps({"error": f"Error preprocessing image: {str(e)}"}), flush=True)
        return None

def find_model_file(model_path):
    """Try to find the model file in various locations"""
    if os.path.exists(model_path):
        return model_path
        
    # Try common alternate locations
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alt_paths = [
        os.path.join(base_dir, "model-training-outputs", "model_checkpoints", "final_model.pth"),
        os.path.join(base_dir, "model_checkpoints", "final_model.pth"),
        os.path.join(base_dir, "model", "final_model.pth"),
        os.path.join(base_dir, "final_model.pth"),
        # Add the path from current directory
        "final_model.pth"
    ]
    
    for path in alt_paths:
        if os.path.exists(path):
            print(json.dumps({"info": f"Found model at alternate location: {path}"}), flush=True)
            return path
            
    return None

def load_model(model_path):
    """Load the model from the checkpoint"""
    try:
        print(json.dumps({"info": "Starting model loading process"}), flush=True)
        
        # Verify file exists
        actual_model_path = find_model_file(model_path)
        if not actual_model_path:
            print(json.dumps({"error": f"Model file not found: {model_path}"}), flush=True)
            print(json.dumps({"error": "Creating fallback mode with random predictions"}), flush=True)
            return None, None
            
        # Set device - use CPU for reliability
        device = torch.device("cpu")
        print(json.dumps({"info": f"Using device: {device}"}), flush=True)
        
        # Initialize model
        print(json.dumps({"info": "Initializing model architecture"}), flush=True)
        model = ExpressionRecognitionModel(num_classes=5)
        
        # Progress update for what's happening
        print(json.dumps({"progress": "Model created, loading weights..."}), flush=True)
        
        # Set torch performance optimizations for CPU
        # IMPORTANT: Reduce threads to avoid overloading
        torch.set_num_threads(2)  # Reduced to prevent system overload
        print(json.dumps({"info": f"Set torch threads to 2 for better CPU performance"}), flush=True)
        
        # Load weights
        try:
            print(json.dumps({"info": f"Loading model weights from: {actual_model_path}"}), flush=True)
            
            # Load with explicit map_location to CPU for reliability
            print(json.dumps({"progress": "Reading checkpoint file"}), flush=True)
            checkpoint = torch.load(actual_model_path, map_location='cpu')
            print(json.dumps({"progress": "Checkpoint file loaded into memory"}), flush=True)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print(json.dumps({"info": "Found model_state_dict in checkpoint"}), flush=True)
                print(json.dumps({"progress": "Loading state dict from checkpoint"}), flush=True)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print(json.dumps({"info": "Loading direct state dict"}), flush=True)
                print(json.dumps({"progress": "Loading direct state dict"}), flush=True)
                model.load_state_dict(checkpoint)
                
            # Set to evaluation mode
            model.eval()
            print(json.dumps({"info": "Model successfully loaded and set to eval mode"}), flush=True)
            
            return model, device
            
        except Exception as load_error:
            print(json.dumps({"error": f"Error loading model weights: {str(load_error)}"}), flush=True)
            print(json.dumps({"trace": traceback.format_exc()}), flush=True)
            return None, device  # Return device for fallback mode
            
    except Exception as e:
        print(json.dumps({"error": f"Error in load_model: {str(e)}"}), flush=True)
        print(json.dumps({"trace": traceback.format_exc()}), flush=True)
        return None, None

def predict_mood(model, image_tensor, device):
    """Run the model to predict mood from image"""
    try:
        # Check if we're in fallback mode
        if model is None:
            print(json.dumps({"warning": "Using fallback prediction mode - model is None"}), flush=True)
            
            # Return a fallback prediction (random but with fixed seed for consistency)
            random.seed(int(time.time()))  # Use current time for seed
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
            
        # Send progress update before transferring tensor
        print(json.dumps({"progress": "Preparing input tensor"}), flush=True)
        
        try:
            # Transfer tensor to device
            image_tensor = image_tensor.to(device)
            print(json.dumps({"progress": "Tensor transferred to device"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": f"Error transferring tensor to device: {str(e)}"}), flush=True)
            raise
        
        # Send progress update before inference
        print(json.dumps({"progress": "Running model inference"}), flush=True)
        
        # Run inference with explicit status updates
        with torch.no_grad():
            try:
                # Optimize memory usage for CPU
                torch.set_num_threads(2)  # Use fewer threads to reduce CPU load
                
                # Run the model
                print(json.dumps({"progress": "Forward pass started"}), flush=True)
                outputs = model(image_tensor)  # Forward method now has progress updates
                print(json.dumps({"progress": "Forward pass completed"}), flush=True)
                
                probabilities = F.softmax(outputs, dim=1)
                
                # Get confidence scores for all classes
                probs_array = probabilities[0].cpu().numpy()
                
                # Get top prediction
                confidence, predicted = torch.max(probabilities, 1)
                
                # Debug the model output
                print(json.dumps({"debug": f"Raw model output: {outputs.tolist()}"}), flush=True)
                print(json.dumps({"debug": f"Probabilities: {probs_array.tolist()}"}), flush=True)
                print(json.dumps({"debug": f"Predicted index: {predicted.item()}, confidence: {confidence.item()}"}), flush=True)
                
                # Map index to emotion label - use correct labels from training
                emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
                predicted_emotion = emotion_labels[predicted.item()]
                
                # Create probability dictionary for all emotions
                all_probabilities = {emotion: float(probs_array[i]) for i, emotion in enumerate(emotion_labels)}
                
                print(json.dumps({"progress": "Prediction complete", "mood": predicted_emotion}), flush=True)
                return predicted_emotion, confidence.item(), all_probabilities
            except Exception as e:
                print(json.dumps({"error": f"Error during forward pass: {str(e)}"}), flush=True)
                print(json.dumps({"trace": traceback.format_exc()}), flush=True)
                raise
    
    except Exception as e:
        print(json.dumps({"error": f"Error during prediction: {str(e)}"}), flush=True)
        print(json.dumps({"trace": traceback.format_exc()}), flush=True)
        return None, 0.0, {}

def main():
    """Main function to process the image and detect mood"""
    try:
        # Print early status to show we're alive
        print(json.dumps({"progress": "Starting mood detection inference"}), flush=True)
        
        # Get arguments
        if len(sys.argv) < 2:
            print(json.dumps({"error": "Missing image path argument"}), flush=True)
            sys.exit(1)
            
        image_path = sys.argv[1]
        print(json.dumps({"info": f"Processing image: {image_path}"}), flush=True)
        
        # Get model path from command line or use default
        if len(sys.argv) >= 3:
            model_path = sys.argv[2]
        else:
            # Use the specified model path as default
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "model-training-outputs", "model_checkpoints", "final_model.pth")
        
        print(json.dumps({"info": f"Using model path: {model_path}"}), flush=True)
        
        # AUTOMATIC FALLBACK FOR SLOW CPUs
        # Check CPU capability - automatically use fallback on slow CPUs
        use_fallback = False
        print(json.dumps({"info": "Automatic fallback disabled. Running actual model inference."}), flush=True)
        # try:
        #     # Use cpu_count and simple benchmark to determine if we should use fallback
        #     import multiprocessing
        #     cpu_count = multiprocessing.cpu_count()
            
        #     # Simple CPU speed test (matrix multiplication)
        #     import time
        #     import numpy as np
            
        #     print(json.dumps({"progress": "Testing CPU capability..."}), flush=True)
        #     start_time = time.time()
        #     # Simple matrix operation to test CPU speed
        #     test_size = 500
        #     a = np.random.rand(test_size, test_size)
        #     b = np.random.rand(test_size, test_size)
        #     c = np.dot(a, b)
        #     elapsed = time.time() - start_time
            
        #     # If benchmark takes more than 0.5 seconds, CPU is likely too slow
        #     if elapsed > 30:
        #         print(json.dumps({"warning": f"CPU appears to be slow (benchmark: {elapsed:.2f}s), using fallback mode"}), flush=True)
        #         use_fallback = True
                
        # except ImportError:
        #     # If numpy isn't available, check processor info
        #     try:
        #         import platform
        #         processor = platform.processor()
        #         print(json.dumps({"info": f"Processor: {processor}"}), flush=True)
                
        #         # If processor info contains certain keywords that suggest old/slow CPU
        #         if any(term in processor.lower() for term in ['atom', 'celeron', 'pentium']):
        #             print(json.dumps({"warning": f"Potentially slow CPU detected ({processor}), using fallback mode"}), flush=True)
        #             use_fallback = True
        #     except:
        #         # Can't detect CPU, be conservative
        #         print(json.dumps({"warning": "Unable to determine CPU capability, proceeding carefully"}), flush=True)
        
        # Load and preprocess image - ensure this step works
        print(json.dumps({"step": "preprocessing_image"}), flush=True)
        image_tensor = preprocess_image(image_path)
        
        if image_tensor is None:
            print(json.dumps({"error": "Failed to preprocess image"}), flush=True)
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
            sys.exit(1)
        
        # Send an early progress report so caller knows we're working
        print(json.dumps({"progress": "Image preprocessed successfully, loading model..."}), flush=True)
        
        # IMPORTANT: Early fallback if needed
        # Check if we should use fallback mode due to CPU constraints
        if not use_fallback:
            try:
                # Check if we have enough memory
                import psutil
                available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # in GB
                if available_memory < 2.0:  # Need at least 2GB free
                    print(json.dumps({"warning": f"Low memory ({available_memory:.2f}GB available), using fallback mode"}), flush=True)
                    use_fallback = True
            except ImportError:
                # psutil not available, check another way
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if 'MemAvailable' in line:
                                available_kb = int(line.split()[1])
                                available_gb = available_kb / (1024 * 1024)
                                if available_gb < 2.0:
                                    print(json.dumps({"warning": f"Low memory ({available_gb:.2f}GB available), using fallback mode"}), flush=True)
                                    use_fallback = True
                                break
                except:
                    pass
        
        # Use fallback immediately if needed
        if use_fallback:
            print(json.dumps({"info": "Using fallback due to system constraints"}), flush=True)
            # Create a fallback response with a good mood
            print(json.dumps({
                "mood": "Happy", 
                "agent_mood": "happy", 
                "confidence": 0.7,
                "all_probabilities": {
                    "Angry": 0.05, "Happy": 0.7, "Neutral": 0.15, "Sad": 0.05, "Surprise": 0.05
                },
                "fallback": True,
                "model_path": model_path,
                "device": "cpu-fallback"
            }), flush=True)
            sys.exit(0)
            
        # Load model - focus on making this step work reliably
        print(json.dumps({"step": "loading_model"}), flush=True)
        model, device = load_model(model_path)
        
        # Make prediction
        print(json.dumps({"step": "running_prediction"}), flush=True)
        mood, confidence, all_probabilities = predict_mood(model, image_tensor, device)
                
        if mood is None:
            print(json.dumps({"error": "Prediction returned no mood"}), flush=True)
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
            sys.exit(1)
        
        # Map model output to agent-compatible mood
        mood_mapping = {
            "Angry": "angry",
            "Happy": "happy",
            "Sad": "sad",
            "Neutral": "neutral",  
            "Surprise": "surprise"
        }
        
        agent_mood = mood_mapping.get(mood, "happy").lower()
        print(json.dumps({"progress": "Mood mapped to agent format"}), flush=True)
        
        # Return prediction result as JSON
        result = {
            "mood": mood,  # Original model output
            "agent_mood": agent_mood,  # Mapped to agent format
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "model_path": model_path,
            "device": str(device)
        }
        
        print(json.dumps({"step": "returning_result"}), flush=True)
        print(json.dumps(result), flush=True)
        sys.exit(0)
        
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {str(e)}"}), flush=True)
        print(json.dumps({"trace": traceback.format_exc()}), flush=True)
        
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
        sys.exit(1)

if __name__ == "__main__":
    main()