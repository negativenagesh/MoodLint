#!/usr/bin/env python3
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import traceback
from torchvision import transforms

# Define the Generator class exactly as in GAN.py
class Generator(nn.Module):
    def __init__(self, img_channels=3, latent_dim=100, num_classes=5):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        # Initial processing of noise and class embedding
        self.init = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Embed labels
        c = self.label_emb(labels)
        
        # Concatenate noise and label embedding
        z = torch.cat([z, c], dim=1)
        
        # Initial processing
        x = self.init(z)
        
        # Reshape to conv dimension
        x = x.view(x.size(0), 128, 8, 8)
        
        # Apply conv blocks
        x = self.conv_blocks(x)
        
        return x

# Define the expression classes
EXPRESSION_CLASSES = {
    'angry': 0,
    'happy': 1,
    'sad': 2,
    'surprise': 3,
    'neutral': 4
}

def process_input_image(image_path, size=64):
    """Process the input image to extract features that influence the latent vector"""
    try:
        # Load and resize image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((size, size), Image.LANCZOS)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor, True
    except Exception as e:
        print(json.dumps({"error": f"Error processing input image: {str(e)}"}), flush=True)
        return None, False

def generate_mood_visualization(input_image_path, target_mood, output_path, generator_path):
    """Generate a mood visualization based on input image and target mood"""
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(json.dumps({"info": f"Using device: {device}"}), flush=True)
        
        # Process input image
        print(json.dumps({"progress": "Processing input image..."}), flush=True)
        image_tensor, success = process_input_image(input_image_path)
        if not success:
            return False
            
        # Load generator model
        print(json.dumps({"progress": "Loading generator model..."}), flush=True)
        generator = Generator().to(device)
        
        # Check if generator path exists
        if not os.path.exists(generator_path):
            print(json.dumps({"error": f"Generator model not found at {generator_path}"}), flush=True)
            return False
            
        generator.load_state_dict(torch.load(generator_path, map_location=device))
        generator.eval()
        
        # Get mood class index
        mood_idx = EXPRESSION_CLASSES.get(target_mood.lower(), 4)  # Default to neutral if mood not found
        print(json.dumps({"info": f"Using mood: {target_mood} (class: {mood_idx})"}), flush=True)
        
        # Generate mood-influenced noise
        # Extract some color statistics from the input image to influence the noise
        image_np = image_tensor[0].cpu().numpy()
        image_mean = np.mean(image_np, axis=(1, 2))
        image_std = np.std(image_np, axis=(1, 2))
        
        # Create base noise vector
        noise = torch.randn(1, 100, device=device)
        
        # Influence first 3 dimensions of noise with image color statistics
        # This creates a subtle connection between input image and output
        for i in range(3):
            noise[0, i] = torch.tensor(image_mean[i], device=device)
            noise[0, i+3] = torch.tensor(image_std[i], device=device)
        
        # Influence by mood (amplify certain dimensions based on mood)
        mood_factor = 1.2  # Default amplification
        if target_mood.lower() == 'angry':
            # Amplify "red" channel influence and high frequency components
            noise[0, 0] *= 1.5
            noise[0, 6:15] *= 1.3
            mood_factor = 1.3
        elif target_mood.lower() == 'happy':
            # Amplify "yellow" influence (red+green)
            noise[0, 0] *= 1.3
            noise[0, 1] *= 1.3
            noise[0, 20:30] *= 1.2
            mood_factor = 1.1
        elif target_mood.lower() == 'sad':
            # Amplify "blue" channel and low frequency components
            noise[0, 2] *= 1.4
            noise[0, 40:50] *= 1.2
            mood_factor = 0.9
        elif target_mood.lower() == 'surprise':
            # More random, high energy
            noise *= 1.3
            mood_factor = 1.4
            
        # Create label tensor
        labels = torch.tensor([mood_idx], dtype=torch.long, device=device)
        
        # Generate the image
        print(json.dumps({"progress": "Generating image..."}), flush=True)
        with torch.no_grad():
            generated = generator(noise, labels)
            
        # Convert tensor to image
        print(json.dumps({"progress": "Converting to image..."}), flush=True)
        img = generated[0].cpu().detach()
        img = img * 0.5 + 0.5  # Denormalize
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()  # CxHxW to HxWxC
        img = (img * 255).astype(np.uint8)
        
        # Save the image
        output_img = Image.fromarray(img)
        output_img.save(output_path)
        
        print(json.dumps({"progress": "Image saved to: " + output_path}), flush=True)
        return True
        
    except Exception as e:
        print(json.dumps({"error": f"Error in generation: {str(e)}"}), flush=True)
        print(json.dumps({"trace": traceback.format_exc()}), flush=True)
        return False

def main():
    # Check arguments
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: generate.py <mood> <output_path> [input_image_path]"}), flush=True)
        return 1
        
    # Get mood and output path
    target_mood = sys.argv[1]
    output_path = sys.argv[2]
    
    # Get input image path if provided, otherwise use a default
    input_image_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not input_image_path:
        print(json.dumps({"error": "No input image provided"}), flush=True)
        return 1
        
    # Path to generator model
    gan_dir = os.path.dirname(os.path.abspath(__file__))
    generator_path = os.path.join(gan_dir, "generator_epoch_180.pt")
    
    print(json.dumps({"info": f"Using generator model: {generator_path}"}), flush=True)
    print(json.dumps({"info": f"Target mood: {target_mood}"}), flush=True)
    print(json.dumps({"info": f"Input image: {input_image_path}"}), flush=True)
    print(json.dumps({"info": f"Output path: {output_path}"}), flush=True)
    
    # Generate the image
    success = generate_mood_visualization(input_image_path, target_mood, output_path, generator_path)
    
    if success:
        print(json.dumps({"status": "success", "output_path": output_path}), flush=True)
        return 0
    else:
        print(json.dumps({"status": "failure"}), flush=True)
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {str(e)}"}), flush=True)
        print(json.dumps({"trace": traceback.format_exc()}), flush=True)
        sys.exit(1)