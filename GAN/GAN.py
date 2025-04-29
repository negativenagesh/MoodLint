# MoodLint: Advanced Facial Expression Transformation
# For Kaggle environment using GPU

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab import files
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths for Kaggle environment
expression_dir = "/kaggle/input/expressionsf/Expressions"
models_dir = "/kaggle/working/models"
os.makedirs(models_dir, exist_ok=True)

# Custom Dataset Class
class ExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.images = []
        self.labels = []
        
        for idx, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(img_path)
                        self.labels.append(idx)
            else:
                print(f"Warning: Class directory {class_dir} not found")
        
        print(f"Loaded {len(self.images)} images across {len(self.classes)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, label_tensor

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Dataset
dataset = ExpressionDataset(root_dir=expression_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Face detection and alignment class
class FaceProcessor:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_and_align_face(self, image):
        """Detect face, align and crop to a standard size"""
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
            
        # Convert RGB to BGR (for OpenCV)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            img_gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("No face detected, using the entire image")
            # If no face, just resize the original image
            aligned_face = cv2.resize(img_np, (256, 256))
            if len(aligned_face.shape) == 2:  # If grayscale
                aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(aligned_face)
        
        # Use the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Add margin
        height, width = img_np.shape[:2]
        margin_ratio = 0.3
        x1 = max(0, x - int(w * margin_ratio))
        y1 = max(0, y - int(h * margin_ratio))
        x2 = min(width, x + w + int(w * margin_ratio))
        y2 = min(height, y + h + int(h * margin_ratio))
        
        # Crop face region
        face_region = img_np[y1:y2, x1:x2]
        
        # Resize to standard size
        aligned_face = cv2.resize(face_region, (256, 256))
        
        # Ensure RGB format
        if len(aligned_face.shape) == 2:  # If grayscale
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(aligned_face)

# Attention Block for Generator
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        proj_query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H*W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        out = self.gamma * out + x
        return out

# Enhanced Generator with Attention
class EnhancedGenerator(nn.Module):
    def __init__(self, num_classes=5):
        super(EnhancedGenerator, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        # Expression condition processing
        self.condition_mlp = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512)
        )
        
        # Attention block
        self.attention = AttentionBlock(512)
        
        # Decoder with skip connections
        self.dec1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.dec5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
        # Normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.bn_dec1 = nn.BatchNorm2d(256)
        self.bn_dec2 = nn.BatchNorm2d(128)
        self.bn_dec3 = nn.BatchNorm2d(64)
        self.bn_dec4 = nn.BatchNorm2d(32)

    def forward(self, x, labels):
        # Encoder path
        e1 = F.leaky_relu(self.bn1(self.enc1(x)), 0.2)  # 32 x 128 x 128
        e2 = F.leaky_relu(self.bn2(self.enc2(e1)), 0.2) # 64 x 64 x 64
        e3 = F.leaky_relu(self.bn3(self.enc3(e2)), 0.2) # 128 x 32 x 32
        e4 = F.leaky_relu(self.bn4(self.enc4(e3)), 0.2) # 256 x 16 x 16
        e5 = F.leaky_relu(self.bn5(self.enc5(e4)), 0.2) # 512 x 8 x 8
        
        # Process expression condition
        labels_onehot = F.one_hot(labels, num_classes=5).float()
        cond_features = self.condition_mlp(labels_onehot)
        cond_features = cond_features.view(labels.size(0), 512, 1, 1).expand(-1, -1, 8, 8)
        
        # Apply attention with conditional features
        combined = e5 + cond_features
        attended = self.attention(combined)
        
        # Decoder path with skip connections
        d1 = F.relu(self.bn_dec1(self.dec1(attended)))  # 256 x 16 x 16
        d1 = torch.cat([d1, e4], dim=1)  # 512 x 16 x 16
        
        d2 = F.relu(self.bn_dec2(self.dec2(d1)))  # 128 x 32 x 32
        d2 = torch.cat([d2, e3], dim=1)  # 256 x 32 x 32
        
        d3 = F.relu(self.bn_dec3(self.dec3(d2)))  # 64 x 64 x 64
        d3 = torch.cat([d3, e2], dim=1)  # 128 x 64 x 64
        
        d4 = F.relu(self.bn_dec4(self.dec4(d3)))  # 32 x 128 x 128
        d4 = torch.cat([d4, e1], dim=1)  # 64 x 128 x 128
        
        d5 = torch.tanh(self.dec5(d4))  # 3 x 256 x 256
        
        return d5

# Enhanced Discriminator
class EnhancedDiscriminator(nn.Module):
    def __init__(self, num_classes=5):
        super(EnhancedDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        # For expression classification
        self.expr_conv = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.expr_fc = nn.Linear(512 * 8 * 8, num_classes)
        
        # For real/fake classification
        self.realfake_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)

    def forward(self, x, return_features=False):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        features = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        # Real/fake output
        realfake_out = self.realfake_conv(features)
        
        # Expression classification
        expr_features = F.leaky_relu(self.bn5(self.expr_conv(features)), 0.2)
        expr_features = expr_features.view(expr_features.size(0), -1)
        expr_out = self.expr_fc(expr_features)
        
        if return_features:
            return realfake_out, expr_out, features
        return realfake_out, expr_out

# Perceptual Loss using VGG features
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        self.model = nn.Sequential()
        for i in range(29):  # Use up to relu4_3 layer
            self.model.add_module(str(i), vgg[i])
            
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        # Normalize inputs
        x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        y = (y + 1) / 2
        
        # Extract features
        x_features = self.model(x)
        y_features = self.model(y)
        
        # Calculate L1 loss on features
        loss = F.l1_loss(x_features, y_features)
        return loss

# Training Function
def train_model(num_epochs=20):
    # Initialize Models
    generator = EnhancedGenerator(num_classes=5).to(device)
    discriminator = EnhancedDiscriminator(num_classes=5).to(device)
    
    # Initialize face processor
    face_processor = FaceProcessor()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min', factor=0.5, patience=5)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min', factor=0.5, patience=5)

    # Loss Functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    criterion_CE = nn.CrossEntropyLoss()
    perceptual_loss = PerceptualLoss()

    print(f"Starting training for {num_epochs} epochs...")
    
    # Training Loop
    for epoch in range(num_epochs):
        total_d_loss = 0
        total_g_loss = 0
        batch_count = 0
        
        for real_images, real_labels in train_loader:
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_images.size(0)
            batch_count += 1
            
            # Generate random target expressions
            target_labels = torch.randint(0, 5, (batch_size,), device=device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Generate fake images
            fake_images = generator(real_images, target_labels)
            
            # Get discriminator outputs
            real_realfake, real_expr = discriminator(real_images)
            fake_realfake, fake_expr = discriminator(fake_images.detach())
            
            # Calculate losses
            d_loss_real = criterion_GAN(real_realfake, torch.ones_like(real_realfake) * 0.9)  # Label smoothing
            d_loss_fake = criterion_GAN(fake_realfake, torch.zeros_like(fake_realfake))
            d_loss_expr = criterion_CE(real_expr, real_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake + d_loss_expr
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            # Get updated discriminator outputs for fake images
            fake_realfake, fake_expr = discriminator(fake_images)
            
            # Calculate losses
            g_loss_GAN = criterion_GAN(fake_realfake, torch.ones_like(fake_realfake))
            g_loss_expr = criterion_CE(fake_expr, target_labels)
            
            # Identity preservation and perceptual losses
            g_loss_L1 = criterion_L1(fake_images, real_images) * 10
            g_loss_perceptual = perceptual_loss(fake_images, real_images) * 2
            
            # Total generator loss
            g_loss = g_loss_GAN + g_loss_expr + g_loss_L1 + g_loss_perceptual
            g_loss.backward()
            optimizer_G.step()
            
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            
            if batch_count % 5 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
                print(f'  -> G_GAN: {g_loss_GAN.item():.4f}, G_expr: {g_loss_expr.item():.4f}, G_L1: {g_loss_L1.item():.4f}, G_perceptual: {g_loss_perceptual.item():.4f}')
        
        avg_d_loss = total_d_loss / batch_count
        avg_g_loss = total_g_loss / batch_count
        print(f'Epoch {epoch+1}/{num_epochs} completed, Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}')
        
        # Update learning rates
        scheduler_G.step(avg_g_loss)
        scheduler_D.step(avg_d_loss)
        
        # Save checkpoint after each epoch
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save(generator.state_dict(), os.path.join(models_dir, f'generator_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(models_dir, f'discriminator_epoch_{epoch+1}.pth'))
    
    # Save final models
    torch.save(generator.state_dict(), os.path.join(models_dir, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(models_dir, 'discriminator.pth'))
    print("Training completed! Models saved.")
    return generator, discriminator

# Function to load a trained model
def load_model():
    generator = EnhancedGenerator(num_classes=5).to(device)
    model_path = os.path.join(models_dir, 'generator.pth')
    
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print("No pre-trained model found. Training a new model...")
        generator, _ = train_model()
    
    return generator

# Enhanced Inference Function
def generate_expression(input_image_path, target_expression, generator, transform, face_processor=None, intensity=1.0):
    generator.eval()
    
    # Load and preprocess the input image
    input_image = Image.open(input_image_path).convert('RGB')
    
    # Apply face detection and alignment if processor is available
    if face_processor:
        aligned_image = face_processor.detect_and_align_face(input_image)
    else:
        aligned_image = input_image.resize((256, 256), Image.LANCZOS)
    
    # Transform for model input
    input_tensor = transform(aligned_image).unsqueeze(0).to(device)
    
    expression_map = {'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3, 'Surprise': 4}
    target_label = torch.tensor([expression_map[target_expression]], device=device)
    
    with torch.no_grad():
        generated_image = generator(input_tensor, target_label)
    
    # Apply intensity control (for more or less pronounced expressions)
    if intensity != 1.0:
        # Interpolate between original and generated image based on intensity
        generated_image = input_tensor * (1 - intensity) + generated_image * intensity
    
    # Convert back to image
    generated_image = (generated_image.squeeze(0).cpu() * 0.5 + 0.5) * 255
    generated_image = generated_image.permute(1, 2, 0).numpy().astype(np.uint8)
    output_image = Image.fromarray(generated_image)
    
    return output_image, aligned_image

# Batch processing function
def process_batch(input_dir, output_dir, generator, transform, target_expression="Happy", use_face_detection=True):
    """Process all images in a directory and save the results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    face_processor = FaceProcessor() if use_face_detection else None
    
    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, img_name)
            output_path = os.path.join(output_dir, f"{target_expression}_{img_name}")
            
            try:
                output_img, _ = generate_expression(
                    input_path, 
                    target_expression,
                    generator,
                    transform,
                    face_processor
                )
                output_img.save(output_path)
                print(f"Processed {img_name} -> {output_path}")
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")

# Function to display results in a notebook
def display_result(input_image_path, target_expression, generator, transform, use_face_detection=True):
    face_processor = FaceProcessor() if use_face_detection else None
    
    output_img, processed_input = generate_expression(
        input_image_path,
        target_expression,
        generator,
        transform,
        face_processor
    )
    
    # Display side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(processed_input)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(output_img)
    plt.title(f"Generated {target_expression} Expression")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return output_img

# Main Execution Block
if __name__ == "__main__":
    # Check if model already exists
    model_path = os.path.join(models_dir, 'generator.pth')
    if not os.path.exists(model_path):
        print("No pre-trained model found. Starting training...")
        train_model(num_epochs=20)
    
    # Load the model
    generator = load_model()
    
    # Example usage
    print("\nModel loaded and ready for inference!")
    print("You can use the following functions:")
    print("1. display_result(input_image_path, target_expression, generator, transform)")
    print("2. process_batch(input_dir, output_dir, generator, transform, target_expression)")
    
    # Example:
    # display_result('/kaggle/input/my-image.jpg', 'Happy', generator, transform)