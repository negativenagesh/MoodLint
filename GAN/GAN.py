import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk
import cv2
import logging
import time

# Set up logging for VSCode
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current file for relative paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define paths - using relative paths for better VSCode portability
expression_dir = os.path.join(parent_dir, "Expressions")
models_dir = os.path.join(current_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# Custom Dataset Class
class ExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.images = []
        self.labels = []
        
        logger.info(f"Loading dataset from {root_dir}")
        for idx, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            if os.path.exists(class_dir):
                class_images = 0
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(img_path)
                        self.labels.append(idx)
                        class_images += 1
                logger.info(f"Loaded {class_images} images from class '{cls}'")
            else:
                logger.warning(f"Class directory {class_dir} not found")
        
        logger.info(f"Total: Loaded {len(self.images)} images across {len(self.classes)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label_tensor = torch.tensor(label, dtype=torch.long)
            return image, label_tensor
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Return a blank image instead of crashing
            blank_image = torch.zeros((3, 256, 256))
            return blank_image, torch.tensor(label, dtype=torch.long)

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Face detection and alignment class
class FaceProcessor:
    def __init__(self):
        try:
            # Initialize face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            logger.error(f"Error initializing face detector: {str(e)}")
            self.face_cascade = None
        
    def detect_and_align_face(self, image):
        """Detect face, align and crop to a standard size"""
        if self.face_cascade is None:
            logger.warning("Face detector not available, resizing the entire image")
            return image.resize((256, 256), Image.LANCZOS)
            
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
            logger.info("No face detected, using the entire image")
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
# Perceptual Loss using VGG features
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        try:
            # Load pretrained VGG16
            vgg = models.vgg16(pretrained=True)
            
            # Move the model to the same device before extracting features
            vgg = vgg.to(device)
            
            # Extract features
            self.model = nn.Sequential()
            for i in range(29):  # Use up to relu4_3 layer
                self.model.add_module(str(i), vgg.features[i])
                
            # Ensure model is on the correct device
            self.model = self.model.to(device)
            self.model.eval()
                
            # Freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.error(f"Error loading VGG model: {str(e)}")
            # Create a dummy model if VGG fails to load
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ).to(device)
            
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

# Load dataset function - to be called when needed
def load_dataset():
    try:
        dataset = ExpressionDataset(root_dir=expression_dir, transform=transform)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        return dataset, train_loader
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

# Function to train model with progress visualization for VSCode
def train_model(num_epochs=40):
    try:
        # Load dataset
        dataset, train_loader = load_dataset()
        
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

        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # For plotting progress
        d_losses = []
        g_losses = []
        
        # Training Loop
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_start = time.time()
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
                    logger.info(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_count}/{len(train_loader)}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
            
            avg_d_loss = total_d_loss / batch_count
            avg_g_loss = total_g_loss / batch_count
            d_losses.append(avg_d_loss)
            g_losses.append(avg_g_loss)
            
            epoch_time = time.time() - epoch_start
            logger.info(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}')
            
            # Update learning rates
            scheduler_G.step(avg_g_loss)
            scheduler_D.step(avg_d_loss)
            
            # Save checkpoint and sample images
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join(models_dir, f'generator_epoch_{epoch+1}.pth')
                torch.save(generator.state_dict(), checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
                
                # Generate and save sample images
                if len(dataset) > 0:
                    try:
                        sample_img, sample_label = dataset[0]
                        sample_img = sample_img.unsqueeze(0).to(device)
                        
                        # Generate samples for each emotion
                        plt.figure(figsize=(15, 10))
                        plt.subplot(2, 3, 1)
                        plt.title("Original")
                        plt.axis('off')
                        img_np = sample_img.squeeze(0).cpu().numpy()
                        img_np = (img_np * 0.5 + 0.5) * 255  # Denormalize
                        img_np = np.transpose(img_np, (1, 2, 0)).astype(np.uint8)
                        plt.imshow(img_np)
                        
                        for idx, emotion in enumerate(['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']):
                            target_label = torch.tensor([idx], device=device)
                            with torch.no_grad():
                                gen_img = generator(sample_img, target_label)
                            
                            gen_np = gen_img.squeeze(0).cpu().numpy()
                            gen_np = (gen_np * 0.5 + 0.5) * 255  # Denormalize
                            gen_np = np.transpose(gen_np, (1, 2, 0)).astype(np.uint8)
                            
                            plt.subplot(2, 3, idx+2)
                            plt.title(emotion)
                            plt.axis('off')
                            plt.imshow(gen_np)
                        
                        sample_path = os.path.join(models_dir, f'samples_epoch_{epoch+1}.png')
                        plt.tight_layout()
                        plt.savefig(sample_path)
                        plt.close()
                        logger.info(f"Sample images saved to {sample_path}")
                    except Exception as e:
                        logger.error(f"Error generating samples: {str(e)}")
        
        # Plot and save training progress
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(d_losses, label='Discriminator Loss')
        plt.title('Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(g_losses, label='Generator Loss')
        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        progress_path = os.path.join(models_dir, 'training_progress.png')
        plt.savefig(progress_path)
        plt.close()
        logger.info(f"Training progress plot saved to {progress_path}")
        
        # Save final models
        torch.save(generator.state_dict(), os.path.join(models_dir, 'generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(models_dir, 'discriminator.pth'))
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s! Models saved.")
        return generator, discriminator
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

# Function to load a trained model
def load_model():
    try:
        generator = EnhancedGenerator(num_classes=5).to(device)
        model_path = os.path.join(models_dir, 'generator.pth')
        
        if os.path.exists(model_path):
            generator.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Model loaded successfully!")
        else:
            logger.warning("No pre-trained model found. Training a new model...")
            generator, _ = train_model()
        
        return generator
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Inference Function
def generate_expression(input_image_path, target_expression, generator, transform, face_processor=None, intensity=1.0):
    try:
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
    except Exception as e:
        logger.error(f"Error generating expression: {str(e)}")
        raise

# Batch processing function
def process_batch(input_dir, output_dir, generator, transform, target_expression="Happy", use_face_detection=True):
    try:
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
                    logger.info(f"Processed {img_name} -> {output_path}")
                except Exception as e:
                    logger.error(f"Error processing {img_name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise

# Simple GUI for the application - compatible with VSCode
def create_ui():
    try:
        # Load the model
        generator = load_model()
        face_processor = FaceProcessor()
        
        # Create main window
        root = tk.Tk()
        root.title("Face Expression GAN")
        root.geometry("800x600")
        
        # Variables
        input_image_path = tk.StringVar()
        selected_expression = tk.StringVar(value="Happy")
        intensity = tk.DoubleVar(value=1.0)
        
        # Create frames
        left_frame = ttk.Frame(root, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Input image display
        input_label = ttk.Label(left_frame, text="Input Image:")
        input_label.pack(pady=5)
        
        input_canvas = tk.Canvas(left_frame, width=256, height=256, bg="light gray")
        input_canvas.pack(pady=10)
        
        # Output image display
        output_label = ttk.Label(right_frame, text="Generated Image:")
        output_label.pack(pady=5)
        
        output_canvas = tk.Canvas(right_frame, width=256, height=256, bg="light gray")
        output_canvas.pack(pady=10)
        
        # Controls
        controls_frame = ttk.Frame(left_frame, padding=10)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Function to update canvases
        def update_input_canvas(img_path):
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            input_canvas.create_image(128, 128, image=photo)
            input_canvas.image = photo  # Keep a reference
        
        def update_output_canvas(img):
            img = img.resize((256, 256), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            output_canvas.create_image(128, 128, image=photo)
            output_canvas.image = photo  # Keep a reference
        
        # Browse button function
        def browse_image():
            filename = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            if filename:
                input_image_path.set(filename)
                update_input_canvas(filename)
                status_label.config(text="Image loaded. Ready to generate.")
        
        # Generate button function
        def generate():
            if not input_image_path.get():
                status_label.config(text="Please select an input image first.")
                return
            
            status_label.config(text="Generating expression... Please wait.")
            root.update()
            
            try:
                output_img, _ = generate_expression(
                    input_image_path.get(),
                    selected_expression.get(),
                    generator,
                    transform,
                    face_processor,
                    intensity.get()
                )
                update_output_canvas(output_img)
                
                # Save the generated image
                save_dir = os.path.join(os.path.dirname(input_image_path.get()), "generated")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"generated_{selected_expression.get().lower()}.png")
                output_img.save(save_path)
                
                status_label.config(text=f"Expression generated and saved to {save_path}")
            except Exception as e:
                status_label.config(text=f"Error: {str(e)}")
                logger.error(f"Generation error: {str(e)}")
        
        # Browse button
        browse_btn = ttk.Button(controls_frame, text="Browse Image", command=browse_image)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Expression dropdown
        expression_label = ttk.Label(controls_frame, text="Target Expression:")
        expression_label.pack(side=tk.LEFT, padx=5)
        
        expression_dropdown = ttk.Combobox(
            controls_frame,
            textvariable=selected_expression,
            values=["Angry", "Happy", "Neutral", "Sad", "Surprise"],
            state="readonly"
        )
        expression_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Intensity slider
        intensity_label = ttk.Label(controls_frame, text="Intensity:")
        intensity_label.pack(side=tk.LEFT, padx=5)
        
        intensity_slider = ttk.Scale(
            controls_frame,
            from_=0.1,
            to=1.0,
            variable=intensity,
            orient=tk.HORIZONTAL,
            length=100
        )
        intensity_slider.pack(side=tk.LEFT, padx=5)
        
        # Generate button
        generate_btn = ttk.Button(controls_frame, text="Generate", command=generate)
        generate_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        status_label = ttk.Label(left_frame, text="Ready. Please select an input image.")
        status_label.pack(pady=10)
        
        # Batch processing section
        batch_frame = ttk.LabelFrame(left_frame, text="Batch Processing", padding=10)
        batch_frame.pack(fill=tk.X, pady=10)
        
        input_dir_var = tk.StringVar()
        output_dir_var = tk.StringVar()
        batch_expression = tk.StringVar(value="Happy")
        
        # Input directory
        def browse_input_dir():
            dir_path = filedialog.askdirectory(title="Select Input Directory")
            if dir_path:
                input_dir_var.set(dir_path)
                
        input_dir_frame = ttk.Frame(batch_frame)
        input_dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_dir_frame, text="Input Dir:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(input_dir_frame, textvariable=input_dir_var, width=30).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(input_dir_frame, text="Browse", command=browse_input_dir).pack(side=tk.LEFT, padx=5)
        
        # Output directory
        def browse_output_dir():
            dir_path = filedialog.askdirectory(title="Select Output Directory")
            if dir_path:
                output_dir_var.set(dir_path)
                
        output_dir_frame = ttk.Frame(batch_frame)
        output_dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_dir_frame, text="Output Dir:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(output_dir_frame, textvariable=output_dir_var, width=30).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_dir_frame, text="Browse", command=browse_output_dir).pack(side=tk.LEFT, padx=5)
        
        # Batch options
        batch_options_frame = ttk.Frame(batch_frame)
        batch_options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(batch_options_frame, text="Expression:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(
            batch_options_frame,
            textvariable=batch_expression,
            values=["Angry", "Happy", "Neutral", "Sad", "Surprise"],
            state="readonly",
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Process batch button
        def run_batch_process():
            if not input_dir_var.get() or not output_dir_var.get():
                status_label.config(text="Please select both input and output directories.")
                return
                
            status_label.config(text="Processing batch... This may take a while.")
            root.update()
            
            try:
                process_batch(
                    input_dir_var.get(),
                    output_dir_var.get(),
                    generator,
                    transform,
                    batch_expression.get(),
                    True  # Use face detection
                )
                status_label.config(text=f"Batch processing completed. Results saved to {output_dir_var.get()}")
            except Exception as e:
                status_label.config(text=f"Batch processing error: {str(e)}")
                logger.error(f"Batch processing error: {str(e)}")
        
        ttk.Button(batch_options_frame, text="Process Batch", command=run_batch_process).pack(side=tk.LEFT, padx=5)
        
        # Training section
        training_frame = ttk.LabelFrame(left_frame, text="Training", padding=10)
        training_frame.pack(fill=tk.X, pady=10)
        
        epochs_var = tk.IntVar(value=20)
        
        training_options_frame = ttk.Frame(training_frame)
        training_options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(training_options_frame, text="Epochs:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(training_options_frame, textvariable=epochs_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Train button
        def run_training():
            status_label.config(text=f"Training model for {epochs_var.get()} epochs. This will take a while...")
            root.update()
            
            try:
                nonlocal generator
                generator, _ = train_model(num_epochs=epochs_var.get())
                status_label.config(text="Training completed successfully!")
            except Exception as e:
                status_label.config(text=f"Training error: {str(e)}")
                logger.error(f"Training error: {str(e)}")
        
        ttk.Button(training_options_frame, text="Train Model", command=run_training).pack(side=tk.LEFT, padx=5)
        
        # Run the UI
        root.mainloop()
    except Exception as e:
        logger.error(f"Error in UI: {str(e)}")
        raise

# Headless mode for VSCode tasks
def run_headless(mode="train", input_path=None, output_path=None, expression=None):
    if mode == "train":
        logger.info("Starting training in headless mode")
        train_model()
    elif mode == "generate" and input_path and output_path and expression:
        logger.info(f"Generating {expression} expression for {input_path}")
        generator = load_model()
        face_processor = FaceProcessor()
        output_img, _ = generate_expression(input_path, expression, generator, transform, face_processor)
        output_img.save(output_path)
        logger.info(f"Generated image saved to {output_path}")
    else:
        logger.error("Invalid headless mode parameters")

# Main Execution
if __name__ == "__main__":
    try:
        # Check if model already exists
        model_path = os.path.join(models_dir, 'generator.pth')
        if not os.path.exists(model_path):
            logger.info("No pre-trained model found. Starting training...")
            train_model()
        
        # Check for command line arguments
        import sys
        if len(sys.argv) > 1:
            # Headless mode
            if sys.argv[1] == "train":
                run_headless(mode="train")
            elif sys.argv[1] == "generate" and len(sys.argv) == 5:
                run_headless(
                    mode="generate", 
                    input_path=sys.argv[2],
                    output_path=sys.argv[3],
                    expression=sys.argv[4]
                )
            else:
                logger.info("Usage: python GAN.py [train | generate input_path output_path expression]")
        else:
            # Launch the UI
            logger.info("Launching UI")
            create_ui()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")