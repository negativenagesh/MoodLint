import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
from PIL import Image

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directories
output_dir = '/kaggle/working/expression_gan_output'
models_dir = f'{output_dir}/models'
samples_dir = f'{output_dir}/samples'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

# Dataset path in Kaggle
dataset_path = '/kaggle/input/expressions'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Define the 5 expression classes
EXPRESSION_CLASSES = {
    'angry': 0,
    'happy': 1,
    'sad': 2,
    'surprise': 3,
    'neutral': 4
}
num_classes = 5

try:
    # Load the dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    # Override with our 5 expression classes
    print(f"Using predefined 5 expression classes: {EXPRESSION_CLASSES}")
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Dataset loaded with {len(dataset)} images, {num_classes} classes.")
    print(f"Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")
    
    # Use our expression classes for class mapping
    class_to_idx = EXPRESSION_CLASSES
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Create dummy data for testing if dataset loading fails
    print("Creating dummy dataset for testing...")
    
    class DummyDataset(Dataset):
        def __init__(self, size=100, transform=None):
            self.size = size
            self.transform = transform
            self.class_to_idx = EXPRESSION_CLASSES
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Generate random image and label
            image = torch.rand(3, 64, 64)
            label = torch.randint(0, 5, (1,)).item()
            return image, label
    
    dummy_dataset = DummyDataset(size=160, transform=transform)
    train_size = int(0.8 * len(dummy_dataset))
    val_size = len(dummy_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dummy_dataset, [train_size, val_size])
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    class_to_idx = EXPRESSION_CLASSES
    print(f"Dummy dataset created with {len(dummy_dataset)} images, {num_classes} classes.")

# Simple Generator Model with InstanceNorm instead of BatchNorm for better handling of small batches
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

# Simple Discriminator Model with InstanceNorm
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, num_classes=5):
        super(Discriminator, self).__init__()
        
        self.img_channels = img_channels
        self.num_classes = num_classes
        
        # Image processing
        self.conv_blocks = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(img_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        
        # Calculate number of neurons after convolutions
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid()
        )
        
        # For classification of expression
        self.aux_layer = nn.Sequential(
            nn.Linear(128 * 8 * 8, num_classes),
        )
        
    def forward(self, img):
        # Process image through conv blocks
        features = self.conv_blocks(img)
        features = features.view(features.size(0), -1)
        
        # Adversarial output (real/fake)
        validity = self.adv_layer(features)
        
        # Class prediction
        label = self.aux_layer(features)
        
        return validity, label

# Initialize models
generator = Generator(img_channels=3, latent_dim=100, num_classes=num_classes).to(device)
discriminator = Discriminator(img_channels=3, num_classes=num_classes).to(device)

# Loss functions
adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Function to generate sample images during training
def generate_sample_images(generator, epoch, batches_done, sample_path):
    """Generate and save sample images during training"""
    # Switch to eval mode for generation
    generator.eval()
    
    n_row = 5  # One for each class
    n_col = 5  # 5 examples per class
    
    # Create figure
    fig, axs = plt.subplots(n_row, n_col, figsize=(10, 10))
    
    # Map from class index to class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Generate image for each class
    for row in range(n_row):
        class_idx = row  # Class index
        
        # Generate multiple samples for this class at once
        z = torch.randn(n_col, 100, device=device)
        labels = torch.full((n_col,), class_idx, dtype=torch.long, device=device)
        
        # Generate images
        with torch.no_grad():
            gen_imgs = generator(z, labels)
        
        # Display each generated image
        for col in range(n_col):
            # Convert image to displayable format
            img = gen_imgs[col].detach().cpu()
            img = (img * 0.5 + 0.5)  # Denormalize
            img = img.permute(1, 2, 0)  # CxHxW to HxWxC
            
            # Display image
            axs[row, col].imshow(img)
            if col == 0:
                axs[row, col].set_ylabel(idx_to_class[class_idx])
            axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{sample_path}/samples_{batches_done}.png")
    plt.close()
    
    # Switch back to train mode
    generator.train()

# Training function
def train(num_epochs=200, sample_interval=500):
    start_time = time.time()
    
    # Tensor for ground truth adversarial labels
    valid = torch.ones(batch_size, 1, device=device)
    fake = torch.zeros(batch_size, 1, device=device)
    
    # Lists to store losses
    G_losses, D_losses = [], []
    
    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0
        
        for i, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            batch_size_actual = imgs.size(0)  # Get actual batch size
            
            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Make sure our ground truth labels match the actual batch size
            valid_batch = valid[:batch_size_actual]
            fake_batch = fake[:batch_size_actual]
            
            # -----------------
            #  Train Generator
            # -----------------
            
            optimizer_G.zero_grad()
            
            # Generate a batch of noise vectors
            z = torch.randn(batch_size_actual, 100, device=device)
            
            # Generate fake images
            gen_imgs = generator(z, labels)
            
            # Discriminator's prediction on generator images
            validity, pred_label = discriminator(gen_imgs)
            
            # Calculate generator loss
            g_loss = 0.5 * (adversarial_loss(validity, valid_batch) + auxiliary_loss(pred_label, labels))
            
            # Backpropagate and update generator
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            optimizer_D.zero_grad()
            
            # Discriminator on real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = 0.5 * (adversarial_loss(real_pred, valid_batch) + auxiliary_loss(real_aux, labels))
            
            # Discriminator on fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = 0.5 * (adversarial_loss(fake_pred, fake_batch) + auxiliary_loss(fake_aux, labels))
            
            # Total discriminator loss
            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            
            # Backpropagate and update discriminator
            d_loss.backward()
            optimizer_D.step()
            
            # Keep track of loss
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
            
            # Print progress
            if (i+1) % 20 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(train_loader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                    f"Time: {elapsed:.2f}s"
                )
            
            # Generate and save sample images periodically
            batches_done = epoch * len(train_loader) + i
            if batches_done % sample_interval == 0:
                generate_sample_images(generator, epoch, batches_done, samples_dir)
        
        # Calculate average losses for this epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)
        
        # Generate samples at the end of each epoch
        generate_sample_images(generator, epoch, epoch+1, samples_dir)
        
        # Save models periodically
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            torch.save(generator.state_dict(), f"{models_dir}/generator_epoch_{epoch+1}.pt")
            torch.save(discriminator.state_dict(), f"{models_dir}/discriminator_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save(generator.state_dict(), f"{models_dir}/generator_final.pt")
    torch.save(discriminator.state_dict(), f"{models_dir}/discriminator_final.pt")
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator')
    plt.plot(D_losses, label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/training_loss.png")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/3600:.2f} hours")
    
    return generator

# Function to generate facial expressions 
def generate_expression(target_expression, generator, num_samples=5):
    """Generate facial expressions of a specified type"""
    
    # Set to evaluation mode
    generator.eval()
    
    # Prepare target class
    if isinstance(target_expression, str):
        # Find the index of the target expression
        target_idx = None
        for cls_name, idx in class_to_idx.items():
            if cls_name.lower() == target_expression.lower():
                target_idx = idx
                break
        
        if target_idx is None:
            print(f"Unknown expression: {target_expression}. Using index 0.")
            target_idx = 0
    else:
        # Use provided index
        target_idx = target_expression
    
    # Generate images
    z = torch.randn(num_samples, 100, device=device)
    labels = torch.full((num_samples,), target_idx, dtype=torch.long, device=device)
    
    with torch.no_grad():
        fake_images = generator(z, labels)
    
    # Convert to numpy images
    images = []
    for i in range(num_samples):
        img = fake_images[i].cpu()
        img = (img * 0.5 + 0.5)  # Denormalize
        img = img.permute(1, 2, 0).numpy()  # CxHxW to HxWxC
        img = (img * 255).astype(np.uint8)
        images.append(Image.fromarray(img))
    
    return images

# Function to modify existing faces 
def modify_face(input_image, target_expression, generator):
    """Apply a target expression to an input image using the generator"""
    # Set to evaluation mode
    generator.eval()
    
    # Process input image
    if isinstance(input_image, str):
        # Load image from path
        image = Image.open(input_image).convert('RGB')
    else:
        # Use provided image
        image = input_image
    
    # Resize to model input size
    image = image.resize((64, 64), Image.LANCZOS)
    
    # Prepare target expression
    if isinstance(target_expression, str):
        target_idx = None
        for cls_name, idx in class_to_idx.items():
            if cls_name.lower() == target_expression.lower():
                target_idx = idx
                break
        
        if target_idx is None:
            print(f"Unknown expression: {target_expression}. Using index 0.")
            target_idx = 0
    else:
        target_idx = target_expression
    
    # Generate multiple variations
    num_variations = 5
    modified_images = []
    
    # Generate multiple variations with different noise inputs
    for _ in range(num_variations):
        # Random noise
        z = torch.randn(1, 100, device=device)
        
        # Target expression 
        label = torch.tensor([target_idx], device=device)
        
        # Generate modified image
        with torch.no_grad():
            modified = generator(z, label)
        
        # Convert to PIL image
        modified_img = modified[0].cpu()
        modified_img = (modified_img * 0.5 + 0.5)  # Denormalize
        modified_img = modified_img.permute(1, 2, 0).numpy()  # CxHxW to HxWxC
        modified_img = (modified_img * 255).astype(np.uint8)
        modified_images.append(Image.fromarray(modified_img))
    
    return modified_images

# Main execution
print("Starting training for 200 epochs...")
generator = train(num_epochs=200, sample_interval=200)

# Generate samples for each expression
print("\nGenerating sample images for each expression...")
for emotion in EXPRESSION_CLASSES.keys():
    output_dir_emotion = f"{samples_dir}/final_{emotion}"
    os.makedirs(output_dir_emotion, exist_ok=True)
    
    images = generate_expression(emotion, generator, num_samples=10)
    
    for i, img in enumerate(images):
        img.save(f"{output_dir_emotion}/{emotion}_{i+1}.png")
    
    print(f"Generated {len(images)} {emotion} expression images")

print("Done!")