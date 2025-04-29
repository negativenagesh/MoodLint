import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Define paths
expression_dir = "/home/vu-lab03-pc24/MoodLint/Expressions"
models_dir = "/home/vu-lab03-pc24/MoodLint/models"
os.makedirs(models_dir, exist_ok=True)

# Load Dataset
dataset = ExpressionDataset(root_dir=expression_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Simplified Pix2PixHD Generator
class Pix2PixHDGenerator(nn.Module):
    def __init__(self, num_classes=5):
        super(Pix2PixHDGenerator, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3 + num_classes, 32, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # Decoder with skip connections
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)  # 128 + 128 from e3
        self.dec3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)  # 64 + 64 from e2
        self.dec4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)   # 32 + 32 from e1
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

    def forward(self, x, labels):
        # Convert labels to one-hot
        labels_onehot = F.one_hot(labels, num_classes=5).float()
        # Reshape for concatenation
        labels_onehot = labels_onehot.view(labels.size(0), -1, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, labels_onehot], dim=1)
        
        # Encoder path
        e1 = F.leaky_relu(self.bn1(self.enc1(x)), 0.2)
        e2 = F.leaky_relu(self.bn2(self.enc2(e1)), 0.2)
        e3 = F.leaky_relu(self.bn3(self.enc3(e2)), 0.2)
        e4 = F.leaky_relu(self.bn4(self.enc4(e3)), 0.2)
        
        # Decoder path with skip connections
        d1 = F.relu(self.bn3(self.dec1(e4)))
        d1 = torch.cat([d1, e3], dim=1)
        d2 = F.relu(self.bn2(self.dec2(d1)))
        d2 = torch.cat([d2, e2], dim=1)
        d3 = F.relu(self.bn1(self.dec3(d2)))
        d3 = torch.cat([d3, e1], dim=1)
        d4 = torch.tanh(self.dec4(d3))
        return d4

# Simplified Pix2PixHD Discriminator
class Pix2PixHDDiscriminator(nn.Module):
    def __init__(self, num_classes=5):
        super(Pix2PixHDDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3 + num_classes, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x, labels):
        # Convert labels to one-hot
        labels_onehot = F.one_hot(labels, num_classes=5).float()
        # Reshape for concatenation
        labels_onehot = labels_onehot.view(labels.size(0), -1, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, labels_onehot], dim=1)
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = self.conv4(x)
        return x

# Function to train model
def train_model(num_epochs=20):
    # Initialize Models
    generator = Pix2PixHDGenerator(num_classes=5).to(device)
    discriminator = Pix2PixHDDiscriminator(num_classes=5).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss Functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

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
            fake_images = generator(real_images, target_labels)
            real_output = discriminator(real_images, real_labels)
            fake_output = discriminator(fake_images.detach(), target_labels)
            
            d_loss_real = criterion_GAN(real_output, torch.ones_like(real_output) * 0.9)  # Label smoothing
            d_loss_fake = criterion_GAN(fake_output, torch.zeros_like(fake_output))
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_images, target_labels)
            g_loss_GAN = criterion_GAN(fake_output, torch.ones_like(fake_output))
            g_loss_L1 = criterion_L1(fake_images, real_images) * 100  # Identity preservation weight
            g_loss = g_loss_GAN + g_loss_L1
            g_loss.backward()
            optimizer_G.step()
            
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            
            if batch_count % 5 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
        
        avg_d_loss = total_d_loss / batch_count
        avg_g_loss = total_g_loss / batch_count
        print(f'Epoch {epoch+1}/{num_epochs} completed, Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}')
        
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
    generator = Pix2PixHDGenerator(num_classes=5).to(device)
    model_path = os.path.join(models_dir, 'generator.pth')
    
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print("No pre-trained model found. Training a new model...")
        generator, _ = train_model()
    
    return generator

# Inference Function
def generate_expression(input_image_path, target_expression, generator, transform):
    generator.eval()
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    expression_map = {'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3, 'Surprise': 4}
    target_label = torch.tensor([expression_map[target_expression]], device=device)
    
    with torch.no_grad():
        generated_image = generator(input_tensor, target_label)
    
    # Convert back to image
    generated_image = (generated_image.squeeze(0).cpu() * 0.5 + 0.5) * 255
    generated_image = generated_image.permute(1, 2, 0).numpy().astype(np.uint8)
    output_image = Image.fromarray(generated_image)
    
    return output_image

# Simple GUI for the application
def create_ui():
    generator = load_model()
    
    # Create main window
    root = tk.Tk()
    root.title("MoodLint - Expression Generator")
    root.geometry("800x600")
    
    # Variables
    input_image_path = tk.StringVar()
    selected_expression = tk.StringVar(value="Happy")
    
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
            output_img = generate_expression(
                input_image_path.get(),
                selected_expression.get(),
                generator,
                transform
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
    
    # Generate button
    generate_btn = ttk.Button(controls_frame, text="Generate", command=generate)
    generate_btn.pack(side=tk.LEFT, padx=5)
    
    # Status label
    status_label = ttk.Label(left_frame, text="Ready. Please select an input image.")
    status_label.pack(pady=10)
    
    # Train button
    def start_training():
        status_label.config(text="Training started. This may take a while...")
        root.update()
        try:
            train_model()
            # Reload the model after training
            nonlocal generator
            generator = load_model()
            status_label.config(text="Training completed successfully!")
        except Exception as e:
            status_label.config(text=f"Training error: {str(e)}")
    
    train_btn = ttk.Button(left_frame, text="Train New Model", command=start_training)
    train_btn.pack(pady=10)
    
    root.mainloop()

# Main Execution
if __name__ == "__main__":
    # Check if model already exists
    model_path = os.path.join(models_dir, 'generator.pth')
    if not os.path.exists(model_path):
        print("No pre-trained model found. Starting training...")
        train_model()
    
    # Launch the UI
    create_ui()