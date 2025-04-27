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
from IPython.display import FileLink, display
import ipywidgets as widgets

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class
class ExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['angry', 'happy', 'sad']
        self.images = []
        self.labels = []
        for idx, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(idx)

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

# Load Dataset (Ensure this path matches your Kaggle dataset)
dataset = ExpressionDataset(root_dir="/kaggle/input/expression", transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Simplified Pix2PixHD Generator
class Pix2PixHDGenerator(nn.Module):
    def __init__(self, num_classes=3):
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
        labels_onehot = F.one_hot(labels, num_classes=3).float()
        labels_onehot = labels_onehot.view(labels.size(0), -1, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, labels_onehot], dim=1)
        
        e1 = F.leaky_relu(self.bn1(self.enc1(x)), 0.2)
        e2 = F.leaky_relu(self.bn2(self.enc2(e1)), 0.2)
        e3 = F.leaky_relu(self.bn3(self.enc3(e2)), 0.2)
        e4 = F.leaky_relu(self.bn4(self.enc4(e3)), 0.2)
        
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
    def __init__(self, num_classes=3):
        super(Pix2PixHDDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3 + num_classes, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x, labels):
        labels_onehot = F.one_hot(labels, num_classes=3).float()
        labels_onehot = labels_onehot.view(labels.size(0), -1, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, labels_onehot], dim=1)
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = self.conv4(x)
        return x

# Initialize Models
generator = Pix2PixHDGenerator(num_classes=3).to(device)
discriminator = Pix2PixHDDiscriminator(num_classes=3).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss Functions
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

# Training Loop
num_epochs = 10  # Increase for better results
for epoch in range(num_epochs):
    for real_images, real_labels in train_loader:
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        batch_size = real_images.size(0)
        target_labels = torch.randint(0, 3, (batch_size,), device=device)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        fake_images = generator(real_images, target_labels)
        real_output = discriminator(real_images, real_labels)
        fake_output = discriminator(fake_images.detach(), target_labels)
        
        d_loss_real = criterion_GAN(real_output, torch.ones_like(real_output) * 0.9)
        d_loss_fake = criterion_GAN(fake_output, torch.zeros_like(fake_output))
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        fake_output = discriminator(fake_images, target_labels)
        g_loss_GAN = criterion_GAN(fake_output, torch.ones_like(fake_output))
        g_loss_L1 = criterion_L1(fake_images, real_images) * 100
        g_loss = g_loss_GAN + g_loss_L1
        g_loss.backward()
        optimizer_G.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

# Save Models
torch.save(generator.state_dict(), '/kaggle/working/generator.pth')
torch.save(discriminator.state_dict(), '/kaggle/working/discriminator.pth')

# Inference Function
def generate_expression(input_image_path, target_expression, generator, transform):
    generator.eval()
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    expression_map = {'angry': 0, 'happy': 1, 'sad': 2}
    target_label = torch.tensor([expression_map[target_expression]], device=device)
    
    with torch.no_grad():
        generated_image = generator(input_tensor, target_label)
    
    generated_image = (generated_image.squeeze(0).cpu() * 0.5 + 0.5) * 255
    generated_image = generated_image.permute(1, 2, 0).numpy().astype(np.uint8)
    output_image = Image.fromarray(generated_image)
    output_image.save('/kaggle/working/generated_image.png')
    return output_image

# User Upload Interface
print("Training completed! Please upload an image to generate a synthetic facial expression.")
uploader = widgets.FileUpload(accept='.jpg,.png', multiple=False)
display(uploader)

def on_upload_change(change):
    if uploader.value:
        uploaded_file = list(uploader.value.values())[0]
        with open('/kaggle/working/uploaded_image.jpg', 'wb') as f:
            f.write(uploaded_file['content'])
        print("Image uploaded successfully!")
        
        target_expression = 'happy'  # Modify or add user input as needed
        generate_expression('/kaggle/working/uploaded_image.jpg', target_expression, generator, transform)
        print("Synthetic image generated!")
        display(FileLink('/kaggle/working/generated_image.png'))

uploader.observe(on_upload_change, names='_counter')