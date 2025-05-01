#!/usr/bin/env python3
import cv2
import sys
import json
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import traceback
import subprocess
import torch
import numpy as np
import torch.nn as nn

try:
    from PIL import Image, ImageTk
    HAS_TKINTER = True
except ImportError:
    print(json.dumps({"error": "Tkinter modules not available. Please install python3-tk"}), flush=True)
    HAS_TKINTER = False

# Send startup message so extension knows we're running
print(json.dumps({"status": "starting", "gui": "initializing"}), flush=True)

# Define the Generator model based on the GAN.py architecture
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

class FutureMoodApp:
    def __init__(self, root, image_path, mood, model_dir=None, output_path=None):
        self.root = root
        self.root.title("MoodLint Future Mood Generator")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Make window appear on top initially
        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.attributes('-topmost', False)
        
        # Set window size and position
        self.root.geometry("900x600")
        self.root.resizable(False, False)
        
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 900) // 2
        y = (screen_height - 600) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # Store image path and mood
        self.image_path = image_path
        self.mood = mood
        self.model_dir = model_dir if model_dir else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "GAN"
        )
        self.output_path = output_path if output_path else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "output", 
            f"future_mood_{int(time.time())}.png"
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Images for display
        self.input_image = None
        self.generated_image = None
        self.input_photo = None
        self.generated_photo = None
        
        # Device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create UI elements
        self.setup_ui()
        
        # Tell extension we've initialized
        print(json.dumps({"status": "ready", "gui": "tkinter"}), flush=True)
        
        # Load the input image if it exists
        if os.path.exists(self.image_path):
            self.load_input_image()
            # Start generation after a short delay to allow UI to initialize
            self.root.after(500, self.generate_image)
        else:
            print(json.dumps({"error": f"Input image not found: {self.image_path}"}), flush=True)
            self.status_var.set(f"Error: Input image not found: {self.image_path}")
    
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text=f"Future Mood Visualization - {self.mood.capitalize()}", 
                  font=("Arial", 18, "bold")).pack()
        ttk.Label(header_frame, text="Using GAN to visualize your future emotional state").pack()
        
        # Create frame for images side by side
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Left side: original image
        left_frame = ttk.LabelFrame(images_frame, text="Your Current Image")
        left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        self.input_canvas = tk.Canvas(left_frame, width=400, height=400, bg="black")
        self.input_canvas.pack(pady=5)
        self.input_canvas.create_text(200, 200, text="Loading Input Image...", fill="white", font=("Arial", 14))
        
        # Right side: generated image
        right_frame = ttk.LabelFrame(images_frame, text="Future Mood Visualization")
        right_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)
        
        self.output_canvas = tk.Canvas(right_frame, width=400, height=400, bg="black")
        self.output_canvas.pack(pady=5)
        self.output_canvas.create_text(200, 200, text="Generating...", fill="white", font=("Arial", 14))
        
        # Status and progress
        self.status_var = tk.StringVar(value="Initializing...")
        self.status = ttk.Label(main_frame, textvariable=self.status_var)
        self.status.pack(pady=5)
        
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=880, mode="indeterminate")
        self.progress.pack(pady=5)
        self.progress.start()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.save_button = ttk.Button(button_frame, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Close", command=self.on_closing).pack(side=tk.LEFT, padx=5)
    
    def load_input_image(self):
        try:
            # Load and display the input image
            image = Image.open(self.image_path)
            image = image.resize((400, 400), Image.LANCZOS)
            self.input_image = image
            self.input_photo = ImageTk.PhotoImage(image)
            
            # Clear canvas and display image
            self.input_canvas.delete("all")
            self.input_canvas.create_image(200, 200, image=self.input_photo)
            
            # Update status
            self.status_var.set("Input image loaded. Generating future mood visualization...")
            
            print(json.dumps({"info": f"Input image loaded from {self.image_path}"}), flush=True)
        except Exception as e:
            error_msg = f"Error loading input image: {str(e)}"
            self.status_var.set(error_msg)
            print(json.dumps({"error": error_msg}), flush=True)
    
    def generate_image(self):
        try:
            self.status_var.set(f"Generating future mood visualization for '{self.mood}' mood...")
            
            # Prepare mood class index
            mood_to_class = {
                'angry': 0,
                'happy': 1,
                'sad': 2,
                'surprise': 3,
                'neutral': 4
            }
            
            # Default to neutral if mood not found
            mood_class = mood_to_class.get(self.mood.lower(), 4)
            
            # Load the generator model
            generator_path = os.path.join(self.model_dir, 'generator_epoch_180.pt')
            
            if not os.path.exists(generator_path):
                self.status_var.set(f"Error: Model not found at {generator_path}")
                print(json.dumps({"error": f"Model not found at {generator_path}"}), flush=True)
                return
            
            generator = Generator().to(self.device)
            generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            generator.eval()
            
            print(json.dumps({"progress": "Generating image..."}), flush=True)
            
            # Convert input image to tensor
            transform = self.prepare_input_image(self.input_image)
            
            # Generate noise based on input image
            z = torch.randn(1, 100, 1, 1, device=self.device)
            
            # Create labels tensor for the desired mood
            labels = torch.full((1,), mood_class, dtype=torch.long, device=self.device)
            
            # Generate the image
            with torch.no_grad():
                generated = generator(z, labels)
                
                # Convert tensor to image
                generated_img = transforms_to_pil(generated[0])
                
                # Resize to display size
                generated_img = generated_img.resize((400, 400), Image.LANCZOS)
                
                # Save the generated image
                generated_img.save(self.output_path)
                
                # Update UI with generated image
                self.generated_image = generated_img
                self.generated_photo = ImageTk.PhotoImage(generated_img)
                
                # Display the generated image
                self.output_canvas.delete("all")
                self.output_canvas.create_image(200, 200, image=self.generated_photo)
                
                # Update status
                self.status_var.set("Future mood visualization complete!")
                self.progress.stop()
                
                # Enable save button
                self.save_button.config(state=tk.NORMAL)
                
                # Notify extension that generation is complete
                print(json.dumps({
                    "status": "generation_complete",
                    "output_path": self.output_path
                }), flush=True)
                
        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            self.status_var.set(error_msg)
            print(json.dumps({"error": error_msg}), flush=True)
            print(json.dumps({"trace": traceback.format_exc()}), flush=True)
            self.progress.stop()
    
    def prepare_input_image(self, image):
        # Convert PIL image to tensor for model input
        transform = transforms_to_tensor(image)
        return transform
    
    def save_image(self):
        # Ask user where to save the file
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All Files", "*.*")],
            title="Save Future Mood Image"
        )
        
        if save_path:
            try:
                if self.generated_image:
                    self.generated_image.save(save_path)
                    self.status_var.set(f"Image saved to: {save_path}")
                    print(json.dumps({"info": f"Image saved to: {save_path}"}), flush=True)
                else:
                    self.status_var.set("Error: No generated image to save")
            except Exception as e:
                error_msg = f"Error saving image: {str(e)}"
                self.status_var.set(error_msg)
                print(json.dumps({"error": error_msg}), flush=True)
    
    def on_closing(self):
        print(json.dumps({"status": "closed"}), flush=True)
        self.root.destroy()
        sys.exit(0)

def transforms_to_tensor(image):
    # Convert PIL image to tensor and normalize
    image = image.resize((64, 64), Image.LANCZOS)
    image = image.convert("RGB")
    
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor format [C, H, W]
    img_tensor = np.transpose(img_array, (2, 0, 1))
    
    # Normalize to [-1, 1]
    img_tensor = (img_tensor - 0.5) / 0.5
    
    return torch.from_numpy(img_tensor).unsqueeze(0)

def transforms_to_pil(tensor):
    # Convert tensor to PIL image
    # Tensor is in format [C, H, W] with values in range [-1, 1]
    
    # Denormalize
    img = tensor.cpu().clone()
    img = img * 0.5 + 0.5
    
    # Convert to numpy and transpose to [H, W, C]
    img = img.numpy().transpose((1, 2, 0))
    
    # Clip values to [0, 1]
    img = np.clip(img, 0, 1)
    
    # Scale to [0, 255] and convert to uint8
    img = (img * 255).astype(np.uint8)
    
    # Create PIL image
    return Image.fromarray(img)

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Missing arguments. Usage: gan_popup.py <image_path> <mood> [output_path]"}), flush=True)
        sys.exit(1)
    
    image_path = sys.argv[1]
    mood = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        # Create the main window
        root = tk.Tk()
        
        # Set style
        style = ttk.Style()
        try:
            style.theme_use('clam')  # Use a modern theme
        except:
            pass  # Fall back to default theme if 'clam' is not available
        
        # Create application
        app = FutureMoodApp(root, image_path, mood, output_path=output_path)
        
        # Start the main loop
        root.mainloop()
    except Exception as e:
        print(json.dumps({"error": f"Critical error: {str(e)}"}), flush=True)
        traceback_str = traceback.format_exc()
        print(json.dumps({"trace": traceback_str}), flush=True)
        sys.exit(1)