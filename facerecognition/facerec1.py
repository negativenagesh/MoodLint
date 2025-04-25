import cv2
import os
import argparse
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

def process_emotion_folder(input_folder, output_folder, crop_folder, resize_folder, 
                           detector, scale_w=1.5, scale_h=1.4, 
                           neck_reduction_factor=0.3, resize_width=600, resize_height=600):
    """
    Process images in a single emotion folder - detecting faces, cropping, and resizing.
    
    Args:
        input_folder: Path to input folder with images
        output_folder: Base output directory
        crop_folder: Subfolder name for cropped images
        resize_folder: Subfolder name for resized images
        detector: MTCNN detector instance
        scale_w: Width scaling factor for face bounding box
        scale_h: Height scaling factor for face bounding box
        neck_reduction_factor: Factor to reduce neck visibility
        resize_width: Width to resize final images
        resize_height: Height to resize final images
    """
    # Create output folders
    crop_output_path = os.path.join(output_folder, crop_folder)
    resize_output_path = os.path.join(output_folder, resize_folder)
    os.makedirs(crop_output_path, exist_ok=True)
    os.makedirs(resize_output_path, exist_ok=True)
    
    # Process each image in the folder
    for filename in tqdm(os.listdir(input_folder), desc=f"Processing {os.path.basename(input_folder)}"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            
            # Use CUDA-enabled OpenCV functions when possible
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error loading {filename}, skipping...")
                continue
                
            # Convert to RGB for MTCNN
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = detector.detect_faces(image_rgb)
            
            if faces:
                # Process the first detected face
                face = faces[0]
                x, y, w_bbox, h_bbox = face['box']
                
                # Expand width equally on both sides for ears
                new_w = int(w_bbox * scale_w)
                x = x - int((new_w - w_bbox) // 2)  # Shift left to balance expansion
                
                # Expand height but reduce lower part (neck)
                new_h = int(h_bbox * scale_h)
                y = y - int((new_h - h_bbox) * 0.2)  # Move up slightly for forehead/hair
                new_h = new_h - int(h_bbox * neck_reduction_factor)  # Reduce lower neck exposure
                
                # Ensure the box stays within the image boundaries
                x = max(0, x)
                y = max(0, y)
                new_w = min(new_w, image.shape[1] - x)
                new_h = min(new_h, image.shape[0] - y)
                
                # Crop the expanded face region
                face_crop = image[y:y+new_h, x:x+new_w]
                
                # Save the cropped face
                crop_path = os.path.join(crop_output_path, filename)
                cv2.imwrite(crop_path, face_crop)
                
                # Resize image
                resized_img = cv2.resize(face_crop, (resize_width, resize_height))
                resize_path = os.path.join(resize_output_path, filename)
                cv2.imwrite(resize_path, resized_img)
            else:
                print(f"No face detected in {filename}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Face detection, cropping, and resizing with CUDA support')
    parser.add_argument('--input_dir', type=str, required=True, help='Base input directory containing emotion folders')
    parser.add_argument('--output_dir', type=str, required=True, help='Base output directory for processed images')
    parser.add_argument('--emotions', type=str, nargs='+', help='Emotion folder names to process (default: all folders)')
    parser.add_argument('--scale_w', type=float, default=1.5, help='Width scaling factor for face bounding box')
    parser.add_argument('--scale_h', type=float, default=1.4, help='Height scaling factor for face bounding box')
    parser.add_argument('--neck_reduction', type=float, default=0.3, help='Factor to reduce neck visibility')
    parser.add_argument('--resize_width', type=int, default=600, help='Width to resize final images')
    parser.add_argument('--resize_height', type=int, default=600, help='Height to resize final images')
    
    args = parser.parse_args()
    
    # Enable CUDA for OpenCV if available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA-enabled devices found: {cv2.cuda.getCudaEnabledDeviceCount()}")
    else:
        print("No CUDA-enabled devices found. Running on CPU.")
    
    # Initialize MTCNN detector
    detector = MTCNN()
    
    # Get list of emotion folders to process
    if args.emotions:
        emotion_folders = args.emotions
        for emotion in emotion_folders:
            if not os.path.isdir(os.path.join(args.input_dir, emotion)):
                print(f"Warning: Folder '{emotion}' not found in {args.input_dir}")
    else:
        # Process all subdirectories in the input directory
        emotion_folders = [f for f in os.listdir(args.input_dir) 
                         if os.path.isdir(os.path.join(args.input_dir, f))]
    
    print(f"Processing {len(emotion_folders)} emotion folders: {emotion_folders}")
    
    # Process each emotion folder
    for emotion in emotion_folders:
        input_folder = os.path.join(args.input_dir, emotion)
        
        # Use custom output folder name if it's the "angry" emotion
        if emotion == "sad":
            crop_folder = "sadcropped"
            resize_folder = "sadcropped_resized"  # You could customize this too
        else:
            crop_folder = f"Cropped_faces/{emotion}"
            resize_folder = f"Resized_faces/{emotion}"
        
        process_emotion_folder(
            input_folder, 
            args.output_dir,
            crop_folder,
            resize_folder,
            detector,
            args.scale_w,
            args.scale_h,
            args.neck_reduction,
            args.resize_width,
            args.resize_height
        )
    
    print("Processing completed successfully!")

if __name__ == "__main__":
    main()