import face_recognition
import cv2
import numpy as np
import os
import dlib
import glob

if dlib.cuda.get_num_devices() > 0:
    print("CUDA is available, using GPU.")
else:
    print("CUDA is not available, using CPU.")

def crop_face(image_path, neck_ratio=0.5, target_size=(224, 224)):
    """
    Crop the face from an image including the neck area and resize to target dimensions.
    
    Args:
        image_path: Path to the image file
        neck_ratio: Ratio to extend below the face to include neck (0.5 = 50% of face height)
        target_size: Tuple (width, height) for the output image size
    """
    image = face_recognition.load_image_file(image_path)
    original_height, original_width = image.shape[:2]

    face_locations = face_recognition.face_locations(image, model="cnn")

    if len(face_locations) == 0:
        print(f"No face found in {image_path}")
        return None

    areas = [(bottom - top) * (right - left) for top, right, bottom, left in face_locations]
    largest_face_index = np.argmax(areas)
    top, right, bottom, left = face_locations[largest_face_index]
    
    # Calculate face height and width
    face_height = bottom - top
    face_width = right - left
    
    # Extend bottom to include neck
    extended_bottom = min(bottom + int(face_height * neck_ratio), original_height)
    
    # Extract face with neck
    face_image = image[top:extended_bottom, left:right]
    
    # Resize to target dimensions for CNN input
    face_image_resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_AREA)
    
    return face_image_resized

def get_image_files(directory):
    """
    Get all image files from a directory.
    """
    # Common image file extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        # Also check subdirectories if needed
        image_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    
    return image_files

def process_images(input_path, output_dir, neck_ratio=0.5, target_size=(224, 224)):
    """
    Process images from a directory or list of image paths, cropping faces and saving the results.
    
    Args:
        input_path: Directory or list of image paths
        output_dir: Directory to save processed images
        neck_ratio: Ratio to extend below the face to include neck
        target_size: Tuple (width, height) for the output image size
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input_path is a directory or a list of files
    if isinstance(input_path, str) and os.path.isdir(input_path):
        image_paths = get_image_files(input_path)
        print(f"Found {len(image_paths)} images in directory: {input_path}")
    else:
        # Assuming it's a list of image paths
        image_paths = input_path if isinstance(input_path, list) else [input_path]

    for image_path in image_paths:
        print(f"Processing: {image_path}")
        face_image = crop_face(image_path, neck_ratio, target_size)
        if face_image is not None:
            # Generate output file path
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{name}_cropped{ext}")

            # Save the image, converting from RGB to BGR for OpenCV
            cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            print(f"Saved cropped image to {output_path}")

# Example usage
if __name__ == "__main__":
    # You can provide either a directory containing images:
    input_directory = "expression/angry"  # Directory containing images
    output_dir = "facerecognition/angrycropped"
    
    # Adjust neck_ratio to include more or less of the neck
    # 0.5 means extend by 50% of face height, increase for more neck area
    # Standard size for many CNN models is 224x224
    process_images(input_directory, output_dir, neck_ratio=0.5, target_size=(224, 224))