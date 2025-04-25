import face_recognition
import cv2
import numpy as np
import os
import dlib

# Check if CUDA is available for GPU usage
if dlib.cuda.get_num_devices() > 0:
    print("CUDA is available, using GPU.")
else:
    print("CUDA is not available, using CPU.")

def crop_face_to_original_size(image_path):
    """
    Crop the face from an image and resize it to the original image dimensions with padding,
    maintaining the aspect ratio.
    """
    # Load the image
    image = face_recognition.load_image_file(image_path)
    original_height, original_width = image.shape[:2]

    # Detect face locations using CNN model for GPU support
    face_locations = face_recognition.face_locations(image, model="cnn")

    if len(face_locations) == 0:
        print(f"No face found in {image_path}")
        return None

    # Find the largest face based on area
    areas = [(bottom - top) * (right - left) for top, right, bottom, left in face_locations]
    largest_face_index = np.argmax(areas)
    top, right, bottom, left = face_locations[largest_face_index]

    # Crop the face
    face_image = image[top:bottom, left:right]
    face_height, face_width = face_image.shape[:2]

    # Calculate scaling factor to fit within original dimensions while maintaining aspect ratio
    scale_width = original_width / face_width
    scale_height = original_height / face_height
    scale = min(scale_width, scale_height)

    # Resize the face image
    new_width = int(face_width * scale)
    new_height = int(face_height * scale)
    resized_face = cv2.resize(face_image, (new_width, new_height))

    # Create a new image with original dimensions, filled with black
    new_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # Calculate offsets to center the resized face
    x_offset = (original_width - new_width) // 2
    y_offset = (original_height - new_height) // 2

    # Place the resized face in the new image
    new_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_face

    return new_image

def process_images(image_paths, output_dir):
    """
    Process a list of images, cropping faces and saving the results.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for image_path in image_paths:
        new_image = crop_face_to_original_size(image_path)
        if new_image is not None:
            # Generate output file path
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{name}_cropped{ext}")

            # Save the image, converting from RGB to BGR for OpenCV
            cv2.imwrite(output_path, cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
            print(f"Saved cropped image to {output_path}")

# Example usage
if __name__ == "__main__":
    # Replace with your list of image paths and output directory
    image_paths = ["image1.jpg", "image2.jpg"]  # Example paths
    output_dir = "output_cropped_images"
    process_images(image_paths, output_dir)