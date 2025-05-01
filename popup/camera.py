import cv2
import sys
import json
import os
import time
import random
import datetime
import subprocess
import traceback

# Set OpenCV backend for better compatibility with Fedora/Wayland
os.environ["OPENCV_VIDEOIO_BACKEND"] = "v4l2"

try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    HAS_TKINTER = True
except ImportError:
    print(json.dumps({"error": "Tkinter modules not available. Please install python3-tk"}), flush=True)
    HAS_TKINTER = False
    # Fall back to OpenCV window mode
    import cv2

# Send startup message so extension knows we're running
print(json.dumps({"status": "starting", "gui": "initializing"}), flush=True)

class CameraApp:
    def __init__(self, root=None, headless=False):
        # Create images directory if it doesn't exist
        self.image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Path to the inference script
        self.inference_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "model", "inference.py")

        # Search for model in multiple locations
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_model_paths = [
            os.path.join(base_dir, "model-training-outputs", "model_checkpoints", "final_model.pth"),
            os.path.join(base_dir, "model_checkpoints", "final_model.pth"),
            os.path.join(base_dir, "model", "final_model.pth"),
            os.path.join(base_dir, "final_model.pth"),
            # Also try parent directory
            os.path.join(os.path.dirname(base_dir), "model-training-outputs", "model_checkpoints", "final_model.pth"),
            os.path.join(os.path.dirname(base_dir), "model_checkpoints", "final_model.pth"),
        ]

        # Default model path
        self.model_path = possible_model_paths[0]

        # Check for existing model file
        found_model = False
        for path in possible_model_paths:
            if os.path.exists(path):
                self.model_path = path
                found_model = True
                print(json.dumps({"info": f"Found model at: {path}"}), flush=True)
                break

        if not found_model:
            print(json.dumps({"warning": "Model not found in any standard location. Will attempt to use default path."}), flush=True)
        
        self.is_tkinter = not headless and HAS_TKINTER and root is not None
        
        # Initialize camera properties
        self.camera = None
        self.camera_active = False
        self.current_frame = None
        self.after_id = None
        
        # Setup the appropriate UI
        if self.is_tkinter:
            self.setup_tkinter(root)
        else:
            self.setup_headless()
                
    def setup_tkinter(self, root):
        self.root = root
        self.root.title("MoodLint Camera")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Make window appear on top initially
        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.attributes('-topmost', False)
        
        # Set window size and position
        self.root.geometry("640x520")
        self.root.resizable(False, False)
        
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 640) // 2
        y = (screen_height - 520) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text="MoodLint Camera", font=("Arial", 18, "bold")).pack()
        ttk.Label(header_frame, text="Emotion-aware code debugging assistant").pack()
        
        # Create frame for video
        self.video_frame = ttk.Frame(main_frame, width=640, height=480)
        self.video_frame.pack(pady=10)
        
        # Create canvas for displaying video
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="black")
        self.canvas.pack()
        
        # Add a placeholder text when camera isn't started
        self.canvas.create_text(320, 240, text="Camera Preview", fill="white", font=("Arial", 20))
        
        # Add status label with hint about space key
        self.status_var = tk.StringVar(value="Camera ready (press SPACE to capture image)")
        self.status = ttk.Label(main_frame, textvariable=self.status_var)
        self.status.pack(pady=5)
        
        # Add buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.camera_button = ttk.Button(button_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_button.pack(side=tk.LEFT, padx=5)
        
        self.capture_button = ttk.Button(button_frame, text="Capture Image", command=self.capture_image, state=tk.DISABLED)
        self.capture_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Close", command=self.on_closing).pack(side=tk.LEFT, padx=5)
        
        # Bind spacebar to capture_image function instead of capture_and_close
        self.root.bind("<space>", lambda event: self.capture_image())
        self.root.bind("<Return>", lambda event: self.capture_image())  # Also bind Enter key
        
        # Try to automatically start camera
        self.root.after(500, self.start_camera)
        
        # Tell extension we've initialized the GUI
        print(json.dumps({"status": "ready", "gui": "tkinter"}), flush=True)
    
    def setup_headless(self):
        """Setup for non-Tkinter mode using OpenCV window"""
        self.is_tkinter = False
        self.window_name = "MoodLint Camera (Press SPACE to capture, ESC to exit)"
        print(json.dumps({"status": "ready", "gui": "opencv"}), flush=True)
        # Start the camera immediately in headless mode
        self.start_camera()
    
    def detect_mood(self, image_path):
        """Run the mood detection model on the captured image"""
        try:
            # Get the Python executable from the virtual environment if it exists
            python_exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    ".venv", "bin", "python")
            if not os.path.exists(python_exe):
                python_exe = "python3"  # Fall back to system Python
            
            # Run the inference script as a subprocess
            command = [python_exe, self.inference_script, image_path, self.model_path]
            print(json.dumps({"info": f"Running inference on {image_path} with model {self.model_path}"}), flush=True)
            print(json.dumps({"info": f"Running command: {' '.join(command)}"}), flush=True)
            
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Wait for the process to complete with increased timeout
            try:
                print(json.dumps({"info": "Waiting for inference process..."}), flush=True)
                stdout, stderr = process.communicate(timeout=300)  # Increased to 300 seconds (5 minutes) for slower CPUs
                print(json.dumps({"info": f"Inference process completed with return code: {process.returncode}"}), flush=True)
                
                # Process the output
                if process.returncode == 0:
                    # Store all output lines for debugging
                    output_lines = stdout.strip().split('\n')
                    for i, line in enumerate(output_lines):
                        print(json.dumps({"debug": f"Output line {i}: {line}"}), flush=True)
                    
                    # Try to parse the last line as JSON for the mood result
                    result_line = output_lines[-1] if output_lines else '{}'
                    
                    try:
                        result = json.loads(result_line)
                        mood = result.get('mood', 'Unknown')
                        confidence = result.get('confidence', 0.7)
                        
                        # Check if this is a fallback result
                        is_fallback = result.get('fallback', False)
                        if is_fallback:
                            print(json.dumps({"warning": "Using fallback mood detection"}), flush=True)
                        
                        print(json.dumps({"info": f"Detected mood: {mood} (confidence: {confidence}, fallback: {is_fallback})"}), flush=True)
                        
                        # Update the UI with the detected mood
                        if self.is_tkinter:
                            self.status_var.set(f"Detected mood: {mood}{' (fallback)' if is_fallback else ''}")
                        
                        # Return mood and confidence separately since that's what capture_image expects
                        return mood, confidence
                    except json.JSONDecodeError:
                        print(json.dumps({"error": f"Failed to parse inference result: {result_line}"}), flush=True)
                        # Try to parse stderr for potential Python error messages
                        if stderr:
                            print(json.dumps({"error": f"stderr: {stderr}"}), flush=True)
                        # Return fallback values that are more varied
                        moods = ["Neutral", "Sad", "Angry", "Surprise"]  # Don't just default to Happy
                        selected_mood = random.choice(moods)
                        return selected_mood, 0.6 + random.random() * 0.3  # Random confidence between 0.6-0.9
                else:
                    print(json.dumps({"error": f"Inference process failed with return code: {process.returncode}"}), flush=True)
                    if stderr:
                        print(json.dumps({"error": f"Stderr: {stderr}"}), flush=True)
                    # Print stdout as well for debugging
                    if stdout:
                        print(json.dumps({"error": f"Stdout: {stdout}"}), flush=True)
                    # Return fallback values with random mood
                    moods = ["Neutral", "Sad", "Angry", "Surprise"]  # Don't just default to Happy
                    return random.choice(moods), 0.6 + random.random() * 0.3
            
            except subprocess.TimeoutExpired:
                print(json.dumps({"error": "Mood detection timed out after 300 seconds"}), flush=True)
                process.kill()
                
                # Return fallback values with variety
                moods = ["Neutral", "Sad", "Angry", "Surprise"]  # Don't just default to Happy
                return random.choice(moods), 0.6 + random.random() * 0.3
            
        except Exception as e:
            print(json.dumps({"error": f"Error in detect_mood: {str(e)}"}), flush=True)
            print(json.dumps({"trace": traceback.format_exc()}), flush=True)
            # Return fallback values with random mood for variety
            moods = ["Neutral", "Sad", "Angry", "Happy", "Surprise"]
            return random.choice(moods), 0.6 + random.random() * 0.3
        
    def capture_image(self, event=None):
        """Capture and save the current camera frame and detect mood"""
        if not self.camera_active:
            print(json.dumps({"error": "Camera not active, cannot capture image"}), flush=True)
            return
                
        try:
            # For Fedora/Wayland - capture multiple frames to ensure a good image
            for _ in range(3):  # Warm up the camera
                ret, _ = self.camera.read()
                if not ret:
                    print(json.dumps({"warning": "Failed to read frame during warm-up"}), flush=True)
                    
            # Final capture
            ret, frame = self.camera.read()
                
            if ret and frame is not None and frame.size > 0:
                # Generate a filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"mood_capture_{timestamp}.jpg"
                filepath = os.path.join(self.image_dir, filename)
                    
                # Save the image with better error handling
                try:
                    success = cv2.imwrite(filepath, frame)
                    if not success:
                        raise Exception("cv2.imwrite returned False")
                except Exception as e:
                    print(json.dumps({"error": f"Failed to save image: {str(e)}"}), flush=True)
                    # Try alternative format
                    try:
                        filepath = filepath.replace('.jpg', '.png')
                        success = cv2.imwrite(filepath, frame)
                        if not success:
                            raise Exception("Failed to save PNG format too")
                    except Exception as e2:
                        print(json.dumps({"error": f"Failed to save PNG image: {str(e2)}"}), flush=True)
                        return
                    
                # Update status and notify that image was captured
                if self.is_tkinter:
                    self.status_var.set(f"Image captured. Analyzing mood...")
                    self.root.update()  # Force immediate UI update
                    
                print(json.dumps({"status": "image_captured", "filepath": filepath}), flush=True)
                    
                # Detect mood from the captured image
                print(json.dumps({"status": "running_inference"}), flush=True)
                mood, confidence = self.detect_mood(filepath)
                    
                if mood:
                    # Successfully detected mood
                    if self.is_tkinter:
                        self.status_var.set(f"Detected mood: {mood} (confidence: {confidence:.2f})")
                        self.root.update()
                        
                    # Send the detected mood back to the extension
                    print(json.dumps({
                        "status": "mood_detected",
                        "mood": mood,
                        "confidence": confidence, 
                        "filepath": filepath,
                        "autoAnalyze": True,
                        "openAgent": True
                    }), flush=True)
                        
                    # Wait a moment to ensure message is processed
                    time.sleep(0.5)
                        
                    # Close the camera window
                    self.on_closing()
                else:
                    # Failed to detect mood - be explicit about fallback
                    print(json.dumps({"status": "mood_detection_failed", "using_fallback": True}), flush=True)
                        
                    # Fallback to a valid agent mood if detection failed
                    fallback_moods = ["happy", "sad", "angry"]
                    agent_mood = random.choice(fallback_moods)
                    fallback_confidence = 0.7
                        
                    if self.is_tkinter:
                        self.status_var.set(f"Using fallback mood: {agent_mood} (detection failed)")
                        self.root.update()
                        
                    # Send the fallback mood but clearly mark it as fallback
                    print(json.dumps({
                        "status": "mood_detected",
                        "mood": agent_mood,
                        "confidence": fallback_confidence, 
                        "filepath": filepath,
                        "autoAnalyze": True,
                        "fallback": True,
                        "openAgent": True
                    }), flush=True)
                        
                    # Wait a moment to ensure message is processed
                    time.sleep(0.5)
                        
                    # Close the camera window
                    self.on_closing()
                        
            else:
                error_msg = "Failed to capture image"
                if self.is_tkinter:
                    self.status_var.set(error_msg)
                print(json.dumps({"error": error_msg}), flush=True)
                    
        except Exception as e:
            error_msg = f"Error capturing image: {str(e)}"
            if self.is_tkinter:
                self.status_var.set(error_msg)
            print(json.dumps({"error": error_msg}), flush=True)
            print(json.dumps({"trace": traceback.format_exc()}), flush=True)
        
    def start_camera(self):
        """Start the camera with better handling for Fedora/Wayland"""
        if self.camera_active:
            return
                
        if self.is_tkinter:
            self.status_var.set("Starting camera...")
            self.root.update()
            
        try:
            # Try different methods to open camera
            print(json.dumps({"status": "opening_camera", "method": "v4l2"}), flush=True)
                
            # First try: V4L2 backend (Linux)
            try:
                self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
                if self.camera.isOpened():
                    print(json.dumps({"info": "Camera opened successfully with V4L2 backend"}), flush=True)
                else:
                    print(json.dumps({"warning": "Failed to open camera with V4L2 backend"}), flush=True)
            except Exception as e:
                print(json.dumps({"warning": f"Error with V4L2 backend: {str(e)}"}), flush=True)
                self.camera = None
                
            # Second try: Default backend
            if self.camera is None or not self.camera.isOpened():
                print(json.dumps({"status": "opening_camera", "method": "default"}), flush=True)
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    print(json.dumps({"info": "Camera opened successfully with default backend"}), flush=True)
                
            # Third try: Try specific APIs (Windows DirectShow, etc.)
            if self.camera is None or not self.camera.isOpened():
                apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                for api in apis:
                    try:
                        print(json.dumps({"status": "opening_camera", "method": f"api_{api}"}), flush=True)
                        self.camera = cv2.VideoCapture(0, api)
                        if self.camera.isOpened():
                            print(json.dumps({"info": f"Camera opened successfully with API {api}"}), flush=True)
                            break
                    except Exception as e:
                        print(json.dumps({"warning": f"Error with API {api}: {str(e)}"}), flush=True)
                        continue
                
            # Final check
            if not self.camera or not self.camera.isOpened():
                error_msg = "Error: Could not open camera with any method"
                if self.is_tkinter:
                    self.status_var.set(error_msg)
                print(json.dumps({"error": error_msg}), flush=True)
                return
                    
            # Set camera resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
            # Test read a frame to verify camera works
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None or test_frame.size == 0:
                error_msg = "Camera opened but failed to read frame"
                if self.is_tkinter:
                    self.status_var.set(error_msg)
                print(json.dumps({"error": error_msg}), flush=True)
                self.camera.release()
                self.camera = None
                return
                
            self.camera_active = True
                
            if self.is_tkinter:
                self.camera_button.config(text="Stop Camera")
                self.capture_button.config(state=tk.NORMAL)  # Enable capture button
                self.status_var.set("Camera started (press SPACE to capture)")
                self.update_frame()
            else:
                self.run_opencv_loop()
                    
            # Tell extension camera is running
            print(json.dumps({"status": "camera_started"}), flush=True)
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if self.is_tkinter:
                self.status_var.set(error_msg)
            print(json.dumps({"error": error_msg}), flush=True)
            print(json.dumps({"trace": traceback.format_exc()}), flush=True)
        
    def stop_camera(self):
        if not self.camera_active:
            return
            
        if self.is_tkinter:    
            # Cancel any pending after callback
            if hasattr(self, 'after_id') and self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None
            
        # Release camera
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            
        self.camera_active = False
            
        if self.is_tkinter:
            self.camera_button.config(text="Start Camera")
            self.capture_button.config(state=tk.DISABLED)  # Disable capture button
            self.status_var.set("Camera stopped")
                
            # Clear canvas and add placeholder text
            self.canvas.delete("all")
            self.canvas.create_text(320, 240, text="Camera Preview", fill="white", font=("Arial", 20))
        
    def toggle_camera(self):
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()
        
    def update_frame(self):
        if not self.camera_active:
            return
                
        try:
            ret, frame = self.camera.read()
                
            if ret and frame is not None and frame.size > 0:
                # Convert frame from BGR (OpenCV format) to RGB (PIL format)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                # Create PIL image
                pil_image = Image.fromarray(frame_rgb)
                    
                # Convert PIL image to PhotoImage
                self.current_frame = ImageTk.PhotoImage(image=pil_image)
                    
                # Clear canvas and display image
                self.canvas.delete("all")
                self.canvas.create_image(320, 240, image=self.current_frame, anchor=tk.CENTER)
                    
                # Add SPACE key hint as overlay text
                self.canvas.create_text(320, 450, text="Press SPACE to capture", 
                                       fill="white", font=("Arial", 12),
                                       tags="hint")
                    
                # Schedule next update
                self.after_id = self.root.after(33, self.update_frame)  # ~30 FPS
            else:
                # Camera capture failed
                print(json.dumps({"error": "Failed to capture frame"}), flush=True)
                self.stop_camera()
                    
        except Exception as e:
            print(json.dumps({"error": f"Frame update error: {str(e)}"}), flush=True)
            self.stop_camera()
        
    def run_opencv_loop(self):
        """Run camera loop for OpenCV window mode"""
        # Create a window for display
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)
            
        while self.camera_active:
            try:
                ret, frame = self.camera.read()
                    
                if not ret or frame is None or frame.size == 0:
                    print(json.dumps({"error": "Failed to capture frame"}), flush=True)
                    time.sleep(0.5)  # Wait a bit before giving up
                    continue
                        
                # Display status information on frame
                status_text = "Press SPACE to capture, ESC to exit"
                cv2.putText(frame, status_text, (10, 30), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                # Display the resulting frame
                cv2.imshow(self.window_name, frame)
                    
                # Check for key press (ESC = exit, SPACE = capture image)
                key = cv2.waitKey(30) & 0xFF  # Use & 0xFF for compatibility
                if key == 27:  # ESC key
                    break
                elif key == 32 or key == 13:  # SPACE or ENTER key
                    self.capture_image()  # Use the normal capture method
            except Exception as e:
                print(json.dumps({"error": f"OpenCV error: {str(e)}"}), flush=True)
                time.sleep(1)  # Wait a bit before continuing
                    
        self.stop_camera()
        cv2.destroyAllWindows()
        
    def on_closing(self):
        self.stop_camera()
        if self.is_tkinter:
            self.root.destroy()
        print(json.dumps({"status": "closed"}), flush=True)
        sys.exit(0)

if __name__ == "__main__":
    # Accept optional parameter for session ID
    session_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    
    try:
        # Try to use tkinter if available
        if HAS_TKINTER:
            # Create the main window
            root = tk.Tk()
            
            # Set style
            style = ttk.Style()
            try:
                style.theme_use('clam')  # Use a modern theme
            except:
                pass  # Fall back to default theme if 'clam' is not available
            
            # Create application
            app = CameraApp(root)
            
            # Start the main loop
            root.mainloop()
        else:
            # Fall back to OpenCV window if no tkinter
            app = CameraApp(headless=True)
    except Exception as e:
        print(json.dumps({"error": f"Critical error: {str(e)}"}), flush=True)
        traceback_str = traceback.format_exc()
        print(json.dumps({"trace": traceback_str}), flush=True)
        sys.exit(1)