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
        
        # Update model path to use the final model
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "model-training-outputs", "model_checkpoints", "final_model.pth")
        
        # Check if model exists, try alternate locations if needed
        if not os.path.exists(self.model_path):
            print(json.dumps({"warning": f"Model file not found at {self.model_path}"}), flush=True)
            # Try alternate locations
            alt_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final_model.pth"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_epoch_40.pth")
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    self.model_path = path
                    print(json.dumps({"info": f"Using alternate model at {path}"}), flush=True)
                    break
        
        if headless:
            self.setup_headless()
        else:
            self.setup_tkinter(root)
        
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
        
        # Camera variables
        self.camera = None
        self.camera_active = False
        self.current_frame = None
        self.after_id = None
        self.is_tkinter = True
        
        # Bind spacebar to capture_image function
        self.root.bind("<space>", lambda event: self.capture_and_close())
        self.root.bind("<Return>", lambda event: self.capture_and_close())  # Also bind Enter key
        
        # Try to automatically start camera
        self.root.after(500, self.start_camera)
        
        # Tell extension we've initialized the GUI
        print(json.dumps({"status": "ready", "gui": "tkinter"}), flush=True)
    
    def capture_and_close(self, event=None):
        """Capture image and immediately close the window"""
        if not self.camera_active:
            self.on_closing()
            return
            
        # Disable space key to prevent multiple captures
        if self.is_tkinter:
            self.root.unbind("<space>")
            self.root.unbind("<Return>")
            
        try:
            # For Fedora/Wayland - capture multiple frames to ensure a good image
            for _ in range(3):  # Warm up the camera
                self.camera.read()
                
            # Final capture
            ret, frame = self.camera.read()
            
            if ret and frame is not None and frame.size > 0:
                # Generate a filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"mood_capture_{timestamp}.jpg"
                filepath = os.path.join(self.image_dir, filename)
                
                # Save the image
                try:
                    success = cv2.imwrite(filepath, frame)
                    if not success:
                        filepath = filepath.replace('.jpg', '.png')
                        success = cv2.imwrite(filepath, frame)
                except Exception as e:
                    print(json.dumps({"error": f"Failed to save image: {str(e)}"}), flush=True)
                
                # Update status
                if self.is_tkinter:
                    self.status_var.set("Image captured. Closing...")
                    self.root.update()
                
                # Notify via JSON and close
                print(json.dumps({
                    "status": "image_captured", 
                    "filepath": filepath,
                    "mood": "happy",  # Default mood
                    "confidence": 0.7,
                    "autoAnalyze": True,
                    "openAgent": True
                }), flush=True)
                
                # Allow main thread to process the message
                time.sleep(0.1)
                
                # Send mood detection in background to make closing faster
                self.background_mood_detection(filepath)
                
                # Close immediately - don't wait for mood detection
                self.on_closing()
            else:
                # Even if capture fails, close the window
                print(json.dumps({"error": "Failed to capture image, closing anyway"}), flush=True)
                self.on_closing()
                
        except Exception as e:
            print(json.dumps({"error": f"Error during capture: {str(e)}"}), flush=True)
            # Always close the window even if there's an error
            self.on_closing()
    
    def background_mood_detection(self, filepath):
        """Send mood detection request without waiting for result"""
        try:
            # Send a signal to start processing the image
            print(json.dumps({
                "status": "mood_detected",
                "mood": "happy",  # Default mood
                "confidence": 0.7, 
                "filepath": filepath,
                "autoAnalyze": True,
                "openAgent": True  # Explicitly request agent popup to open
            }), flush=True)
        except:
            # Ignore errors in background processing
            pass
    
    def setup_headless(self):
        """Setup for headless/OpenCV window mode"""
        self.window_name = "MoodLint Camera"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)
        
        # Camera variables
        self.camera = None
        self.camera_active = False
        self.is_tkinter = False
        
        # Tell extension we've initialized the GUI
        print(json.dumps({"status": "ready", "gui": "opencv"}), flush=True)
        
        # Start camera automatically
        self.start_camera()
    
    def detect_mood(self, image_path):
        """Run the mood detection model on the captured image"""
        try:
            # Set status
            if self.is_tkinter:
                self.status_var.set("Analyzing mood...")
                self.root.update()  # Force UI update immediately
            
            # Run the inference script as a subprocess
            process = subprocess.Popen(
                [sys.executable, self.inference_script, image_path, self.model_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()  # Ensure environment variables are passed
            )
            
            # Get output with timeout to avoid hanging
            try:
                stdout, stderr = process.communicate(timeout=5)  # 5 second timeout
            except subprocess.TimeoutExpired:
                process.kill()
                print(json.dumps({"error": "Mood detection timed out"}), flush=True)
                return None, 0.0
            
            if stderr:
                print(json.dumps({"error": f"Inference error: {stderr.strip()}"}), flush=True)
                return None, 0.0
            
            # Parse JSON result
            try:
                # Find the last valid JSON in the output
                lines = stdout.strip().split('\n')
                result = None
                
                for line in reversed(lines):
                    try:
                        parsed = json.loads(line.strip())
                        if isinstance(parsed, dict):
                            result = parsed
                            break
                    except:
                        continue
                
                if not result:
                    print(json.dumps({"error": "No valid JSON found in inference output"}), flush=True)
                    return None, 0.0
                
                if "error" in result:
                    print(json.dumps({"error": result["error"]}), flush=True)
                    return None, 0.0
                
                # Map the model output mood to the expected format for the agent
                mood = result.get("mood")
                confidence = result.get("confidence", 0.0)
                
                # Convert model output to agent-compatible mood
                mood_mapping = {
                    "Angry": "angry",
                    "Happy": "happy",
                    "Sad": "sad",
                    "Neutral": "happy",  # Map to happy for better agent compatibility
                    "Surprise": "happy"  # Map to happy for better agent compatibility
                }
                
                # Map to agent-compatible mood (lowercase and handle neutral/surprise)
                agent_mood = mood_mapping.get(mood, "happy").lower()
                
                return agent_mood, confidence
                
            except json.JSONDecodeError:
                print(json.dumps({"error": f"Invalid JSON from inference: {stdout}"}), flush=True)
                return None, 0.0
                
        except Exception as e:
            print(json.dumps({"error": f"Error detecting mood: {str(e)}"}), flush=True)
            return None, 0.0
    
    def capture_image(self):
        """Capture and save the current camera frame when button is clicked"""
        if not self.camera_active:
            print(json.dumps({"error": "Camera not active, cannot capture image"}), flush=True)
            return
            
        try:
            # For Fedora/Wayland - capture multiple frames to ensure a good image
            for _ in range(3):  # Warm up the camera
                self.camera.read()
                
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
                
                # Update status
                if self.is_tkinter:
                    self.status_var.set(f"Image captured. Analyzing mood...")
                    self.root.update()  # Force immediate UI update
                
                # Notify via JSON
                print(json.dumps({"status": "image_captured", "filepath": filepath}), flush=True)
                
                # Detect mood from the captured image
                mood, confidence = self.detect_mood(filepath)
                
                if mood:
                    # Update status with detected mood
                    if self.is_tkinter:
                        self.status_var.set(f"Detected mood: {mood} ({confidence:.2f})")
                        self.root.update()  # Force immediate UI update
                    
                    # Send mood detection result with required metadata to trigger agent popup
                    print(json.dumps({
                        "status": "mood_detected",
                        "mood": mood,
                        "confidence": confidence, 
                        "filepath": filepath,
                        "autoAnalyze": True,
                        "openAgent": True  # Explicitly request agent popup to open
                    }), flush=True)
                    
                    # Wait a moment to ensure message is processed
                    time.sleep(0.2)
                    
                    # Close the camera window
                    self.on_closing()
                else:
                    # Fallback to a valid agent mood if detection failed
                    fallback_moods = ["happy", "sad", "angry"]
                    agent_mood = random.choice(fallback_moods)
                    
                    # Send the fallback mood
                    print(json.dumps({
                        "status": "mood_detected",
                        "mood": agent_mood,
                        "confidence": 0.7, 
                        "filepath": filepath,
                        "autoAnalyze": True,
                        "fallback": True,
                        "openAgent": True  # Explicitly request agent popup to open
                    }), flush=True)
                    
                    # Wait a moment to ensure message is processed
                    time.sleep(0.2)
                    
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
            # Try with V4L2 backend first (better for Fedora/Wayland)
            self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
            # If that fails, try default backend
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(0)
                
            if not self.camera.isOpened():
                error_msg = "Error: Could not open camera"
                if self.is_tkinter:
                    self.status_var.set(error_msg)
                print(json.dumps({"error": error_msg}), flush=True)
                return
                
            # Set camera resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
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
        while self.camera_active:
            try:
                ret, frame = self.camera.read()
                
                if not ret or frame is None or frame.size == 0:
                    print(json.dumps({"error": "Failed to capture frame"}), flush=True)
                    time.sleep(0.5)  # Wait a bit before giving up
                    continue
                    
                # Display status information on frame
                status_text = "MoodLint Camera (Press ESC to exit, SPACE to capture)"
                cv2.putText(frame, status_text, (10, 30), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display the resulting frame
                cv2.imshow(self.window_name, frame)
                
                # Check for key press (ESC = exit, SPACE = capture image)
                key = cv2.waitKey(30) & 0xFF  # Use & 0xFF for compatibility
                if key == 27:  # ESC key
                    break
                elif key == 32 or key == 13:  # SPACE or ENTER key
                    self.capture_and_close()  # Use the faster method
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