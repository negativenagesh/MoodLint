#!/usr/bin/env python3

import cv2
import sys
import json
import os
import time

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
        
        # Add status label
        self.status_var = tk.StringVar(value="Camera ready")
        self.status = ttk.Label(main_frame, textvariable=self.status_var)
        self.status.pack(pady=5)
        
        # Add buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.camera_button = ttk.Button(button_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Close", command=self.on_closing).pack(side=tk.LEFT, padx=5)
        
        # Camera variables
        self.camera = None
        self.camera_active = False
        self.current_frame = None
        self.after_id = None
        self.is_tkinter = True
        
        # Try to automatically start camera
        self.root.after(500, self.start_camera)
        
        # Tell extension we've initialized the GUI
        print(json.dumps({"status": "ready", "gui": "tkinter"}), flush=True)
    
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
    
    def start_camera(self):
        if self.camera_active:
            return
            
        if self.is_tkinter:
            self.status_var.set("Starting camera...")
        
        try:
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
                self.status_var.set("Camera started")
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
            if self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        self.camera_active = False
        
        if self.is_tkinter:
            self.camera_button.config(text="Start Camera")
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
            
            if ret:
                # Convert frame from BGR (OpenCV format) to RGB (PIL format)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create PIL image
                pil_image = Image.fromarray(frame_rgb)
                
                # Convert PIL image to PhotoImage
                self.current_frame = ImageTk.PhotoImage(image=pil_image)
                
                # Clear canvas and display image
                self.canvas.delete("all")
                self.canvas.create_image(320, 240, image=self.current_frame, anchor=tk.CENTER)
                
                # Send mock detection data (you can add actual detection code here)
                # For now, we'll just print to stdout as a placeholder
                if time.time() % 5 < 0.1:  # Approximately every 5 seconds
                    print(json.dumps({"mood": "focused", "confidence": 0.85}), flush=True)
                
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
        last_mood_time = 0
        
        while self.camera_active:
            try:
                ret, frame = self.camera.read()
                
                if not ret:
                    print(json.dumps({"error": "Failed to capture frame"}), flush=True)
                    break
                    
                # Display status information on frame
                status_text = "MoodLint Camera (Press ESC to exit)"
                cv2.putText(frame, status_text, (10, 30), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display the resulting frame
                cv2.imshow(self.window_name, frame)
                
                # Send mock detection data approximately every 5 seconds
                current_time = time.time()
                if current_time - last_mood_time >= 5:
                    print(json.dumps({"mood": "focused", "confidence": 0.85}), flush=True)
                    last_mood_time = current_time
                
                # Check for key press (ESC = exit)
                key = cv2.waitKey(30)
                if key == 27:  # ESC key
                    break
            except Exception as e:
                print(json.dumps({"error": f"OpenCV error: {str(e)}"}), flush=True)
                break
                
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
        sys.exit(1)