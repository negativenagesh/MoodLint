#!/usr/bin/env python3
import os
import sys
import json
import tkinter as tk
from tkinter import ttk
import subprocess
import traceback
import time
from PIL import Image, ImageTk

# First check for tkinter and show a clear error if it's missing
try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TKINTER = True
except ImportError:
    print(json.dumps({"error": "Tkinter modules not available. Please install python3-tk"}), flush=True)
    sys.exit(1)

# Now try to import PIL
try:
    from PIL import Image, ImageTk
    HAS_PIL_TK = True
except ImportError:
    print(json.dumps({"error": "PIL/Pillow ImageTk not available. Please install python3-pillow-tk"}), flush=True)
    sys.exit(1)

# Send startup message so extension knows we're running
print(json.dumps({"status": "starting", "gui": "initializing"}), flush=True)

class MoodPredictionApp:
    def __init__(self, root, image_dir=None):
        self.root = root
        self.root.title("MoodLint Future Mood Prediction")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Make window appear on top initially
        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.attributes('-topmost', False)
        
        # Set window size and position
        self.root.geometry("700x550")
        self.root.resizable(False, False)
        
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 700) // 2
        y = (screen_height - 550) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # Image directory (containing recent mood captures)
        self.image_dir = image_dir
        
        # Setup UI
        self.setup_ui()
        
        # Start prediction after UI is initialized
        self.root.after(500, self.predict_mood)
        
        # Tell extension we've initialized
        print(json.dumps({"status": "ready", "gui": "tkinter"}), flush=True)
        
    def setup_ui(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(header_frame, text="Future Mood Prediction", 
                 font=("Arial", 18, "bold")).pack()
        ttk.Label(header_frame, text="Using LSTM sequence model to predict your next mood").pack()
        
        # Sequence images frame
        sequence_frame = ttk.LabelFrame(main_frame, text="Your Recent Mood Sequence")
        sequence_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Sequence images canvas (horizontal scrolling)
        self.sequence_canvas_frame = ttk.Frame(sequence_frame)
        self.sequence_canvas_frame.pack(fill=tk.X, pady=10)
        
        # Create 5 small canvases for the sequence
        self.sequence_canvases = []
        for i in range(5):
            canvas = tk.Canvas(self.sequence_canvas_frame, width=100, height=100, bg="black")
            canvas.grid(row=0, column=i, padx=5)
            canvas.create_text(50, 50, text=f"Image {i+1}", fill="white", font=("Arial", 10))
            self.sequence_canvases.append(canvas)
            
        # Prediction result frame
        result_frame = ttk.LabelFrame(main_frame, text="Predicted Future Mood")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        # Prediction result
        self.result_text = tk.StringVar(value="Analyzing your mood sequence...")
        result_label = ttk.Label(result_frame, textvariable=self.result_text, 
                               font=("Arial", 16))
        result_label.pack(pady=20)
        
        # Confidence bar
        confidence_frame = ttk.Frame(result_frame)
        confidence_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(confidence_frame, text="Prediction Confidence:").pack(anchor=tk.W)
        
        confidence_bar_frame = ttk.Frame(confidence_frame)
        confidence_bar_frame.pack(fill=tk.X, pady=5)
        
        self.confidence_bar = ttk.Progressbar(confidence_bar_frame, length=600)
        self.confidence_bar.pack(fill=tk.X)
        
        # Details text
        explanation_frame = ttk.Frame(result_frame)
        explanation_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.explanation_text = tk.Text(explanation_frame, height=6, wrap=tk.WORD, 
                                     font=("Arial", 12))
        self.explanation_text.pack(fill=tk.BOTH, expand=True)
        self.explanation_text.insert(tk.END, "Analyzing your mood pattern from recent captures...\n\n"
                                   "The LSTM neural network is examining your sequence of emotions "
                                   "to predict your future mood state.")
        self.explanation_text.config(state=tk.DISABLED)
        
        # Status and progress
        self.status_var = tk.StringVar(value="Initializing prediction model...")
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.progress = ttk.Progressbar(status_frame, orient="horizontal", length=660, mode="indeterminate")
        self.progress.pack(pady=5)
        self.progress.start()
        
        self.status = ttk.Label(status_frame, textvariable=self.status_var)
        self.status.pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Close", command=self.on_closing).pack()
    
    def predict_mood(self):
        """Run the lstm_popup.py script to predict the next mood"""
        try:
            self.status_var.set("Finding recent mood images...")
            
            if not self.image_dir or not os.path.exists(self.image_dir):
                self.show_error("Cannot find the image directory with your mood captures.")
                return
            
            # Show the images in the UI
            self.display_image_sequence()
            
            # Get path to lstm_popup.py script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            lstm_script = os.path.join(script_dir, "lstm_popup.py")
            
            if not os.path.exists(lstm_script):
                self.show_error(f"LSTM prediction script not found at: {lstm_script}")
                return
            
            # Execute with Python
            python_exe = sys.executable
            cmd = [python_exe, lstm_script, self.image_dir]
            
            self.status_var.set("Running LSTM mood prediction...")
            
            # Run the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor output
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                
                output_lines.append(line.strip())
                print(json.dumps({"debug": f"LSTM output: {line.strip()}"}), flush=True)
                
                try:
                    result = json.loads(line.strip())
                    
                    # Update progress in UI
                    if "progress" in result:
                        self.status_var.set(result["progress"])
                    
                    # If we have a successful prediction, update the UI
                    if result.get("status") == "success" and "mood" in result:
                        self.update_prediction_result(result)
                        
                    # Handle errors
                    if "error" in result:
                        self.show_error(result["error"])
                        
                except json.JSONDecodeError:
                    # Not a JSON line, just ignore
                    pass
                
            # Wait for process to finish
            process.wait()
            
            # Check return code
            if process.returncode != 0:
                stderr = process.stderr.read()
                self.show_error(f"Prediction failed: {stderr}")
                
            # Stop progress bar when done
            self.progress.stop()
            
        except Exception as e:
            error_msg = f"Error running mood prediction: {str(e)}"
            print(json.dumps({"error": error_msg}), flush=True)
            print(json.dumps({"trace": traceback.format_exc()}), flush=True)
            self.show_error(error_msg)
            self.progress.stop()
    
    def display_image_sequence(self):
        """Find and display the sequence of mood images"""
        try:
            # Find image files in directory
            image_files = []
            for file in sorted(os.listdir(self.image_dir)):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(self.image_dir, file))
            
            # Get the most recent 5 (or fewer if we don't have 5)
            sequence_images = image_files[-5:] if len(image_files) >= 5 else image_files
            
            # Add placeholders if we don't have 5 images
            while len(sequence_images) < 5:
                sequence_images.insert(0, None)
            
            # Display the images
            self.image_references = []  # Keep references to prevent garbage collection
            for i, img_path in enumerate(sequence_images):
                if img_path:
                    try:
                        img = Image.open(img_path)
                        img = img.resize((100, 100), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        self.image_references.append(photo)
                        
                        # Clear canvas and display image
                        self.sequence_canvases[i].delete("all")
                        self.sequence_canvases[i].create_image(50, 50, image=photo)
                    except Exception as e:
                        print(json.dumps({"error": f"Error loading image {img_path}: {str(e)}"}), flush=True)
                        self.sequence_canvases[i].delete("all")
                        self.sequence_canvases[i].create_text(50, 50, text="Error", fill="red")
            
        except Exception as e:
            print(json.dumps({"error": f"Error displaying image sequence: {str(e)}"}), flush=True)
    
    def update_prediction_result(self, result):
        """Update UI with prediction result"""
        # Stop the progress bar
        self.progress.stop()
        
        # Get prediction results
        mood = result.get("mood", "Unknown")
        confidence = result.get("confidence", 0)
        message = result.get("message", f"You're likely to feel {mood} soon.")
        
        # Update result text
        self.result_text.set(f"Your future mood: {mood.upper()}")
        
        # Update confidence bar
        self.confidence_bar["value"] = confidence * 100
        
        # Update explanation text
        self.explanation_text.config(state=tk.NORMAL)
        self.explanation_text.delete(1.0, tk.END)
        self.explanation_text.insert(tk.END, message + "\n\n")
        
        # Add detailed explanation based on mood
        explanation = self.get_mood_explanation(mood)
        self.explanation_text.insert(tk.END, explanation)
        self.explanation_text.config(state=tk.DISABLED)
        
        # Update status
        self.status_var.set(f"Prediction complete: {mood} (confidence: {confidence:.2f})")
        
        # Notify extension
        print(json.dumps({
            "status": "prediction_complete",
            "mood": mood,
            "confidence": confidence,
            "message": message
        }), flush=True)
    
    def get_mood_explanation(self, mood):
        """Get a detailed explanation for the predicted mood"""
        mood = mood.lower()
        
        if mood == "angry":
            return ("The LSTM model detected a pattern leading to frustration or anger. "
                   "Consider taking short breaks when coding to prevent burnout.")
                   
        elif mood == "happy":
            return ("Your mood sequence suggests a positive trajectory. This is a great "
                   "time for creative problem-solving and tackling complex challenges.")
                   
        elif mood == "sad":
            return ("The model predicts you may experience a downturn in mood. Consider "
                   "working on structured, well-defined tasks rather than open-ended problems.")
                   
        elif mood == "surprise":
            return ("Your emotional sequence indicates you might experience unexpected "
                   "breakthroughs or realizations in your work soon.")
                   
        elif mood == "neutral":
            return ("Your predicted neutral mood is ideal for methodical debugging and "
                   "systematic work that requires focus and attention to detail.")
                   
        else:
            return ("Based on your recent mood sequence, the LSTM neural network "
                   "has predicted your future emotional state. Consider how this "
                   "might affect your programming productivity.")
    
    def show_error(self, error_message):
        """Display an error in the UI"""
        self.progress.stop()
        self.status_var.set(f"Error: {error_message}")
        
        # Update explanation text
        self.explanation_text.config(state=tk.NORMAL)
        self.explanation_text.delete(1.0, tk.END)
        self.explanation_text.insert(tk.END, f"Error: {error_message}\n\n")
        self.explanation_text.insert(tk.END, "Please try again or check the logs for more details.")
        self.explanation_text.config(state=tk.DISABLED)
        
        # Notify extension
        print(json.dumps({"error": error_message}), flush=True)
    
    def on_closing(self):
        """Handle window closing"""
        print(json.dumps({"status": "closed"}), flush=True)
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing image directory. Usage: lstm_popup_ui.py <image_directory>"}), flush=True)
        sys.exit(1)
    
    image_dir = sys.argv[1]
    
    try:
        # Create the main window
        root = tk.Tk()
        
        # Set theme
        style = ttk.Style()
        try:
            style.theme_use('clam')  # Use a modern theme
        except:
            pass  # Fall back to default theme if 'clam' is not available
        
        # Create application
        app = MoodPredictionApp(root, image_dir)
        
        # Start the main loop
        root.mainloop()
    except Exception as e:
        print(json.dumps({"error": f"Critical error: {str(e)}"}), flush=True)
        traceback_str = traceback.format_exc()
        print(json.dumps({"trace": traceback_str}), flush=True)
        sys.exit(1)