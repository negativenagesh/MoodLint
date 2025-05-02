import os
import sys
import json
import tkinter as tk
from tkinter import ttk
import subprocess
import traceback
import time
from PIL import Image, ImageTk
from math import sin, pi

try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TKINTER = True
except ImportError:
    print(json.dumps({"error": "Tkinter modules not available. Please install python3-tk"}), flush=True)
    sys.exit(1)

try:
    from PIL import Image, ImageTk
    HAS_PIL_TK = True
except ImportError:
    print(json.dumps({"error": "PIL/Pillow ImageTk not available. Please install python3-pillow-tk"}), flush=True)
    sys.exit(1)

# Send startup message so extension knows we're running
print(json.dumps({"status": "starting", "gui": "initializing"}), flush=True)

# Define a color theme
COLORS = {
    "bg": "#f5f5f7",
    "primary": "#4a6fa5",
    "secondary": "#335c81",
    "accent": "#1e3a5f",
    "text": "#333333",
    "light_text": "#666666",
    "error": "#b23b3b",
    "success": "#32965d",
    "neutral": "#9b9b9b",
    # Mood-specific colors
    "happy": "#f4d03f",
    "sad": "#5dade2",
    "angry": "#e74c3c",
    "surprise": "#9b59b6",
    "neutral_mood": "#7f8c8d"
}

class AnimatedProgressBar(ttk.Frame):
    """Custom animated progress bar with gradient effect"""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.config(height=20)
        
        self.canvas = tk.Canvas(self, height=10, bd=0, highlightthickness=0, bg=COLORS["bg"])
        self.canvas.pack(fill=tk.X, expand=True, padx=2, pady=2)
        
        self.progress = 0
        self.is_running = False
        self.pulse_size = 0.2  # Size of the moving pulse as a fraction of the width
        
        # Create the base progress bar (gray background)
        self.base_rect = self.canvas.create_rectangle(0, 0, 0, 10, fill="#e0e0e0", width=0)
        
        # Create the progress rectangle
        self.rect = self.canvas.create_rectangle(0, 0, 0, 10, fill=COLORS["primary"], width=0)
        self.highlight = self.canvas.create_rectangle(0, 0, 0, 10, fill=COLORS["accent"], width=0, stipple="gray50")
        
        # Bind resize event
        self.canvas.bind("<Configure>", self._on_resize)
        
    def _on_resize(self, event):
        """Handle resize events"""
        width = event.width
        self.canvas.coords(self.base_rect, 0, 0, width, 10)
        self._update_progress()
        
    def _update_progress(self):
        """Update the progress bar visuals"""
        width = self.canvas.winfo_width()
        progress_width = int(width * self.progress)
        
        # Update main progress bar
        self.canvas.coords(self.rect, 0, 0, progress_width, 10)
        
        # Calculate highlight position for animation
        if self.is_running:
            pulse_width = int(width * self.pulse_size)
            pulse_pos = (self.animation_counter % 100) / 100 * (width - pulse_width)
            self.canvas.coords(self.highlight, pulse_pos, 0, pulse_pos + pulse_width, 10)
            self.canvas.itemconfig(self.highlight, state="normal")
        else:
            self.canvas.itemconfig(self.highlight, state="hidden")
    
    def start(self):
        """Start the indeterminate animation"""
        self.is_running = True
        self.animation_counter = 0
        self._animate()
    
    def stop(self):
        """Stop the animation"""
        self.is_running = False
        self.canvas.itemconfig(self.highlight, state="hidden")
    
    def _animate(self):
        """Animate the progress bar"""
        if not self.is_running:
            return
            
        self.animation_counter += 3
        self._update_progress()
        self.after(50, self._animate)
    
    def set_progress(self, value):
        """Set the progress value (0.0 to 1.0)"""
        self.progress = min(1.0, max(0.0, value))
        self._update_progress()
        
        # Change color based on progress
        if self.progress > 0.7:
            self.canvas.itemconfig(self.rect, fill=COLORS["success"])
        elif self.progress > 0.4:
            self.canvas.itemconfig(self.rect, fill=COLORS["primary"])
        else:
            self.canvas.itemconfig(self.rect, fill=COLORS["sad"])

class ImageSequenceFrame(ttk.LabelFrame):
    """Custom frame for displaying the image sequence with animations"""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Create the frame for the images
        self.images_frame = ttk.Frame(self)
        self.images_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        # Create labeled images
        self.image_frames = []
        self.canvases = []
        self.image_refs = []  # Keep references to images
        
        for i in range(5):
            frame = ttk.Frame(self.images_frame)
            frame.grid(row=0, column=i, padx=10)
            
            # Canvas for the image
            canvas = tk.Canvas(frame, width=120, height=120, bd=0, highlightthickness=1,
                              highlightbackground=COLORS["neutral"])
            canvas.pack()
            
            # Label with timestamp
            label = ttk.Label(frame, text=f"T-{5-i}", font=("Arial", 9))
            label.pack(pady=(5, 0))
            
            self.image_frames.append(frame)
            self.canvases.append(canvas)
            
    def set_images(self, image_paths):
        """Set the sequence of images with a fade-in effect"""
        self.image_refs = []  # Clear old references
        
        # Calculate number of empty slots to fill
        num_images = len(image_paths) if image_paths else 0
        num_empty = max(0, 5 - num_images)
        
        # Clear canvases
        for canvas in self.canvases:
            canvas.delete("all")
            
        # Add empty placeholders if needed
        for i in range(num_empty):
            self.canvases[i].create_rectangle(5, 5, 115, 115, fill="#f0f0f0", outline="")
            self.canvases[i].create_text(60, 60, text="No Data", fill=COLORS["light_text"])
            
        # Add actual images
        if image_paths:
            for i, path in enumerate(image_paths):
                canvas_idx = i + num_empty
                if path and os.path.exists(path):
                    try:
                        # Load and resize image
                        img = Image.open(path).convert('RGB')
                        img = img.resize((110, 110), Image.LANCZOS)
                        
                        # Create photo image and keep reference
                        photo = ImageTk.PhotoImage(img)
                        self.image_refs.append(photo)
                        
                        # Display with decorative border
                        self.canvases[canvas_idx].create_rectangle(4, 4, 116, 116, 
                                                                 outline=COLORS["accent"], width=2)
                        self.canvases[canvas_idx].create_image(60, 60, image=photo)
                    except Exception as e:
                        self.canvases[canvas_idx].create_rectangle(5, 5, 115, 115, fill="#fff0f0", outline="")
                        self.canvases[canvas_idx].create_text(60, 60, text="Error", fill=COLORS["error"])
                else:
                    self.canvases[canvas_idx].create_rectangle(5, 5, 115, 115, fill="#f0f0f0", outline="")
                    self.canvases[canvas_idx].create_text(60, 60, text="Missing", fill=COLORS["light_text"])

class MoodPredictionApp:
    def __init__(self, root, image_dir=None):
        self.root = root
        self.root.title("MoodLint Future Mood Prediction")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure the app to use the color theme
        self.style = ttk.Style()
        self.configure_style()
        
        # Make window appear on top initially
        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.attributes('-topmost', False)
        
        # Set window size and position
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 800) // 2
        y = (screen_height - 600) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # Image directory (containing recent mood captures)
        self.image_dir = image_dir
        
        # Setup UI
        self.setup_ui()
        
        # Start prediction after UI is initialized
        self.root.after(500, self.predict_mood)
        
        # Tell extension we've initialized
        print(json.dumps({"status": "ready", "gui": "tkinter"}), flush=True)
    
    def configure_style(self):
        """Configure the ttk style with our custom theme"""
        self.root.configure(bg=COLORS["bg"])
        
        # Try to use a modern theme as base
        try:
            self.style.theme_use('clam')
        except:
            pass
        
        # Configure frame styles
        self.style.configure('TFrame', background=COLORS["bg"])
        self.style.configure('TLabelframe', background=COLORS["bg"])
        self.style.configure('TLabelframe.Label', background=COLORS["bg"], foreground=COLORS["text"], font=('Arial', 12, 'bold'))
        
        # Configure label styles
        self.style.configure('TLabel', background=COLORS["bg"], foreground=COLORS["text"], font=('Arial', 11))
        self.style.configure('Header.TLabel', font=('Arial', 18, 'bold'), foreground=COLORS["accent"])
        self.style.configure('Subheader.TLabel', font=('Arial', 12), foreground=COLORS["secondary"])
        self.style.configure('Result.TLabel', font=('Arial', 16, 'bold'), foreground=COLORS["primary"])
        
        # Configure button styles
        self.style.configure('TButton', font=('Arial', 11))
        self.style.configure('Primary.TButton', background=COLORS["primary"], foreground="white")
        
        # Configure progress bar styles
        self.style.configure("Horizontal.TProgressbar", 
                            background=COLORS["primary"],
                            troughcolor=COLORS["bg"],
                            borderwidth=0)
        
    def setup_ui(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(header_frame, text="Future Mood Prediction", 
                 style='Header.TLabel').pack()
        ttk.Label(header_frame, text="Using LSTM sequence model to predict your next mood",
                 style='Subheader.TLabel').pack()
        
        # Sequence images frame
        self.sequence_frame = ImageSequenceFrame(main_frame, text="Your Recent Mood Sequence")
        self.sequence_frame.pack(fill=tk.X, padx=5, pady=10)
            
        # Prediction result frame
        result_frame = ttk.LabelFrame(main_frame, text="Predicted Future Mood")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        # Prediction result
        result_inner_frame = ttk.Frame(result_frame)
        result_inner_frame.pack(fill=tk.X, pady=20, padx=20)
        
        self.mood_icon_canvas = tk.Canvas(result_inner_frame, width=64, height=64, 
                                        bg=COLORS["bg"], highlightthickness=0)
        self.mood_icon_canvas.pack(side=tk.LEFT, padx=(0, 15))
        self.draw_neutral_face(self.mood_icon_canvas)
        
        self.result_text = tk.StringVar(value="Analyzing your mood sequence...")
        result_label = ttk.Label(result_inner_frame, textvariable=self.result_text, 
                               style='Result.TLabel')
        result_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Confidence bar
        confidence_frame = ttk.Frame(result_frame)
        confidence_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(confidence_frame, text="Prediction Confidence:").pack(anchor=tk.W)
        
        self.confidence_bar = AnimatedProgressBar(confidence_frame)
        self.confidence_bar.pack(fill=tk.X, pady=5)
        self.confidence_bar.set_progress(0.1)
        self.confidence_bar.start()
        
        # Details text
        explanation_frame = ttk.Frame(result_frame)
        explanation_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.explanation_text = tk.Text(explanation_frame, height=6, wrap=tk.WORD, 
                                     font=("Arial", 12), borderwidth=1, relief="solid",
                                     padx=10, pady=10)
        self.explanation_text.pack(fill=tk.BOTH, expand=True)
        self.explanation_text.insert(tk.END, "Analyzing your mood pattern from recent captures...\n\n"
                                   "The LSTM neural network is examining your sequence of emotions "
                                   "to predict your future mood state.")
        self.explanation_text.config(state=tk.DISABLED)
        
        # Status and progress
        self.status_var = tk.StringVar(value="Initializing prediction model...")
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.progress = AnimatedProgressBar(status_frame)
        self.progress.pack(pady=5, fill=tk.X)
        self.progress.start()
        
        self.status = ttk.Label(status_frame, textvariable=self.status_var)
        self.status.pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Close", command=self.on_closing, style='Primary.TButton').pack()
    
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
            
            # Display images in the sequence frame
            self.sequence_frame.set_images(sequence_images)
            
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
        self.confidence_bar.set_progress(confidence)
        
        # Update mood icon
        self.draw_mood_icon(mood.lower())
        
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
    
    def draw_mood_icon(self, mood):
        """Draw a mood icon based on the predicted mood"""
        canvas = self.mood_icon_canvas
        canvas.delete("all")
        
        # Get mood color
        color = COLORS.get(mood, COLORS["neutral_mood"])
        
        # Draw face background circle
        canvas.create_oval(5, 5, 59, 59, fill=color, outline=COLORS["secondary"], width=2)
        
        if mood == "happy":
            # Draw happy face (smile)
            canvas.create_arc(15, 20, 49, 54, start=0, extent=-180, style="arc", width=3, outline="black")
            # Eyes
            canvas.create_oval(20, 20, 26, 26, fill="black")
            canvas.create_oval(38, 20, 44, 26, fill="black")
        elif mood == "sad":
            # Draw sad face (frown)
            canvas.create_arc(15, 35, 49, 55, start=0, extent=180, style="arc", width=3, outline="black")
            # Eyes
            canvas.create_oval(20, 20, 26, 26, fill="black")
            canvas.create_oval(38, 20, 44, 26, fill="black")
        elif mood == "angry":
            # Draw angry face
            canvas.create_arc(15, 35, 49, 55, start=0, extent=180, style="arc", width=3, outline="black")
            # Eyebrows
            canvas.create_line(15, 15, 25, 20, width=3, fill="black")
            canvas.create_line(49, 15, 39, 20, width=3, fill="black")
            # Eyes
            canvas.create_oval(20, 23, 26, 29, fill="black")
            canvas.create_oval(38, 23, 44, 29, fill="black")
        elif mood == "surprise":
            # Draw surprised face (O mouth)
            canvas.create_oval(25, 35, 39, 49, fill="black")
            # Eyes
            canvas.create_oval(20, 20, 26, 26, fill="black")
            canvas.create_oval(38, 20, 44, 26, fill="black")
            # Raised eyebrows
            canvas.create_line(15, 13, 25, 16, width=2, fill="black")
            canvas.create_line(49, 13, 39, 16, width=2, fill="black")
        else:
            # Draw neutral face (straight line)
            canvas.create_line(20, 40, 44, 40, width=3, fill="black")
            # Eyes
            canvas.create_oval(20, 20, 26, 26, fill="black")
            canvas.create_oval(38, 20, 44, 26, fill="black")
    
    def draw_neutral_face(self, canvas):
        """Draw a neutral face for initial state"""
        canvas.delete("all")
        
        # Draw face background circle
        canvas.create_oval(5, 5, 59, 59, fill=COLORS["neutral_mood"], outline=COLORS["secondary"], width=2)
        
        # Draw neutral face (straight line)
        canvas.create_line(20, 40, 44, 40, width=3, fill="black")
        
        # Eyes
        canvas.create_oval(20, 20, 26, 26, fill="black")
        canvas.create_oval(38, 20, 44, 26, fill="black")
    
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
        
        # Update result text
        self.result_text.set("Prediction Error")
        
        # Draw error icon
        self.mood_icon_canvas.delete("all")
        self.mood_icon_canvas.create_oval(5, 5, 59, 59, fill="#ffeeee", outline=COLORS["error"], width=2)
        self.mood_icon_canvas.create_text(32, 32, text="!", font=("Arial", 24, "bold"), fill=COLORS["error"])
        
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
        
        # Create application
        app = MoodPredictionApp(root, image_dir)
        
        # Start the main loop
        root.mainloop()
    except Exception as e:
        print(json.dumps({"error": f"Critical error: {str(e)}"}), flush=True)
        traceback_str = traceback.format_exc()
        print(json.dumps({"trace": traceback_str}), flush=True)
        sys.exit(1)