import tkinter as tk
from tkinter import ttk, scrolledtext
import json
import sys
import threading
import time
import os
import asyncio

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the debug_code function from the agent workflow
from agents.workflow import debug_code

class AgentDebugApp:
    def __init__(self, root, code, filename, mood, query=""):
        self.root = root
        self.code = code
        self.filename = filename
        self.mood = mood
        self.query = query
        self.is_complete = False
        
        # Set up the UI
        self.setup_ui()
        
        # Start agent processing in a separate thread
        self.agent_thread = threading.Thread(target=self.run_agent_workflow)
        self.agent_thread.daemon = True
        self.agent_thread.start()
        
        # Start progress animation
        self.update_progress()

    def setup_ui(self):
        self.root.title(f"MoodLint Agent - {self.mood.capitalize()} Mode")
        self.root.geometry("700x600")
        self.root.minsize(600, 500)
        
        # Make window appear on top
        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.attributes('-topmost', False)
        
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 700) // 2
        y = (screen_height - 600) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # Configure style
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Mood.TLabel', font=('Arial', 12), foreground=self.get_mood_color())
        
        # Create main container
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header_frame, text="MoodLint Debugging Assistant", style='Header.TLabel').pack(side=tk.LEFT)
        self.mood_label = ttk.Label(header_frame, text=f"Mood: {self.mood.capitalize()}", style='Mood.TLabel')
        self.mood_label.pack(side=tk.RIGHT)
        
        # Create file info frame
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(file_frame, text=f"File: {self.filename}").pack(anchor=tk.W)
        
        # Create progress frame
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.StringVar(value="Initializing agent...")
        self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.LEFT)
        
        self.progress_dots = tk.StringVar(value="")
        self.progress_dots_label = ttk.Label(self.progress_frame, textvariable=self.progress_dots)
        self.progress_dots_label.pack(side=tk.LEFT)
        
        # Create result text area
        result_frame = ttk.LabelFrame(main_frame, text="Analysis Results")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=20)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.config(state=tk.DISABLED)
        
        # Create buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.close_button = ttk.Button(button_frame, text="Close", command=self.on_close)
        self.close_button.pack(side=tk.RIGHT)

    def get_mood_color(self):
        mood_colors = {
            "happy": "#4CAF50",    # Green
            "frustrated": "#FF9800", # Orange
            "exhausted": "#9C27B0", # Purple
            "sad": "#2196F3",       # Blue
            "angry": "#F44336"      # Red
        }
        return mood_colors.get(self.mood.lower(), "#000000")
    
    def update_progress(self):
        """Update the progress animation"""
        if not self.is_complete:
            # Update the dots animation
            dots = self.progress_dots.get()
            if len(dots) >= 3:
                dots = ""
            else:
                dots += "."
            self.progress_dots.set(dots)
            
            # Schedule the next update
            self.root.after(500, self.update_progress)
    
    def run_agent_workflow(self):
        """Run the agent workflow in a separate thread"""
        try:
            # Update status
            self.update_status("Analyzing code...")
            
            # Create event loop for async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run debug_code function
            result = loop.run_until_complete(debug_code(
                code=self.code,
                filename=self.filename,
                mood=self.mood,
                query=self.query
            ))
            
            # Update UI with results
            self.root.after(0, lambda: self.update_results(result))
        except Exception as e:
            # Handle errors
            error_message = f"Error during analysis: {str(e)}"
            self.root.after(0, lambda: self.update_error(error_message))
    
    def update_status(self, message):
        """Update the status label"""
        self.root.after(0, lambda: self.progress_var.set(message))
    
    def update_results(self, result):
        """Update the results panel with agent output"""
        self.is_complete = True
        self.progress_var.set("Analysis complete")
        self.progress_dots.set("")
        
        # Enable text widget for editing
        self.result_text.config(state=tk.NORMAL)
        
        # Clear any existing text
        self.result_text.delete(1.0, tk.END)
        
        # Insert the response
        if result.get("success", False) and result.get("response"):
            self.result_text.insert(tk.END, result["response"])
        else:
            error = result.get("error", "Unknown error occurred")
            self.result_text.insert(tk.END, f"Error: {error}")
        
        # Disable editing
        self.result_text.config(state=tk.DISABLED)
        
        # Scroll to the top
        self.result_text.see("1.0")
        
        # Send result back to VSCode via stdout
        print(json.dumps({"status": "complete", "result": result}), flush=True)
    
    def update_error(self, error_message):
        """Display an error in the results panel"""
        self.is_complete = True
        self.progress_var.set("Analysis failed")
        self.progress_dots.set("")
        
        # Enable text widget for editing
        self.result_text.config(state=tk.NORMAL)
        
        # Clear any existing text
        self.result_text.delete(1.0, tk.END)
        
        # Insert the error message
        self.result_text.insert(tk.END, error_message)
        
        # Disable editing
        self.result_text.config(state=tk.DISABLED)
        
        # Send error back to VSCode via stdout
        print(json.dumps({"status": "error", "message": error_message}), flush=True)
    
    def on_close(self):
        """Handle window close"""
        if not self.is_complete:
            print(json.dumps({"status": "canceled"}), flush=True)
        self.root.destroy()
        sys.exit(0)

def main():
    """Main function to parse arguments and start the application"""
    if len(sys.argv) < 4:
        print(json.dumps({
            "status": "error", 
            "message": "Usage: agent_popup.py <code_file> <filename> <mood> [query]"
        }), flush=True)
        sys.exit(1)
    
    code_file = sys.argv[1]
    filename = sys.argv[2]
    mood = sys.argv[3]
    query = sys.argv[4] if len(sys.argv) > 4 else ""
    
    # Read the code file
    try:
        with open(code_file, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Error reading code file: {str(e)}"}), flush=True)
        sys.exit(1)
    
    # Initialize the UI
    print(json.dumps({"status": "starting"}), flush=True)
    
    root = tk.Tk()
    app = AgentDebugApp(root, code, filename, mood, query)
    
    print(json.dumps({"status": "ready"}), flush=True)
    
    # Start the UI event loop
    try:
        root.mainloop()
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"UI error: {str(e)}"}), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()