import sys
import os
import json
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import traceback

# Add parent directory to path so we can import from agents package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.agent_manager import AgentManager

class AgentDebugApp:
    def __init__(self, root, mood, filename=None, code=None, query=None):
        self.root = root
        self.mood = mood.lower()  # Ensure lowercase for mood normalization
        self.filename = filename
        self.code = code
        self.query = query
        
        # Setup API key, defaulting to environment variable
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        
        # Configure the window
        self.setup_window()
        
        # Start analysis in a separate thread to keep UI responsive
        if self.code and self.filename:
            # Start analysis immediately
            self.start_analysis()
        else:
            # Display welcome message if no code provided
            self.update_response_text(
                f"MoodLint Agent initialized with {self.mood} mood.\n\n"
                f"No code file was provided for analysis. Please open a code file and try again."
            )
        
    def setup_window(self):
        """Setup the UI elements"""
        self.root.title(f"MoodLint Agent - {self.mood.capitalize()} Mood")
        
        # Set window size and position
        window_width = 800
        window_height = 600
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Make window non-resizable
        self.root.resizable(False, False)
        
        # Set mood-specific color for header
        mood_color = self.get_mood_color()
        
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with mood indication
        header_frame = tk.Frame(main_frame, bg=mood_color, padx=10, pady=10)
        header_frame.pack(fill=tk.X)
        
        tk.Label(
            header_frame, 
            text=f"MoodLint Agent: {self.mood.capitalize()} Mood", 
            font=("Arial", 16, "bold"),
            bg=mood_color,
            fg="white"
        ).pack(side=tk.LEFT)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Create the content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Code info frame (if applicable)
        if self.filename:
            info_frame = ttk.Frame(content_frame)
            info_frame.pack(fill=tk.X, pady=(0, 10))
            ttk.Label(info_frame, text=f"File: {self.filename}").pack(anchor=tk.W)
            if self.query:
                ttk.Label(info_frame, text=f"Query: {self.query}").pack(anchor=tk.W)
        
        # Response text area with custom styling
        self.response_text = scrolledtext.ScrolledText(
            content_frame, 
            wrap=tk.WORD, 
            font=("Consolas", 11),
            background="#f8f8f8", 
            padx=10,
            pady=10
        )
        self.response_text.pack(fill=tk.BOTH, expand=True)
        self.response_text.config(state=tk.DISABLED)  # Make it read-only initially
        
        # Insert initial message
        self.update_response_text(f"Preparing to analyze with {self.mood} mood in mind...\n")
        
        # Bottom button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Add copy button
        self.copy_button = ttk.Button(
            button_frame, 
            text="Copy to Clipboard", 
            command=self.copy_to_clipboard,
            state=tk.DISABLED
        )
        self.copy_button.pack(side=tk.LEFT)
        
        # Add close button
        ttk.Button(
            button_frame, 
            text="Close", 
            command=self.close_app
        ).pack(side=tk.RIGHT)
    
    def get_mood_color(self):
        """Return color for the current mood"""
        mood_colors = {
            "happy": "#32CD32",  # lime green
            "sad": "#4169E1",    # royal blue
            "angry": "#FF4500",  # orangered
            "frustrated": "#FFA500",  # orange
            "exhausted": "#800080"   # purple
        }
        return mood_colors.get(self.mood.lower(), "#000000")
    
    def update_response_text(self, text):
        """Update the response text area with new content"""
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, text)
        self.response_text.config(state=tk.DISABLED)
        self.response_text.see(1.0)  # Scroll to top
        
        # Enable copy button if we have a response
        if text.strip():
            self.copy_button.config(state=tk.NORMAL)
    
    def copy_to_clipboard(self):
        """Copy the response to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(self.response_text.get(1.0, tk.END))
        self.status_var.set("Copied to clipboard!")
    
    def close_app(self):
        """Close the application"""
        print(json.dumps({"status": "closed"}), flush=True)
        self.root.destroy()
    
    def start_analysis(self):
        """Start code analysis in a separate thread"""
        self.status_var.set("Analyzing code...")
        self.update_response_text("Analyzing your code with the mood-aware agent...\nPlease wait...")
        
        # Start analysis in a separate thread
        threading.Thread(target=self.perform_analysis, daemon=True).start()
    
    def perform_analysis(self):
        """Perform the code analysis and update UI with results"""
        try:
            # Create agent manager
            agent_manager = AgentManager(api_key=self.api_key)
            
            # Send message that analysis is starting
            print(json.dumps({"status": "analyzing"}), flush=True)
            
            # Get debugging results from appropriate mood agent
            result = agent_manager.debug_code(
                code=self.code,
                filename=self.filename,
                mood=self.mood,
                user_query=self.query or ""
            )
            
            # Update UI with response
            if result["success"]:
                self.root.after(0, lambda: self.status_var.set("Analysis complete"))
                self.root.after(0, lambda: self.update_response_text(result["response"]))
                
                # Send success result back to extension
                print(json.dumps({
                    "status": "complete", 
                    "result": result
                }), flush=True)
            else:
                error_message = result.get("error", "Unknown error")
                self.root.after(0, lambda: self.status_var.set(f"Error: {error_message}"))
                self.root.after(0, lambda: self.update_response_text(result["response"]))
                
                # Send error result back to extension
                print(json.dumps({
                    "status": "error", 
                    "message": error_message,
                    "result": result
                }), flush=True)
                
        except Exception as e:
            error_message = str(e)
            error_trace = traceback.format_exc()
            self.root.after(0, lambda: self.status_var.set(f"System error occurred"))
            self.root.after(0, lambda: self.update_response_text(
                f"An error occurred while analyzing your code:\n\n{error_message}\n\n"
                f"This might be due to a system issue or invalid input. "
                f"Please check your code and try again."
            ))
            
            # Send error back to extension
            print(json.dumps({
                "status": "error", 
                "message": error_message,
                "traceback": error_trace
            }), flush=True)

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing mood argument"}), flush=True)
        return
    
    # Get mood from arguments
    mood = sys.argv[1]
    
    # Optional filename, code, and query
    filename = sys.argv[2] if len(sys.argv) > 2 else None
    query = sys.argv[3] if len(sys.argv) > 3 else None
    
    # If filename is provided, load code from file
    code = None
    if filename and os.path.exists(filename):
        try:
            with open(filename, 'r') as file:
                code = file.read()
        except Exception as e:
            print(json.dumps({"error": f"Error reading file: {str(e)}"}), flush=True)
            return
    
    # Initialize Tkinter
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    try:
        style.theme_use('clam')  # Use a modern theme
    except:
        pass  # Fall back to default theme
    
    # Create app
    app = AgentDebugApp(root, mood, filename, code, query)
    
    # Start Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": f"Critical error: {str(e)}"}), flush=True)
        sys.exit(1)