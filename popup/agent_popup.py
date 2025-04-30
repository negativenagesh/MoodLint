import sys
import os
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, Entry
import threading
import traceback
import subprocess

# Check dependencies first before importing agent modules
def check_dependencies():
    missing_deps = []
    required_deps = ["langchain_core", "langchain", "langgraph", "google.generativeai"]
    
    for dep in required_deps:
        try:
            if "." in dep:
                module_name, submodule = dep.split(".", 1)
                __import__(module_name)
            else:
                __import__(dep)
        except ImportError:
            pkg_name = "google-generativeai" if dep == "google.generativeai" else dep
            missing_deps.append(pkg_name)
    
    return missing_deps

# Add parent directory to path so we can import from agents package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import agent manager, but handle import errors gracefully
try:
    from agents.agent_manager import AgentManager
    AGENT_IMPORT_SUCCESS = True
except ImportError as e:
    print(json.dumps({"error": f"Import error: {str(e)}"}), flush=True)
    AGENT_IMPORT_SUCCESS = False

class AgentDebugApp:
    def __init__(self, root, mood, filename=None, code=None, query=None):
        self.root = root
        self.mood = mood.lower()  # Ensure lowercase for mood normalization
        self.filename = filename
        self.code = code
        self.query = query
        
        # File selection and query UI elements
        self.file_path_var = tk.StringVar()
        self.query_var = tk.StringVar()
        
        # If filename is provided, set it in the variable
        if self.filename:
            self.file_path_var.set(self.filename)
        
        # If query is provided, set it in the variable
        if self.query:
            self.query_var.set(self.query)
        
        # Setup API key, defaulting to environment variable
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        
        # Initialize copy_button as None before setup_window
        self.copy_button = None
        self.analyze_button = None
        
        # Configure the window
        self.setup_window()
        
        # Check if dependencies are available
        missing_deps = check_dependencies()
        if missing_deps:
            install_cmd = f"pip install {' '.join(missing_deps)}"
            self.update_response_text(
                f"⚠️ Missing dependencies detected: {', '.join(missing_deps)}\n\n"
                f"Please install the required packages with:\n\n"
                f"{install_cmd}\n\n"
                f"Then restart VS Code and try again."
            )
            print(json.dumps({
                "status": "missing_dependencies",
                "missing": missing_deps,
                "install_command": install_cmd
            }), flush=True)
            return
            
        # Start analysis if we have all dependencies and code
        if AGENT_IMPORT_SUCCESS and self.code and self.filename:
            # Start analysis immediately for command-line mode
            self.start_analysis()
        elif not AGENT_IMPORT_SUCCESS:
            # Display error about agent import
            self.update_response_text(
                f"⚠️ Error loading MoodLint Agent system\n\n"
                f"There was an error importing the agent system components. "
                f"This may be due to missing dependencies.\n\n"
                f"Please make sure you have installed all required packages:\n"
                f"pip install langchain langchain_core langgraph google-generativeai"
            )
        else:
            # Display welcome message with instructions
            self.update_response_text(
                f"MoodLint Agent initialized with {self.mood} mood.\n\n"
                f"Please select a file to analyze using the browse button above, "
                f"optionally enter a query, and click 'Start Analysis'."
            )
        
    def setup_window(self):
        """Setup the UI elements"""
        self.root.title(f"MoodLint Agent - {self.mood.capitalize()} Mood")
        
        # Set window size and position
        window_width = 800
        window_height = 650  # Made taller to accommodate file selection UI
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Make window resizable
        self.root.resizable(True, True)
        
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
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=(10, 0))
        
        # File path entry and browse button
        ttk.Label(file_frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=60).grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        
        # Query frame
        query_frame = ttk.LabelFrame(main_frame, text="Query (Optional)", padding="10")
        query_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Query entry
        ttk.Label(query_frame, text="Query:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(query_frame, textvariable=self.query_var, width=60).grid(row=0, column=1, padx=5, sticky=tk.EW)
        
        # Analyze button
        self.analyze_button = ttk.Button(
            query_frame, 
            text="Start Analysis", 
            command=self.on_analyze_clicked
        )
        self.analyze_button.grid(row=0, column=2, padx=5)
        
        # Configure grid columns
        file_frame.columnconfigure(1, weight=1)
        query_frame.columnconfigure(1, weight=1)
        
        # Create the content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
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
        
        # Insert initial message
        self.update_response_text(f"Preparing to analyze with {self.mood} mood in mind...\n")
    
    def browse_file(self):
        """Open file browser dialog and update file path variable"""
        file_path = filedialog.askopenfilename(
            title="Select File to Analyze",
            filetypes=[
                ("Python Files", "*.py"),
                ("JavaScript Files", "*.js"),
                ("HTML Files", "*.html"),
                ("CSS Files", "*.css"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.filename = file_path
            self.status_var.set(f"File selected: {os.path.basename(file_path)}")
    
    def on_analyze_clicked(self):
        """Handle start analysis button click"""
        # Get current values from UI
        self.filename = self.file_path_var.get()
        self.query = self.query_var.get()
        
        if not self.filename or not os.path.exists(self.filename):
            self.update_response_text("Please select a valid file to analyze.")
            self.status_var.set("Error: No valid file selected")
            return
        
        # Read the code file
        try:
            with open(self.filename, 'r') as file:
                self.code = file.read()
            
            # Disable the analyze button during analysis
            self.analyze_button.config(state=tk.DISABLED)
            
            # Start the analysis
            self.start_analysis()
        except Exception as e:
            self.update_response_text(f"Error reading file: {str(e)}\nPlease select a different file.")
            self.status_var.set("Error reading file")
    
    def get_mood_color(self):
        """Return color for the current mood"""
        mood_colors = {
            "happy": "#32CD32",  # lime green
            "sad": "#4169E1",    # royal blue
            "angry": "#FF4500",  # orangered
            "frustrated": "#FFA500",  # orange
            "exhausted": "#800080",   # purple
            "neutral": "#708090"  # slate gray (added for neutral mood)
        }
        return mood_colors.get(self.mood.lower(), "#708090")  # Default to slate gray
    
    def update_response_text(self, text):
        """Update the response text area with new content"""
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, text)
        self.response_text.config(state=tk.DISABLED)
        self.response_text.see(1.0)  # Scroll to top
        
        # Enable copy button if we have a response and the button exists
        if text.strip() and hasattr(self, 'copy_button') and self.copy_button is not None:
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
        finally:
            # Re-enable the analyze button
            if hasattr(self, 'analyze_button') and self.analyze_button is not None:
                self.root.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))

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