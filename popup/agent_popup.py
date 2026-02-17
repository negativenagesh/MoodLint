import sys
import os
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, Entry
import threading
import traceback
import asyncio
import re

# Add parent directory to path so we can import from agents package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import DiffViewer - try relative import first, fall back to absolute
try:
    from .diff_viewer import DiffViewer
except ImportError:
    # If relative import fails (when run as script), use absolute import
    from diff_viewer import DiffViewer

# Load the API key from .env file directly
def load_api_key_from_dotenv():
    """Load API key directly from .env file"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    api_key = None
    
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as file:
                for line in file:
                    # Look for OPENAI_API_KEY=value
                    match = re.match(r'^OPENAI_API_KEY=(.+)$', line.strip())
                    if match:
                        api_key = match.group(1)
                        print(f"Loaded API key from .env file: {api_key[:7]}...{api_key[-4:]} (length: {len(api_key)})")
                        break
        except Exception as e:
            print(f"Error reading .env file: {str(e)}")
    
    return api_key

# Get the API key before any class definition
OPENAI_API_KEY = load_api_key_from_dotenv() or os.environ.get("OPENAI_API_KEY")

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
        
        # Setup API key, using the global variable we loaded from .env
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            print("WARNING: No OpenAI API key found in .env file or environment variables")
        else:
            print(f"Using API key: {self.api_key[:7]}...{self.api_key[-4:]} (length: {len(self.api_key)})")
        
        # Initialize copy_button as None before setup_window
        self.copy_button = None
        self.analyze_button = None
        
        # Configure the window
        self.setup_window()
            
        # Start analysis if we have code and filename
        if self.code and self.filename:
            # Start analysis immediately for command-line mode
            self.start_analysis()
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
        file_entry = tk.Entry(file_frame, textvariable=self.file_path_var, width=60, 
                              insertbackground="#000000", insertwidth=2)
        file_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        
        # Query frame
        query_frame = ttk.LabelFrame(main_frame, text="Query (Optional)", padding="10")
        query_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Query entry with visible cursor bar
        ttk.Label(query_frame, text="Query:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        query_entry = tk.Entry(query_frame, textvariable=self.query_var, width=60,
                               insertbackground="#000000", insertwidth=2)
        query_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        
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
            foreground="#000000",
            padx=10,
            pady=10
        )
        self.response_text.pack(fill=tk.BOTH, expand=True)
        self.response_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.response_text.config(state=tk.DISABLED)  # Make it read-only initially
        
        # Frame for tool results (Diff views, etc.)
        self.tool_results_frame = ttk.Frame(content_frame)
        self.tool_results_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        
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
        
        if (not self.filename or not os.path.exists(self.filename)) and not self.query:
            self.update_response_text("Please select a file to analyze or enter a query.")
            self.status_var.set("Error: No file or query provided")
            return
        
        # Read the code file if provided
        try:
            if self.filename and os.path.exists(self.filename):
                with open(self.filename, 'r') as file:
                    self.code = file.read()
            else:
                self.code = "" # No code context
                self.filename = "No File Selected" # Placeholder
            
            # Disable the analyze button during analysis
            self.analyze_button.config(state=tk.DISABLED)
            
            # Start the analysis
            self.start_analysis()
            
            # Clear previous tool results
            for widget in self.tool_results_frame.winfo_children():
                widget.destroy()
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
        
    def append_stream_chunk(self, chunk):
        """Append streamed text to the response area safely in main thread"""
        def _update():
            try:
                self.response_text.config(state=tk.NORMAL)
                self.response_text.insert(tk.END, chunk)
                self.response_text.config(state=tk.DISABLED)
                self.response_text.see(tk.END)
            except Exception:
                pass # Ignore errors if window closed
        self.root.after(0, _update)
    
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
            # Check for API key
            if not self.api_key:
                raise ValueError("No OpenAI API key available. Please set OPENAI_API_KEY in .env file or environment.")
            
            # Try to use tool-based debugging
            try:
                from agents.mood_agents.base_agent import MoodAgent
                from agents.mood_agents.happy_agent import HappyAgent
                from agents.mood_agents.sad_agent import SadAgent
                from agents.mood_agents.angry_agent import AngryAgent
                from agents.mood_agents.neutral_agent import NeutralAgent
                from agents.mood_agents.surprise_agent import SurpriseAgent
                
                print(f"Using tool-based debugging for {self.filename} with mood: {self.mood}")
                
                # Create appropriate mood agent
                agent_map = {
                    "happy": HappyAgent,
                    "sad": SadAgent,
                    "angry": AngryAgent,
                    "neutral": NeutralAgent,
                    "surprise": SurpriseAgent
                }
                
                agent_class = agent_map.get(self.mood.lower(), NeutralAgent)
                agent = agent_class(api_key=self.api_key)
                
                # Use tool-based debugging
                
                # Define output directory for generated files
                file_dir = os.path.dirname(os.path.abspath(__file__))
                root_dir = os.path.dirname(file_dir)
                generated_dir = os.path.join(root_dir, "generated")
                
                # Ensure directory exists
                if not os.path.exists(generated_dir):
                    try:
                        os.makedirs(generated_dir)
                        print(f"Created generated directory: {generated_dir}")
                    except Exception as e:
                        print(f"Error creating generated directory: {str(e)}")
                
                # Clear text initially to prepare for streaming
                self.root.after(0, lambda: self.update_response_text(""))
                
                result = agent.debug_code_with_tools(
                    code=self.code,
                    filename=self.filename,
                    user_query=self.query or "Analyze this code and suggest improvements",
                    output_dir=generated_dir,
                    stream_callback=self.append_stream_chunk
                )
                
                # Update UI with response
                if result.get("success"):
                    response_text = result.get("response", "No response generated")
                    tool_calls = result.get("tool_calls", [])
                    
                    # Update response text
                    self.root.after(0, lambda: self.status_var.set(f"Analysis complete ({result.get('iterations', 0)} iterations)"))
                    self.root.after(0, lambda: self.update_response_text(response_text))
                    
                    # Display tool execution results
                    if tool_calls:
                        self.root.after(0, lambda: self.display_tool_results(tool_calls))
                    
                    # Send success result
                    print(json.dumps({
                        "status": "complete",
                        "result": {
                            "success": True,
                            "response": response_text,
                            "tool_calls": len(tool_calls),
                            "mood": self.mood
                        }
                    }), flush=True)
                else:
                    error_msg = result.get("error", "Unknown error")
                    self.root.after(0, lambda: self.status_var.set(f"Error: {error_msg}"))
                    self.root.after(0, lambda: self.update_response_text(f"Error: {error_msg}"))
                    
            except Exception as e:
                # Fallback to old method if tool method fails
                error_trace = traceback.format_exc()
                print(f"Tool-based debugging failed, falling back: {str(e)}\n{error_trace}")
                
                from agents.utils.openai_client import OpenAIClient
                client = OpenAIClient(api_key=self.api_key)
                
                # Create event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run async analysis
                result = loop.run_until_complete(
                    client.analyze_code(
                        code=self.code,
                        filename=self.filename,
                        mood=self.mood,
                        query=self.query or ""
                    )
                )
                
                if result.get("success"):
                    self.root.after(0, lambda: self.status_var.set("Analysis complete"))
                    self.root.after(0, lambda: self.update_response_text(result.get("response", "")))
                    
                    print(json.dumps({
                        "status": "complete",
                        "result": result
                    }), flush=True)
                else:
                    error_msg = result.get("error", "Analysis failed")
                    self.root.after(0, lambda: self.status_var.set(f"Error: {error_msg}"))
                    self.root.after(0, lambda: self.update_response_text(result.get("response", error_msg)))
                    
        except Exception as e:
            error_message = str(e)
            error_trace = traceback.format_exc()
            self.root.after(0, lambda: self.status_var.set("Analysis failed"))
            
            error_response = (
                f"# Analysis Error\n\n"
                f"I encountered an error while analyzing your code:\n\n"
                f"```\n{error_message}\n```\n\n"
                f"Please check that all dependencies are installed and try again."
            )
            
            self.root.after(0, lambda: self.update_response_text(error_response))
            
            print(json.dumps({
                "status": "error",
                "result": {
                    "success": False,
                    "response": "Analysis failed.",
                    "error": error_message,
                    "traceback": error_trace
                }
            }), flush=True)
        finally:
            # Re-enable the analyze button
            if hasattr(self, 'analyze_button') and self.analyze_button is not None:
                self.root.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))
    
    def display_tool_results(self, tool_calls):
        """Display results of tool executions, including diff views."""
        for tool_call in tool_calls:
            tool_name = tool_call["tool"]
            result = tool_call["result"]
            
            if tool_name == "edit_file" and result.get("success"):
                # Show diff view
                diff_viewer = DiffViewer(
                    self.tool_results_frame,
                    filepath=result["filepath"],
                    original=result["original_content"],
                    modified=result["new_content"]
                )
                diff_viewer.pack(fill=tk.BOTH, expand=True, pady=10)
                
                # Add separator
                separator = tk.Label(
                    self.tool_results_frame,
                    text=f"✅ {result['changes_count']} change(s) made to {os.path.basename(result['filepath'])}",
                    bg="#1f4d2b",
                    fg="#6bff6b",
                    font=("Arial", 10, "bold"),
                    pady=5
                )
                separator.pack(fill=tk.X, pady=5)
            
            elif tool_name == "create_file" and result.get("success"):
                # Show created file notification
                header = tk.Label(
                    self.tool_results_frame,
                    text=f"✨ Created: {result['filepath']} ({result['size']} chars)",
                    bg="#2b3d4d",
                    fg="#6bb6ff",
                    font=("Arial", 10, "bold"),
                    pady=10
                )
                header.pack(fill=tk.X, pady=(10, 0))
                
                # Show content
                content_text = scrolledtext.ScrolledText(
                    self.tool_results_frame,
                    height=10,
                    font=("Consolas", 10),
                    background="#2d2d2d",
                    foreground="#d4d4d4"
                )
                content_text.pack(fill=tk.X, pady=(0, 10))
                content_text.insert(tk.END, result["content"])
                content_text.config(state=tk.DISABLED)
            
            elif tool_name == "read_file" and result.get("success"):
                # Show read file notification
                read_label = tk.Label(
                    self.tool_results_frame,
                    text=f"📖 Read: {result['filepath']} ({result['size']} chars)",
                    bg="#3d2d4d",
                    fg="#d6a6ff",
                    font=("Arial", 9),
                    pady=3
                )
                read_label.pack(fill=tk.X, pady=2)

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