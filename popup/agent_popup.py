import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import json
import sys
import threading
import time
import os
import asyncio
import traceback

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the debug_code function from the agent workflow
from agents.workflow import debug_code

# Ensure virtual environment is in path
venv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "lib", "python3.8", "site-packages")
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)
    print(f"Added virtual environment path: {venv_path}")

class AgentDebugApp:
    def __init__(self, root, mood, query="", initial_file=None):
        self.root = root
        self.mood = mood
        self.query = query
        self.is_complete = False
        self.code = ""
        self.filename = ""
        self.selected_files = []
        self.initial_file = initial_file  # Store the initial file path
        
        # Set up the UI
        self.setup_ui()
        
        # If initial file was provided, load it
        if self.initial_file and os.path.exists(self.initial_file):
            self.selected_files.append(self.initial_file)
            self.file_list.insert(tk.END, os.path.basename(self.initial_file))
            # Select the file in the listbox
            self.file_list.selection_set(0)
    
    def setup_ui(self):
        self.root.title(f"MoodLint Agent - {self.mood.capitalize()} Mode")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)
        
        # Make window appear on top
        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.attributes('-topmost', False)
        
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 800) // 2
        y = (screen_height - 700) // 2
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
        
        # File selection section
        file_select_frame = ttk.LabelFrame(main_frame, text="Select Files")
        file_select_frame.pack(fill=tk.X, pady=(0, 10))
        
        file_buttons_frame = ttk.Frame(file_select_frame)
        file_buttons_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(file_buttons_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_buttons_frame, text="Clear Selection", command=self.clear_files).pack(side=tk.LEFT, padx=5)
        
        # File list
        self.file_list = tk.Listbox(file_select_frame, height=3)
        self.file_list.pack(fill=tk.X, padx=5, pady=5)
        
        # Query section
        query_frame = ttk.LabelFrame(main_frame, text="Query")
        query_frame.pack(fill=tk.X, pady=(0, 10))
        
        query_label = ttk.Label(query_frame, text="What would you like to know about this code?")
        query_label.pack(anchor=tk.W, padx=5, pady=(5, 0))
        
        self.query_input = scrolledtext.ScrolledText(query_frame, wrap=tk.WORD, height=3)
        self.query_input.pack(fill=tk.X, padx=5, pady=5)
        if self.query:
            self.query_input.insert(tk.END, self.query)
        
        # Analyze button
        analyze_frame = ttk.Frame(main_frame)
        analyze_frame.pack(fill=tk.X)
        
        self.analyze_btn = ttk.Button(analyze_frame, text="Analyze Selected Files", command=self.run_analysis)
        self.analyze_btn.pack(anchor=tk.CENTER, pady=5)
        
        # Create progress frame
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.StringVar(value="Ready to analyze")
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
        
        # Set up close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

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
    
    def add_files(self):
        """Add files to the list"""
        filepaths = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=[
                ("Python Files", "*.py"),
                ("JavaScript Files", "*.js"),
                ("TypeScript Files", "*.ts"),
                ("HTML Files", "*.html"),
                ("CSS Files", "*.css"),
                ("All Files", "*.*")
            ]
        )
        
        if not filepaths:
            return
            
        for filepath in filepaths:
            if filepath not in self.selected_files:
                self.selected_files.append(filepath)
                self.file_list.insert(tk.END, os.path.basename(filepath))
    
    def clear_files(self):
        """Clear all files from the list"""
        self.selected_files = []
        self.file_list.delete(0, tk.END)
    
    def update_progress(self):
        """Update the progress animation"""
        if not self.is_complete:
            dots = self.progress_dots.get()
            
            if len(dots) >= 3:
                dots = ""
            else:
                dots += "."
                
            self.progress_dots.set(dots)
            self.root.after(500, self.update_progress)
    
    def run_analysis(self):
        """Run analysis on selected files"""
        # Check if files are selected
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select at least one file to analyze.")
            return
        
        # Get the selected index from the listbox
        selected_idx = self.file_list.curselection()
        if not selected_idx:
            messagebox.showwarning("No Selection", "Please select a file from the list.")
            return
        
        selected_idx = selected_idx[0]
        file_path = self.selected_files[selected_idx]
        self.filename = os.path.basename(file_path)
        
        # Read the selected file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.code = f.read()
            print(f"Successfully read file: {file_path}, length: {len(self.code)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {str(e)}")
            return
        
        # Get the query
        self.query = self.query_input.get("1.0", tk.END).strip()
        print(f"Query: '{self.query}'")
        
        # Check API key
        if not os.environ.get("GOOGLE_API_KEY"):
            api_key_error = "GOOGLE_API_KEY is not set in environment variables"
            print(api_key_error)
            messagebox.showerror("API Key Error", 
                "Google API Key is not set. Please add it to your .env file or environment variables.")
            return
        
        # Start analysis
        self.is_complete = False
        self.progress_var.set("Analyzing code...")
        self.progress_dots.set("")
        self.analyze_btn.config(state=tk.DISABLED)
        self.update_progress()
        
        # Log analysis start
        print(f"Starting analysis of {self.filename} with mood: {self.mood}")
        
        # Start agent processing in a separate thread
        self.agent_thread = threading.Thread(target=self.run_agent_workflow)
        self.agent_thread.daemon = True
        self.agent_thread.start()
    
    def run_agent_workflow(self):
        """Run the agent workflow in a background thread"""
        try:
            print(f"Starting agent workflow thread")
            
            # Create event loop for async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Ensure we're using the correct mood parameter
            print(f"Calling debug_code with: filename={self.filename}, mood={self.mood}, code length={len(self.code)}")
            
            # Run debug_code function
            result = loop.run_until_complete(debug_code(
                code=self.code,
                filename=self.filename,
                mood=self.mood,
                query=self.query
            ))
            
            print(f"Debug code completed, result: {result.keys() if isinstance(result, dict) else 'not a dict'}")
            
            # Check for raw Gemini response first
            if isinstance(result, dict):
                print(f"Result type: dict with keys {result.keys()}")
                
                # Check for raw Gemini response first - this is the most direct source
                if "raw_response" in result and result["raw_response"]:
                    print(f"Found raw_response in result, using as primary response")
                    result["response"] = result["raw_response"]
                    self.root.after(0, lambda: self.update_results(result))
                    return
                
                # Direct extraction from Gemini API result
                if "response" in result and result["response"]:
                    response_length = len(result["response"]) if isinstance(result["response"], str) else "non-string"
                    print(f"Found response directly in result, length: {response_length}")
                    
                    # Only check for fallback if response is too short or looks generic
                    fallback_indicators = ["API connection", "wasn't able to generate", "check that your API key"]
                    is_fallback = (isinstance(result["response"], str) and 
                                   any(indicator in result["response"] for indicator in fallback_indicators))
                    
                    if not is_fallback:
                        # Valid non-fallback response found
                        self.root.after(0, lambda: self.update_results(result))
                        return
                    else:
                        print("WARNING: Detected fallback response, attempting to find actual response")
                        
                        # If we have a nested result structure, look deeper
                        if "result" in result and isinstance(result["result"], dict):
                            print(f"Looking in nested result: {result['result'].keys()}")
                            
                            # Try to directly access Gemini response if it exists
                            if "gemini_response" in result["result"]:
                                print("Found gemini_response in nested result")
                                result["response"] = result["result"]["gemini_response"]
                                self.root.after(0, lambda: self.update_results(result))
                                return
                            elif "raw_response" in result["result"]:
                                print("Found raw_response in nested result")
                                result["response"] = result["result"]["raw_response"]
                                self.root.after(0, lambda: self.update_results(result))
                                return
                
                # Check for Gemini client-specific nested response format
                if "output" in result and isinstance(result["output"], dict):
                    if "response" in result["output"] and result["output"]["response"]:
                        print(f"Found response in output property")
                        result["response"] = result["output"]["response"]
                        self.root.after(0, lambda: self.update_results(result))
                        return
                
                # Look for nested response in result
                if "result" in result and isinstance(result["result"], dict):
                    nested = result["result"]
                    print(f"Nested result keys: {nested.keys()}")
                    
                    if "response" in nested and nested["response"]:
                        print(f"Found response in nested result")
                        result["response"] = nested["response"]
                        self.root.after(0, lambda: self.update_results(result))
                        return
                
                # Check if we have a gemini_client_output property
                if "gemini_client_output" in result:
                    print("Found gemini_client_output in result")
                    if isinstance(result["gemini_client_output"], str):
                        result["response"] = result["gemini_client_output"]
                        self.root.after(0, lambda: self.update_results(result))
                        return
                    elif isinstance(result["gemini_client_output"], dict) and "response" in result["gemini_client_output"]:
                        result["response"] = result["gemini_client_output"]["response"]
                        self.root.after(0, lambda: self.update_results(result))
                        return
            
            # Create a more useful fallback response that includes the query
            if isinstance(result, dict):
                if "success" not in result:
                    result["success"] = True
                if "mood" not in result:
                    result["mood"] = self.mood
                
                # Create a response if none exists, making it more relevant to the query
                if "response" not in result or not result["response"]:
                    query_context = f" about '{self.query}'" if self.query else ""
                    result["response"] = (
                        f"I analyzed your {self.filename} file from a {self.mood} perspective{query_context}.\n\n"
                        "While I couldn't generate a detailed analysis, here are some general observations:\n\n"
                        f"1. The file appears to be a {self.filename.split('.')[-1]} file that likely contains application logic\n"
                        f"2. For files like this, common issues include error handling, performance optimization, and maintainability\n"
                        f"3. Consider adding more comprehensive documentation and unit tests\n\n"
                        f"To get a better analysis, please try again with a more specific query or check your API configuration."
                    )
            elif isinstance(result, str):
                # Convert string response to proper structure
                result = {
                    "success": True,
                    "mood": self.mood,
                    "response": result,
                    "query": self.query
                }
            else:
                # Create fallback result with better information
                result = {
                    "success": True,
                    "mood": self.mood,
                    "response": (
                        f"Analysis of {self.filename} from a {self.mood} perspective was completed, but the response format was unexpected.\n\n"
                        "Please ensure your API key is correctly configured and has sufficient permissions."
                    ),
                    "result": {"mood": self.mood}
                }
            
            # Update UI with results
            self.root.after(0, lambda: self.update_results(result))
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error during analysis: {str(e)}\n{error_traceback}")
            error_message = f"Error during analysis: {str(e)}\n\nDetails: {error_traceback}"
            self.root.after(0, lambda: self.update_error(error_message))
    
    def update_status(self, message):
        """Update the status label"""
        self.root.after(0, lambda: self.progress_var.set(message))
    
    def update_results(self, result):
        """Update the results panel with agent output"""
        self.is_complete = True
        self.progress_var.set("Analysis complete")
        self.progress_dots.set("")
        self.analyze_btn.config(state=tk.NORMAL)
        
        # Enable text widget for editing
        self.result_text.config(state=tk.NORMAL)
        
        # Clear any existing text
        self.result_text.delete("1.0", tk.END)
        
        print(f"Result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
        
        # Extract response text from result with priority to raw_response
        response_text = None
        
        if isinstance(result, dict):
            # First priority: Direct raw_response that looks substantial
            if "raw_response" in result and result["raw_response"] and len(result["raw_response"]) > 100:
                response_text = result["raw_response"]
                print(f"Using direct raw_response of length: {len(response_text)}")
            
            # Second priority: Check nested raw_response
            elif "result" in result and isinstance(result["result"], dict) and "raw_response" in result["result"] and len(result["result"]["raw_response"]) > 100:
                response_text = result["result"]["raw_response"]
                print(f"Using nested raw_response of length: {len(response_text)}")
            
            # Third priority: Check output.raw_response (from workflow DebugResponse)
            elif "output" in result and isinstance(result["output"], dict) and "raw_response" in result["output"] and len(result["output"]["raw_response"]) > 100:
                response_text = result["output"]["raw_response"]
                print(f"Using output.raw_response of length: {len(response_text)}")
            
            # Fourth priority: Standard response if it's substantial and not a fallback
            elif "response" in result and result["response"] and len(result["response"]) > 100:
                # Check if it looks like a fallback
                fallback_indicators = ["API connection", "wasn't able to generate", "check that your API key"]
                is_fallback = any(indicator in result["response"] for indicator in fallback_indicators)
                
                if not is_fallback:
                    response_text = result["response"]
                    print(f"Using standard response of length: {len(response_text)}")
            
            # Fifth priority: Check nested response if it's substantial
            elif "result" in result and isinstance(result["result"], dict) and "response" in result["result"] and len(result["result"]["response"]) > 100:
                fallback_indicators = ["API connection", "wasn't able to generate", "check that your API key"]
                is_fallback = any(indicator in result["result"]["response"] for indicator in fallback_indicators)
                
                if not is_fallback:
                    response_text = result["result"]["response"]
                    print(f"Using nested response of length: {len(response_text)}")
            
            # Sixth priority: Check output response (from workflow DebugResponse)
            elif "output" in result and isinstance(result["output"], dict) and "response" in result["output"] and len(result["output"]["response"]) > 100:
                fallback_indicators = ["API connection", "wasn't able to generate", "check that your API key"]
                is_fallback = any(indicator in result["output"]["response"] for indicator in fallback_indicators)
                
                if not is_fallback:
                    response_text = result["output"]["response"]
                    print(f"Using output.response of length: {len(response_text)}")
        
        # If we found a valid response, display it
        if response_text and len(response_text) > 100:
            self.result_text.insert(tk.END, response_text)
        else:
            # Create a better fallback message if no valid response was found
            standard_response = None
            if isinstance(result, dict) and "response" in result and result["response"]:
                standard_response = result["response"]
            
            # If we at least have a standard response that's not too short, use it
            if standard_response and len(standard_response) > 50:
                self.result_text.insert(tk.END, standard_response)
            else:
                # Final fallback if we have nothing useful
                query_context = f" about '{self.query}'" if self.query else ""
                fallback_message = (
                    f"I've analyzed your {self.filename} file from a {self.mood} perspective{query_context}.\n\n"
                    "I received a response from the Gemini API, but couldn't properly process it for display.\n\n"
                    "Please check the console output for more details about the response that was received."
                )
                self.result_text.insert(tk.END, fallback_message)
        
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
        self.analyze_btn.config(state=tk.NORMAL)
        
        # Enable text widget for editing
        self.result_text.config(state=tk.NORMAL)
        
        # Clear any existing text
        self.result_text.delete("1.0", tk.END)
        
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
    load_dotenv()
    
    # Check if API key is available and print it (obscured)
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        print(f"GOOGLE_API_KEY found, starts with {api_key[:4]}...")
    else:
        print(json.dumps({
            "status": "error", 
            "message": "GOOGLE_API_KEY environment variable is not set. Please set it or add it to .env file."
        }), flush=True)
        sys.exit(1)

    # Parse command line arguments
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error", 
            "message": "Usage: agent_popup.py <mood> [file_path] [query]"
        }), flush=True)
        sys.exit(1)
    
    # First argument is always the mood
    mood = sys.argv[1]
    
    # Optional file path and query
    file_path = None
    query = ""
    
    if len(sys.argv) > 2:
        # Second argument could be a file path
        potential_file = sys.argv[2]
        if os.path.exists(potential_file):
            file_path = potential_file
            # If there's a third argument, it's the query
            if len(sys.argv) > 3:
                query = sys.argv[3]
        else:
            # If second argument isn't a file, treat it as a query
            query = potential_file
    
    # Initialize the UI
    print(json.dumps({"status": "starting"}), flush=True)
    
    root = tk.Tk()
    app = AgentDebugApp(root, mood, query, file_path)
    
    print(json.dumps({"status": "ready"}), flush=True)
    
    # Start the UI event loop
    try:
        root.mainloop()
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(json.dumps({"status": "error", "message": f"UI error: {str(e)}\n{error_traceback}"}), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()