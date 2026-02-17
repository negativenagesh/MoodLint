"""Diff viewer widget for displaying side-by-side file comparisons."""
import tkinter as tk
from tkinter import Frame, Label, Text, Scrollbar
import difflib
import os


class DiffViewer(Frame):
    """2-column diff viewer for file edits with syntax highlighting."""
    
    def __init__(self, parent, filepath, original, modified):
        super().__init__(parent)
        self.filepath = filepath
        self.original = original
        self.modified = modified
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create the diff viewer UI."""
        # Title
        title = Label(
            self, 
            text=f"📝 File Edit: {os.path.basename(self.filepath)}", 
            font=("Arial", 12, "bold"),
            bg="#2d2d2d",
            fg="#ffffff"
        )
        title.pack(pady=5, fill=tk.X)
        
        # Container for both columns
        container = Frame(self, bg="#1e1e1e")
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left column: Original  
        left_frame = Frame(container, bg="#1e1e1e")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        left_header = Label(
            left_frame, 
            text="❌ Original", 
            bg="#3c3c3c", 
            fg="#ff6b6b",
            font=("Arial", 10, "bold"),
            pady=5
        )
        left_header.pack(fill=tk.X)
        
        left_scroll = Scrollbar(left_frame)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.left_text = Text(
            left_frame, 
            bg="#1e1e1e", 
            fg="#d4d4d4",
            font=("Courier", 10),
            yscrollcommand=left_scroll.set,
            wrap=tk.NONE,
            width=50
        )
        self.left_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll.config(command=self.on_scroll)
        
        # Right column: Modified
        right_frame = Frame(container, bg="#1e1e1e")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)
        
        right_header = Label(
            right_frame, 
            text="✅ Modified", 
            bg="#3c3c3c", 
            fg="#6bff6b",
            font=("Arial", 10, "bold"),
            pady=5
        )
        right_header.pack(fill=tk.X)
        
        right_scroll = Scrollbar(right_frame)
        right_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.right_text = Text(
            right_frame, 
            bg="#1e1e1e", 
            fg="#d4d4d4",
            font=("Courier", 10),
            yscrollcommand=right_scroll.set,
            wrap=tk.NONE,
            width=50
        )
        self.right_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_scroll.config(command=self.on_scroll)
        
        # Populate diff
        self.show_diff()
        
        # Sync scrolling
        self.left_text.bind('<MouseWheel>', self.sync_scroll)
        self.right_text.bind('<MouseWheel>', self.sync_scroll)
        self.left_text.bind('<Button-4>', self.sync_scroll)  # Linux scroll up
        self.left_text.bind('<Button-5>', self.sync_scroll)  # Linux scroll down
        self.right_text.bind('<Button-4>', self.sync_scroll)
        self.right_text.bind('<Button-5>', self.sync_scroll)
    
    def show_diff(self):
        """Display side-by-side diff with highlighting."""
        orig_lines = self.original.splitlines()
        mod_lines = self.modified.splitlines()
        
        # Use difflib to find changes
        matcher = difflib.SequenceMatcher(None, orig_lines, mod_lines)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # No changes
                for line in orig_lines[i1:i2]:
                    self.left_text.insert(tk.END, line + '\n')
                    self.right_text.insert(tk.END, line + '\n')
            
            elif tag == 'delete':
                # Lines removed
                for line in orig_lines[i1:i2]:
                    self.left_text.insert(tk.END, line + '\n', 'deleted')
                    self.right_text.insert(tk.END, '\n')  # Empty line on right
            
            elif tag == 'insert':
                # Lines added
                for line in mod_lines[j1:j2]:
                    self.left_text.insert(tk.END, '\n')  # Empty line on left
                    self.right_text.insert(tk.END, line + '\n', 'added')
            
            elif tag == 'replace':
                # Lines changed
                max_lines = max(i2 - i1, j2 - j1)
                for i in range(max_lines):
                    left_line = orig_lines[i1 + i] if (i1 + i) < i2 else ""
                    right_line = mod_lines[j1 + i] if (j1 + i) < j2 else ""
                    
                    if left_line:
                        self.left_text.insert(tk.END, left_line + '\n', 'deleted')
                    else:
                        self.left_text.insert(tk.END, '\n')
                    
                    if right_line:
                        self.right_text.insert(tk.END, right_line + '\n', 'added')
                    else:
                        self.right_text.insert(tk.END, '\n')
        
        # Configure color tags (VS Code-like colors)
        self.left_text.tag_config('deleted', background='#4d1f1f', foreground='#ff6b6b')
        self.right_text.tag_config('added', background='#1f4d2b', foreground='#6bff6b')
        
        # Make read-only
        self.left_text.config(state=tk.DISABLED)
        self.right_text.config(state=tk.DISABLED)
    
    def on_scroll(self, *args):
        """Handle scrollbar movement."""
        self.left_text.yview(*args)
        self.right_text.yview(*args)
    
    def sync_scroll(self, event):
        """Sync scrolling between both text widgets."""
        # Get current position
        if event.widget == self.left_text:
            pos = self.left_text.yview()
        else:
            pos = self.right_text.yview()
        
        # Sync both
        self.left_text.yview_moveto(pos[0])
        self.right_text.yview_moveto(pos[0])
        
        return "break"  # Prevent default scroll behavior
