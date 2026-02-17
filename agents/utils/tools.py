"""Tool registry for file operations with OpenAI function calling."""
import os
import json
from typing import Dict, List, Any


class ToolRegistry:
    """Registry for all available tools."""
    
    @staticmethod
    def get_tool_definitions() -> List[Dict]:
        """Returns OpenAI function calling schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the complete contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Absolute path to the file to read"
                            }
                        },
                        "required": ["filepath"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit a file by replacing specific content. Returns original and new content for diff view.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Absolute path to file"
                            },
                            "search": {
                                "type": "string",
                                "description": "Exact text to find and replace"
                            },
                            "replace": {
                                "type": "string",
                                "description": "Replacement text"
                            }
                        },
                        "required": ["filepath", "search", "replace"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_file",
                    "description": "Create a new file with specified content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path for the new file"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["filepath", "content"]
                    }
                }
            }
        ]
    
    @staticmethod
    def execute_tool(tool_name: str, arguments: Dict) -> Dict:
        """Execute a tool and return results."""
        if tool_name == "read_file":
            return read_file_tool(arguments["filepath"])
        elif tool_name == "edit_file":
            return edit_file_tool(
                arguments["filepath"],
                arguments["search"],
                arguments["replace"]
            )
        elif tool_name == "create_file":
            return create_file_tool(
                arguments["filepath"],
                arguments["content"]
            )
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}


def read_file_tool(filepath: str) -> Dict:
    """Read file content."""
    try:
        if not os.path.exists(filepath):
            return {"success": False, "error": f"File not found: {filepath}"}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"[Tool] Read file: {filepath} ({len(content)} chars)")
        
        return {
            "success": True,
            "content": content,
            "filepath": filepath,
            "size": len(content)
        }
    except Exception as e:
        print(f"[Tool] Error reading {filepath}: {str(e)}")
        return {"success": False, "error": str(e), "filepath": filepath}


def edit_file_tool(filepath: str, search: str, replace: str) -> Dict:
    """Edit file by replacing text. Returns both original and new content for diff."""
    try:
        if not os.path.exists(filepath):
            return {"success": False, "error": f"File not found: {filepath}"}
        
        # Read original content
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Check if search text exists
        if search not in original_content:
            return {
                "success": False,
                "error": f"Search text not found in {filepath}",
                "filepath": filepath
            }
        
        # Count occurrences
        occurrences = original_content.count(search)
        
        # Create new content
        new_content = original_content.replace(search, replace)
        
        # Write new content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"[Tool] Edited file: {filepath} ({occurrences} replacements)")
        
        return {
            "success": True,
            "filepath": filepath,
            "original_content": original_content,
            "new_content": new_content,
            "message": f"Successfully edited {os.path.basename(filepath)}",
            "changes_count": occurrences
        }
    except Exception as e:
        print(f"[Tool] Error editing {filepath}: {str(e)}")
        return {"success": False, "error": str(e), "filepath": filepath}


def create_file_tool(filepath: str, content: str) -> Dict:
    """Create a new file."""
    try:
        # Create directory if needed
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(filepath):
            return {
                "success": False,
                "error": f"File already exists: {filepath}",
                "filepath": filepath
            }
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"[Tool] Created file: {filepath} ({len(content)} chars)")
        
        return {
            "success": True,
            "filepath": filepath,
            "content": content,
            "message": f"Successfully created {os.path.basename(filepath)}",
            "size": len(content)
        }
    except Exception as e:
        print(f"[Tool] Error creating {filepath}: {str(e)}")
        return {"success": False, "error": str(e), "filepath": filepath}
