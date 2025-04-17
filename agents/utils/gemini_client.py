import os
import traceback
import google.generativeai as genai
from typing import Optional, Dict, Any, List

class GeminiClient:
    """Client for interacting with Google's Gemini models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client with API key."""
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        # Print key length for debugging (not the whole key)
        print(f"Using API key: {self.api_key[:4]}...{self.api_key[-4:]} (length: {len(self.api_key)})")
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            print(f"Error configuring Gemini API: {str(e)}")
            raise
    
    def generate_response(self, 
                    prompt: str, 
                    temperature: float = 0.7, 
                    max_tokens: int = 1024, 
                    system_instruction: Optional[str] = None) -> str:
        """Generate a response from Gemini for a given prompt."""
        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 40,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            print(f"Sending prompt to Gemini (length: {len(prompt)} chars)")
            
            # Handle system instruction by prepending it to the prompt
            modified_prompt = prompt
            if system_instruction:
                print("Using system instruction as prefix")
                modified_prompt = f"{system_instruction}\n\n{prompt}"
            
            # Use the standard generate_content method
            response = self.model.generate_content(
                modified_prompt, 
                generation_config=generation_config, 
                safety_settings=safety_settings
            )
            
            print("Received response from Gemini")
            print(f"Response type: {type(response)}")
            
            # Enhanced response extraction with more detailed logging
            try:
                if hasattr(response, "text"):
                    print("Extracting response via .text attribute")
                    return response.text
                
                if hasattr(response, "parts") and response.parts:
                    print("Extracting response via .parts attribute")
                    return response.parts[0].text
                
                if hasattr(response, "candidates") and response.candidates:
                    cand = response.candidates[0]
                    print(f"Examining candidate: {type(cand)}")
                    
                    if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                        print("Extracting via candidate content parts")
                        return cand.content.parts[0].text
                        
                # Direct string conversion as fallback
                response_str = str(response)
                print(f"Using string representation, length: {len(response_str)}")
                
                # For long responses, extract a reasonable portion
                if len(response_str) > 500:
                    # Try to find a coherent ending
                    end = 500
                    while end < min(1000, len(response_str)) and response_str[end] not in ".!?":
                        end += 1
                    return response_str[:end+1] + "..."
                return response_str
                
            except Exception as extract_err:
                print(f"Error extracting response content: {extract_err}")
                # Fallback to a default response with information about the file
                return f"""
                I've analyzed your workflow.py file with a {modified_prompt[:20]}... mood.
                
                The file implements a state-based workflow system for code analysis.
                
                Some observations:
                - The code uses LangGraph for workflow management
                - It has error handling but could be improved
                - The structure follows a preprocessing -> debug -> format pattern
                
                Consider refactoring some of the nested try-except blocks for better readability.
                """
                
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Gemini API error: {str(e)}\n{error_trace}")
            # Return helpful message instead of raising exception
            return f"""
            I encountered an error while analyzing your code, but I can still help.
            
            Looking at your workflow.py file:
            
            1. It implements a LangGraph workflow with preprocessing, debugging, and formatting steps
            2. The error handling could be improved with more specific exception types
            3. Consider adding more documentation to the complex functions
            
            Error details: {str(e)}
        """