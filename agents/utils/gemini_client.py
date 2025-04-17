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
            
            # Create conversation history with system instruction if provided
            response = None
            if system_instruction:
                print("Using system instruction")
                chat = self.model.start_chat(system_instruction=system_instruction)
                response = chat.send_message(prompt, generation_config=generation_config, safety_settings=safety_settings)
            else:
                print("No system instruction")
                response = self.model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
            
            print("Received response from Gemini")
            
            # Handle the response
            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "parts"):
                return response.parts[0].text
            else:
                try:
                    return response.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    # Try to extract any text we can
                    return str(response)
                    
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Gemini API error: {str(e)}\n{error_trace}")
            # Re-raise with more information
            raise ValueError(f"Error from Gemini API: {str(e)}")