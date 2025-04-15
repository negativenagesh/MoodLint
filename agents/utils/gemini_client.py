import os
import google.generativeai as genai
from typing import Optional, Dict, Any, List

class GeminiClient:
    """Client for interacting with Google's Gemini models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client with API key."""
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
    
    def generate_response(self, 
                         prompt: str, 
                         temperature: float = 0.7, 
                         max_tokens: int = 1024, 
                         system_instruction: Optional[str] = None) -> str:
        """Generate a response from Gemini for a given prompt."""
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
        
        # Create conversation history with system instruction if provided
        chat = None
        if system_instruction:
            chat = self.model.start_chat(system_instruction=system_instruction)
            response = chat.send_message(prompt, generation_config=generation_config, safety_settings=safety_settings)
        else:
            response = self.model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
        
        # Handle the response
        if hasattr(response, "text"):
            return response.text
        elif hasattr(response, "parts"):
            return response.parts[0].text
        else:
            return response.candidates[0].content.parts[0].text