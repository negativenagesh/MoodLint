import os
import traceback
import google.generativeai as genai
from typing import Optional, Dict, Any

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

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_instruction: Optional[str] = None
    ) -> str:
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
                return (
                    "I've analyzed your file, but couldn't extract a detailed response.\n"
                    "Consider checking the Gemini API output format or increasing verbosity."
                )

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Gemini API error: {str(e)}\n{error_trace}")
            # Return helpful message instead of raising exception
            return (
                "I encountered an error while analyzing your code, but I can still help.\n\n"
                f"Error details: {str(e)}"
            )

    async def analyze_code(
        self,
        code: str,
        filename: str,
        mood: str,
        query: str = "",
        model: str = "gemini-1.5-flash"
    ) -> Dict[str, Any]:
        """
        Analyze code using the Gemini API.

        Args:
            code: The source code to analyze
            filename: The name of the file containing the code
            mood: The mood to use for analysis
            query: Optional user query about the code
            model: The Gemini model to use

        Returns:
            Dict containing the analysis and metadata
        """
        try:
            print(f"Analyzing code ({filename}) with mood: {mood}")
            if query:
                print(f"User query: {query}")

            # Construct the prompt
            file_extension = filename.split('.')[-1] if '.' in filename else 'txt'

            # Base prompt structure
            prompt = (
                f"As a software developer in a {mood} mood, analyze this {file_extension} code:\n\n"
                f"```{file_extension}\n{code}\n```\n"
            )
            # Add user query if provided
            if query:
                prompt += f"\nI specifically want to know: {query}\n"
            else:
                prompt += (
                    "\nProvide a thorough analysis of the code focusing on:\n"
                    "1. Overall structure and functionality\n"
                    "2. Potential bugs or issues\n"
                    "3. Improvements that could be made\n"
                    "4. Best practices that are followed or missed\n"
                )

            # Add mood-specific instructions
            if mood.lower() == "angry":
                prompt += "\nGive your analysis in a critical, direct tone, focusing on things that could irritate a developer."
            elif mood.lower() == "happy":
                prompt += "\nGive your analysis in an optimistic tone, highlighting the positive aspects while still noting improvements."
            elif mood.lower() == "sad":
                prompt += "\nGive your analysis in a thoughtful, slightly melancholy tone."
            elif mood.lower() == "frustrated":
                prompt += "\nGive your analysis focusing on pain points that might frustrate a developer."
            elif mood.lower() == "exhausted":
                prompt += "\nGive your analysis focusing on complexity and cognitive load, noting things that would tire a developer."

            # Generate the response
            response_text = self.generate_response(prompt, temperature=0.7, max_tokens=4096)
            print(f"Generated response, length: {len(response_text)}")

            # Return successful result
            return {
                "success": True,
                "mood": mood,
                "query": query,
                "response": response_text
            }

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error analyzing code: {str(e)}\n{error_details}")

            # Create a helpful default response
            default_response = (
                f"As a {mood} developer looking at your {filename} file:\n\n"
                "I notice this appears to be a file that implements some functionality.\n\n"
                "Without being able to fully analyze it due to technical issues, I'd recommend:\n"
                "1. Ensuring your code is well-documented\n"
                f"2. Following best practices for {file_extension} files\n"
                "3. Adding appropriate error handling\n"
                "4. Considering performance optimizations where relevant\n\n"
                f"Error details: {str(e)}"
            )

            return {
                "success": False,
                "error": f"Code analysis error: {str(e)}",
                "details": error_details,
                "mood": mood,
                "response": default_response
            }