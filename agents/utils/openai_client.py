import os
import json
import traceback
from typing import Optional, Dict, Any, List
from openai import OpenAI

class OpenAIClient:
    """Client for interacting with OpenAI's GPT models."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client with API key."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")

        # Print key length for debugging (not the whole key)
        print(f"Using API key: {self.api_key[:7]}...{self.api_key[-4:]} (length: {len(self.api_key)})")

        try:
            self.client = OpenAI(api_key=self.api_key)
            self.model = "gpt-4o-mini"
            print(f"Initialized OpenAI client with model: {self.model}")
        except Exception as e:
            print(f"Error configuring OpenAI API: {str(e)}")
            raise

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate a response from OpenAI for a given prompt."""
        try:
            print(f"Sending prompt to OpenAI (length: {len(prompt)} chars)")

            # Construct messages array
            messages = []
            
            # Add system instruction if provided
            if system_instruction:
                print("Using system instruction")
                messages.append({
                    "role": "system",
                    "content": system_instruction
                })
            
            # Add user prompt
            messages.append({
                "role": "user",
                "content": prompt
            })

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            print("Received response from OpenAI")
            
            # Extract response text
            if response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                print(f"Extracted response, length: {len(response_text)}")
                return response_text
            else:
                print("Warning: No choices in response")
                return "No response generated from OpenAI."

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"OpenAI API error: {str(e)}\n{error_trace}")
            # Return helpful message instead of raising exception
            return (
                "I encountered an error while analyzing your code, but I can still help.\n\n"
                f"Error details: {str(e)}"
            )

    def generate_response_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        max_iterations: int = 10,
        temperature: float = 0.7,
        system_instruction: Optional[str] = None,
        stream_callback=None
    ) -> Dict[str, Any]:
        """Generate response with tool calling support and streaming."""
        from .tools import ToolRegistry
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        messages.append({"role": "user", "content": prompt})
        
        tool_calls_made = []
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            print(f"[OpenAI] Tool iteration {iterations}/{max_iterations}")
            
            try:
                # API Call with streaming enabled if callback provided
                stream = True if stream_callback else False
                
                response_stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=4096,
                    stream=stream
                )
                
                # If not streaming, use original logic (simplifies existing calls)
                if not stream:
                    message = response_stream.choices[0].message
                    # ... [reuse existing non-streaming logic or just accumulate below] ...
                    # Actually, better to unify logic. Let's assume we handle object vs stream.
                    # But response_stream IS the object if stream=False.
                    # For simplicity in this edit, I will implement the streaming logic specifically.
                    pass # Handled by block below (except response_stream isn't a stream)
                
                # Accumulators
                full_content = ""
                tool_calls_accumulated = []
                current_tool_call = None
                
                if stream:
                    print("[OpenAI] Streaming response...")
                    for chunk in response_stream:
                        delta = chunk.choices[0].delta
                        
                        # Handle content
                        if delta.content:
                            content_chunk = delta.content
                            full_content += content_chunk
                            if stream_callback:
                                stream_callback(content_chunk)
                        
                        # Handle tool calls
                        if delta.tool_calls:
                            for tc_chunk in delta.tool_calls:
                                # If starting a new tool call
                                if tc_chunk.index >= len(tool_calls_accumulated):
                                    tool_calls_accumulated.append({
                                        "id": "",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                
                                tc_data = tool_calls_accumulated[tc_chunk.index]
                                
                                if tc_chunk.id:
                                    tc_data["id"] += tc_chunk.id
                                
                                if tc_chunk.function:
                                    if tc_chunk.function.name:
                                        tc_data["function"]["name"] += tc_chunk.function.name
                                    if tc_chunk.function.arguments:
                                        tc_data["function"]["arguments"] += tc_chunk.function.arguments
                else:
                    # Non-streaming fallback
                    response = response_stream # It's actually the response object
                    message = response.choices[0].message
                    full_content = message.content or ""
                    if message.tool_calls:
                        for tc in message.tool_calls:
                            tool_calls_accumulated.append({
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            })
                
                # Process results
                print(f"[OpenAI] Iteration complete. Content len: {len(full_content)}, Tool calls: {len(tool_calls_accumulated)}")
                
                # If we have tool calls
                if tool_calls_accumulated:
                    # Add assistant message with tool calls
                    assistant_msg = {
                        "role": "assistant",
                        "content": full_content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"]
                                }
                            }
                            for tc in tool_calls_accumulated
                        ]
                    }
                    messages.append(assistant_msg)
                    
                    # Execute each tool
                    for tc in tool_calls_accumulated:
                        tool_name = tc["function"]["name"]
                        try:
                            arguments = json.loads(tc["function"]["arguments"])
                            
                            print(f"[Tool] Executing {tool_name} with args: {arguments}")
                            
                            # Execute tool
                            result = ToolRegistry.execute_tool(tool_name, arguments)
                            
                            # Store tool call info
                            tool_calls_made.append({
                                "tool": tool_name,
                                "arguments": arguments,
                                "result": result
                            })
                            
                            # Add tool result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": json.dumps(result)
                            })
                        except json.JSONDecodeError:
                            print(f"[Tool] Error decoding arguments for {tool_name}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": json.dumps({"error": "Invalid JSON arguments"})
                            })
                else:
                    # No tool calls, this is the final response
                    return {
                        "success": True,
                        "response": full_content,
                        "tool_calls": tool_calls_made,
                        "iterations": iterations
                    }
            
            except Exception as e:
                print(f"[OpenAI] Error: {str(e)}")
                traceback.print_exc()
                return {
                    "success": False,
                    "error": str(e),
                    "tool_calls": tool_calls_made,
                    "iterations": iterations
                }
        
        # Max iterations reached
        return {
            "success": True,
            "response": "Maximum tool iterations reached",
            "tool_calls": tool_calls_made,
            "iterations": max_iterations
        }

    async def analyze_code(
        self,
        code: str,
        filename: str,
        mood: str,
        query: str = "",
        model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """
        Analyze code using the OpenAI API.

        Args:
            code: The source code to analyze
            filename: The name of the file containing the code
            mood: The mood to use for analysis
            query: Optional user query about the code
            model: The OpenAI model to use (defaults to gpt-4o-mini)

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
                prompt += "\nYour response should be forceful and include strong language (but remain professional)."
            elif mood.lower() == "happy":
                prompt += "\nGive your analysis in an optimistic tone, highlighting the positive aspects while still noting improvements."
                prompt += "\nUse enthusiastic language and focus on the potential of the code."
            elif mood.lower() == "neutral":
                prompt += "\nGive your analysis in a balanced, objective tone without strong emotional cues."
                prompt += "\nFocus on clear, practical observations and balanced insights."
            elif mood.lower() == "surprise":
                prompt += "\nGive your analysis in a tone of discovery and revelation, highlighting unexpected aspects."
                prompt += "\nEmphasize interesting patterns, unusual approaches, or unexpected design choices in the code."
            elif mood.lower() == "sad":
                prompt += "\nGive your analysis in a thoughtful, slightly melancholy tone."
                prompt += "\nNote missed opportunities and what could have been with the code."

            # Add format guidance for better response structure
            prompt += "\n\nPlease structure your response clearly with headers and bullet points where appropriate."
            prompt += "\nBegin with a brief summary of the code's purpose and main findings."

            # Generate the response using the synchronous method
            response_text = self.generate_response(prompt, temperature=0.7, max_tokens=4096)
            print(f"Generated response, length: {len(response_text)}")

            # Validate response - if it's too short, it might be an error
            if len(response_text) < 100:
                print(f"Warning: Response is unusually short ({len(response_text)} chars)")
                # Add some context to short responses
                response_text = (
                    f"Analysis of {filename} from a {mood} perspective:\n\n{response_text}\n\n"
                    "Note: The analysis is brief - you may want to try again with a different query."
                )

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
