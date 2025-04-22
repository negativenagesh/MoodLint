import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from .agent_manager import AgentManager
from .utils.gemini_client import GeminiClient  # Import the GeminiClient directly

class DebugRequest(BaseModel):
    code: str = Field(..., description="Source code to debug")
    filename: str = Field(..., description="Filename of the code")
    mood: str = Field(..., description="User's current mood")
    query: Optional[str] = Field(None, description="Specific user query about the code")

class DebugResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    mood: str = Field(..., description="The mood used for debugging")
    response: Optional[str] = Field(None, description="The debugging response")
    error: Optional[str] = Field(None, description="Error message if any")
    raw_response: Optional[str] = Field(None, description="Raw response from Gemini API")

def create_workflow():
    # Verify API key is available before creating agent manager
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running.")
    
    agent_manager = AgentManager()

    # Accepts DebugRequest, returns dict
    def preprocess_request(state):
        try:
            # Handle both DebugRequest and dict inputs
            if isinstance(state, DebugRequest):
                return {
                    "code": state.code,
                    "filename": state.filename,
                    "mood": state.mood.lower().strip(),
                    "query": state.query or "",
                    "normalized_mood": agent_manager.normalize_mood(state.mood)
                }
            return state
        except Exception as e:
            print(f"Error in preprocess_request: {str(e)}")
            return {
                "error": f"Preprocessing error: {str(e)}",
                "traceback": traceback.format_exc()
            }

    # Accepts dict or DebugRequest, returns dict
    def debug_with_agent(state):
        try:
            # Check if there was an error in previous step
            if "error" in state:
                return state
                    
            # Convert to dict if it's a DebugRequest
            if isinstance(state, DebugRequest):
                state = preprocess_request(state)
    
            # Safely access state values with fallbacks
            code = state["code"] if "code" in state else ""
            filename = state["filename"] if "filename" in state else "unknown.py"
            mood = state.get("normalized_mood", agent_manager.normalize_mood(state.get("mood", "neutral")))
            query = state.get("query", "")
    
            # Validate inputs - prevent empty or very large code
            if not code:
                raise ValueError("No code provided for analysis")
            if len(code) > 1000000:  # 1MB limit
                raise ValueError("Code file too large for analysis")
    
            print(f"Debug with agent: mood={mood}, filename={filename}")
            
            # Add explicit try-except block for agent debugging
            try:
                result = agent_manager.debug_code(
                    code=code,
                    filename=filename,
                    mood=mood,
                    user_query=query
                )
                
                # Detailed validation of result
                if result is None:
                    raise ValueError("Agent returned None result")
                
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict result, got {type(result).__name__}")
                
                # Enhanced results structure - ensure all important keys exist
                enhanced_result = {
                    "success": result.get("success", True),
                    "mood": result.get("mood", mood),
                    "error": result.get("error", None),
                    "query": query,  # Store the original query
                }
                
                # Store raw response if available
                if "gemini_response" in result:
                    enhanced_result["raw_response"] = result["gemini_response"]
                    print(f"Stored raw Gemini response, length: {len(enhanced_result['raw_response'])}")
                elif "raw_response" in result:
                    enhanced_result["raw_response"] = result["raw_response"]
                    print(f"Stored raw_response from result, length: {len(enhanced_result['raw_response'])}")
                
                # Make sure we have the response key
                if "response" in result and result["response"]:
                    enhanced_result["response"] = result["response"]
                    print(f"Using direct response from result, length: {len(enhanced_result['response'])}")
                elif "analysis" in result and result["analysis"]:
                    enhanced_result["response"] = f"Analysis of {filename}:\n\n{result['analysis']}"
                    print(f"Using analysis from result, transformed to response")
                elif "raw_response" in enhanced_result:
                    # Use raw response if available but no formatted response
                    enhanced_result["response"] = enhanced_result["raw_response"]
                    print(f"Using raw_response as response")
                else:
                    # Create a fallback response if none exists
                    query_context = f" regarding '{query}'" if query else ""
                    enhanced_result["response"] = f"I've analyzed your {filename} file with a {mood} perspective{query_context}."
                    enhanced_result["success"] = False
                    enhanced_result["error"] = "No response was generated by the agent"
                    print(f"Created fallback response due to missing response")
                
                # Return the enhanced result
                return {**state, "result": enhanced_result}
                
            except Exception as agent_error:
                print(f"Agent execution error: {str(agent_error)}")
                error_trace = traceback.format_exc()
                print(f"Agent error traceback: {error_trace}")
                
                # Create a helpful error response
                query_context = f" regarding your query: '{query}'" if query else ""
                fallback_response = f"I attempted to analyze your {filename} file as a {mood} developer{query_context}.\n\nHowever, I encountered an error: {str(agent_error)}"
                
                return {
                    **state,
                    "result": {
                        "success": False,
                        "mood": mood,
                        "error": f"Agent execution error: {str(agent_error)}",
                        "traceback": error_trace,
                        "response": fallback_response,
                        "query": query
                    }
                }
        except Exception as e:
            print(f"Outer error in debug_with_agent: {str(e)}")
            error_trace = traceback.format_exc()
            return {
                "error": f"Error in code processing: {str(e)}",
                "traceback": error_trace,
                "mood": state.get("mood", "neutral"),
                "response": f"Error processing your {state.get('filename', 'code')} file: {str(e)}",
                "query": state.get("query", "")
            }

    # Accepts dict, returns dict with DebugResponse
    def format_response(state):
        try:
            # Convert to dict if it's a DebugRequest
            if isinstance(state, DebugRequest):
                state = preprocess_request(state)
                
            # Check if there was an error in previous steps
            if "error" in state and "result" not in state:
                error_message = state["error"]
                mood = state.get("mood", "unknown")
                query_context = f" regarding your query: '{state.get('query', '')}'" if state.get("query") else ""
                response = state.get("response", f"I encountered an error while analyzing your code{query_context}: {error_message}")
                
                print(f"Error detected in state: {error_message}")
                return {"output": DebugResponse(
                    success=False,
                    mood=mood,
                    response=response,
                    error=state["error"] + "\n" + state.get("traceback", ""),
                    raw_response=state.get("raw_response")
                )}
                
            result = state.get("result", {})
            print(f"Result keys from agent: {result.keys()}")
            
            mood = result.get("mood", state.get("normalized_mood", state.get("mood", "unknown")))
            raw_response = result.get("raw_response")
            
            if result.get("success", False) and "response" in result and result["response"]:
                # Make sure we include the response
                response = result.get("response", "No response generated")
                print(f"Successful response from agent, length: {len(response)}")
                return {"output": DebugResponse(
                    success=True,
                    mood=mood,
                    response=response,
                    error=None,
                    raw_response=raw_response
                )}
            else:
                error_msg = result.get("error", "Unknown error occurred")
                response = result.get("response")
                print(f"Error in result: {error_msg}")
                
                # If we have a raw response and no proper response, use the raw response
                if (not response or len(response or "") < 50) and raw_response:
                    print(f"Using raw_response because response is missing or too short")
                    response = raw_response
                elif not response:
                    query_context = f" regarding '{state.get('query', '')}'" if state.get("query") else ""
                    response = f"I couldn't fully analyze your code{query_context} due to an error: {error_msg}"
                
                return {"output": DebugResponse(
                    success=False,
                    mood=mood,
                    response=response,
                    error=error_msg,
                    raw_response=raw_response
                )}
        except Exception as e:
            print(f"Error in format_response: {str(e)}")
            return {"output": DebugResponse(
                success=False,
                mood="unknown",
                response=f"An unexpected error occurred while processing your request: {str(e)}",
                error=f"Formatting error: {str(e)}\n{traceback.format_exc()}",
                raw_response=None
            )}

    workflow = StateGraph(input=DebugRequest, output=DebugResponse)
    workflow.add_node("preprocess", preprocess_request)
    workflow.add_node("debug", debug_with_agent)
    workflow.add_node("format", format_response)
    workflow.set_entry_point("preprocess")
    workflow.add_edge("preprocess", "debug")
    workflow.add_edge("debug", "format")
    return workflow.compile()

async def debug_code(
    code: str,
    filename: str,
    mood: str,
    query: Optional[str] = None
) -> Dict[str, Any]:
    try:
        # Check API key first
        if not os.environ.get("GOOGLE_API_KEY"):
            return {
                "success": False,
                "error": "GOOGLE_API_KEY environment variable is not set. Please set it before running.",
                "mood": mood,
                "response": f"I can't analyze your {filename} file because the API key is missing. Please set the GOOGLE_API_KEY environment variable."
            }
            
        workflow = create_workflow()
        request = DebugRequest(
            code=code,
            filename=filename,
            mood=mood,
            query=query
        )
        
        print(f"Invoking workflow with request for file: {filename}, mood: {mood}")
        result = await workflow.ainvoke(request)
        print(f"Debug code completed, result: {result.keys() if isinstance(result, dict) else type(result)}")
        
        # Store the raw response for debugging
        raw_response = None
        
        # Enhanced response extraction with better debugging
        if isinstance(result, dict) and "output" in result and isinstance(result["output"], DebugResponse):
            response = result["output"]
            print(f"Found response in output object, extracting")
            
            # Store raw response if available
            if hasattr(response, "raw_response") and response.raw_response:
                raw_response = response.raw_response
                print(f"Found raw_response in output object, length: {len(raw_response)}")
            
            # For Pydantic v1
            if hasattr(response, "dict"):
                response_dict = response.dict()
                print(f"Extracted via dict() method, keys: {response_dict.keys()}")
                # Make sure raw_response carries through
                if raw_response and "raw_response" not in response_dict:
                    response_dict["raw_response"] = raw_response
                
                # Ensure we preserve the actual response content even if it comes through raw_response
                if "response" in response_dict and (not response_dict["response"] or len(response_dict["response"]) < 100) and "raw_response" in response_dict:
                    response_dict["response"] = response_dict["raw_response"]
                    print(f"Using raw_response as main response due to empty or short response")
                
                return response_dict
                
            # For Pydantic v2
            response_dict = response.model_dump()
            print(f"Extracted via model_dump() method, keys: {response_dict.keys()}")
            # Make sure raw_response carries through
            if raw_response and "raw_response" not in response_dict:
                response_dict["raw_response"] = raw_response
            
            # Ensure we preserve the actual response content even if it comes through raw_response
            if "response" in response_dict and (not response_dict["response"] or len(response_dict["response"]) < 100) and "raw_response" in response_dict:
                response_dict["response"] = response_dict["raw_response"]
                print(f"Using raw_response as main response due to empty or short response")
            
            return response_dict
        
        # If we have a direct response in the result
        if isinstance(result, dict):
            # Keep track of the original Gemini response
            if "raw_response" in result:
                raw_response = result["raw_response"]
                print(f"Found raw_response in result dictionary")
            elif "gemini_response" in result:
                raw_response = result["gemini_response"]
                print(f"Found gemini_response in result dictionary")
                if "raw_response" not in result:
                    result["raw_response"] = raw_response
            
            # Check for fallback response
            fallback_indicators = ["API connection", "wasn't able to generate", "check that your API key"]
            is_fallback = False
            
            if "response" in result and isinstance(result["response"], str):
                is_fallback = any(indicator in result["response"] for indicator in fallback_indicators)
                if is_fallback and raw_response:
                    print(f"Detected fallback response, replacing with raw_response")
                    result["response"] = raw_response
            
            if "response" in result and result["response"] and not is_fallback:
                print(f"Found response directly in result, length: {len(result['response'])}")
                # Add the raw response to the result if we have it
                if raw_response and "raw_response" not in result:
                    result["raw_response"] = raw_response
                return result
                
            # Check for nested result
            if "result" in result and isinstance(result["result"], dict):
                nested = result["result"]
                print(f"Nested result keys: {nested.keys()}")
                
                # Extract raw response from nested result if available
                if "raw_response" in nested:
                    raw_response = nested["raw_response"]
                    print(f"Found raw_response in nested result")
                elif "gemini_response" in nested:
                    raw_response = nested["gemini_response"]
                    print(f"Found gemini_response in nested result")
                    
                # Check if the nested response is a fallback
                if "response" in nested:
                    nested_is_fallback = False
                    if isinstance(nested["response"], str):
                        nested_is_fallback = any(indicator in nested["response"] for indicator in fallback_indicators)
                        
                    if nested_is_fallback and raw_response:
                        print(f"Detected fallback in nested response, using raw_response")
                        nested["response"] = raw_response
                        
                    print(f"Found response in nested result, length: {len(nested['response'])}")
                    # Combine top-level and nested result data
                    combined_result = {
                        "success": nested.get("success", True),
                        "mood": nested.get("mood", mood),
                        "response": nested["response"],
                        "result": nested
                    }
                    
                    # Add raw response if available
                    if raw_response:
                        combined_result["raw_response"] = raw_response
                        
                    return combined_result
        
        # Direct string response
        if isinstance(result, str) and len(result) > 0:
            print(f"Result is a string, length: {len(result)}")
            response_dict = {
                "success": True,
                "mood": mood,
                "response": result,
                "result": {"mood": mood}
            }
            
            # Add raw response if available
            if raw_response:
                response_dict["raw_response"] = raw_response
                
            return response_dict
            
        # Create more useful fallback response
        print(f"Unable to extract standard response format, using fallback")
        
        # If we have a raw response, use it instead of the generic fallback
        if raw_response:
            print(f"Using raw_response as the primary response instead of fallback text")
            return {
                "success": True,
                "mood": mood,
                "response": raw_response,
                "raw_response": raw_response,
                "result": {"mood": mood, "raw_response": raw_response}
            }
        
        # No raw response available, use generic fallback
        query_context = f" about your query: '{query}'" if query else ""
        fallback_response = (
            f"I analyzed your {filename} file from the perspective of a {mood} developer{query_context}.\n\n"
            "However, I wasn't able to generate a detailed analysis. This might be due to an issue with the API connection.\n\n"
            "I recommend checking that your API key has the proper permissions and trying again with a more specific query."
        )
        
        fallback_result = {
            "success": True,
            "mood": mood,
            "response": fallback_response,
            "result": {"mood": mood}
        }
        
        return fallback_result
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Workflow error: {str(e)}\n{error_details}")
        query_context = f" regarding your query: '{query}'" if query else ""
        return {
            "success": False,
            "error": f"Workflow error: {str(e)}",
            "details": error_details,
            "mood": mood,
            "response": f"I encountered an error while analyzing your {filename} file{query_context}: {str(e)}"
        }