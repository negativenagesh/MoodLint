import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio
import sys
import traceback

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
    # In the debug_with_agent function, around line 83, modify this section:

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
                
                return {**state, "result": result}
            except Exception as agent_error:
                print(f"Agent execution error: {str(agent_error)}")
                error_trace = traceback.format_exc()
                print(f"Agent error traceback: {error_trace}")
                return {
                    **state,
                    "result": {
                        "success": False,
                        "error": f"Agent execution error: {str(agent_error)}",
                        "traceback": error_trace
                    }
                }
        except Exception as e:
            print(f"Outer error in debug_with_agent: {str(e)}")
            error_trace = traceback.format_exc()
            return {
                "error": f"Error in code processing: {str(e)}",
                "traceback": error_trace
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
                print(f"Error detected in state: {error_message}")
                return {"output": DebugResponse(
                    success=False,
                    mood="unknown",
                    response=f"I encountered an error while analyzing your code: {error_message}",
                    error=state["error"] + "\n" + state.get("traceback", "")
                )}
                
            result = state.get("result", {})
            print(f"Result keys from agent: {result.keys()}")
            
            if result.get("success", False):
                # Make sure we include the response
                response = result.get("response", "No response generated")
                print(f"Successful response from agent, length: {len(response)}")
                return {"output": DebugResponse(
                    success=True,
                    mood=result.get("mood", state.get("normalized_mood", state.get("mood", "unknown"))),
                    response=response,
                    error=None
                )}
            else:
                error_msg = result.get("error", "Unknown error occurred")
                response = result.get("response")
                print(f"Error in result: {error_msg}")
                
                if not response:
                    response = f"I couldn't fully analyze your code due to an error: {error_msg}"
                
                return {"output": DebugResponse(
                    success=False,
                    mood=result.get("mood", state.get("normalized_mood", state.get("mood", "unknown"))),
                    response=response,
                    error=error_msg
                )}
        except Exception as e:
            print(f"Error in format_response: {str(e)}")
            return {"output": DebugResponse(
                success=False,
                mood="unknown",
                response=f"An unexpected error occurred while processing your request: {str(e)}",
                error=f"Formatting error: {str(e)}\n{traceback.format_exc()}"
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
                "response": None
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
        response = result["output"] if "output" in result else result
        
        if isinstance(response, DebugResponse):
            # For Pydantic v1
            if hasattr(response, "dict"):
                return response.dict()
            # For Pydantic v2
            return response.model_dump()
        return response
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Workflow error: {str(e)}\n{error_details}")
        return {
            "success": False,
            "error": f"Workflow error: {str(e)}",
            "details": error_details,
            "mood": mood,
            "response": None
        }