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
    """
    Create the workflow for code debugging.
    
    Returns:
        The configured workflow or None if creation fails
    """
    try:
        # Verify API key is available before creating agent manager
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY environment variable is not set.")
            return None
        
        # Create agent manager
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
                    
                    # Get direct raw response from Gemini API if available
                    raw_gemini_response = None
                    if "raw_response" in result and result["raw_response"]:
                        raw_gemini_response = result["raw_response"]
                        print(f"Found raw_response in agent result, length: {len(raw_gemini_response)}")
                    elif "gemini_response" in result and result["gemini_response"]:
                        raw_gemini_response = result["gemini_response"]
                        print(f"Found gemini_response in agent result, length: {len(raw_gemini_response)}")
                    elif "response" in result and isinstance(result["response"], str) and len(result["response"]) > 100:
                        # If response is substantial, save it as raw_response too (backup approach)
                        raw_gemini_response = result["response"]
                        print(f"Using substantial response as raw_gemini_response, length: {len(raw_gemini_response)}")
                    
                    # Enhanced results structure with raw response preservation
                    enhanced_result = {
                        "success": result.get("success", True),
                        "mood": result.get("mood", mood),
                        "error": result.get("error", None),
                        "query": query,
                        "raw_response": raw_gemini_response
                    }
                    
                    # Ensure we have a response, prioritizing meaningful content
                    if "response" in result and result["response"] and len(result["response"]) > 100:
                        enhanced_result["response"] = result["response"]
                        print(f"Using direct response from result, length: {len(enhanced_result['response'])}")
                    elif raw_gemini_response:
                        # Use raw response if available but no formatted response
                        enhanced_result["response"] = raw_gemini_response
                        print(f"Using raw_gemini_response as primary response")
                    elif "analysis" in result and result["analysis"]:
                        enhanced_result["response"] = f"Analysis of {filename}:\n\n{result['analysis']}"
                        print(f"Using analysis from result, transformed to response")
                    else:
                        # Create a fallback response if none exists
                        query_context = f" regarding '{query}'" if query else ""
                        enhanced_result["response"] = f"I've analyzed your {filename} file with a {mood} perspective{query_context}."
                        enhanced_result["success"] = False
                        enhanced_result["error"] = "No response was generated by the agent"
                        print(f"Created fallback response due to missing response")
                    
                    # Return the enhanced result, preserving both raw and formatted responses
                    return {**state, "result": enhanced_result, "raw_response": raw_gemini_response}
                    
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
                    
                # Capture raw_response from state if available
                raw_response = state.get("raw_response")
                if raw_response:
                    print(f"Found raw_response at top level of state, length: {len(raw_response)}")
                    
                # Check if there was an error in previous steps
                if "error" in state and "result" not in state:
                    error_message = state["error"]
                    mood = state.get("mood", "unknown")
                    query_context = f" regarding your query: '{state.get('query', '')}'" if state.get("query") else ""
                    
                    # If we have a raw response, use it instead of a generic error message
                    if raw_response and len(raw_response) > 100:
                        response = raw_response
                        print(f"Using raw_response for error state")
                    else:
                        response = state.get("response", f"I encountered an error while analyzing your code{query_context}: {error_message}")
                    
                    print(f"Error detected in state: {error_message}")
                    return {"output": DebugResponse(
                        success=False,
                        mood=mood,
                        response=response,
                        error=state["error"] + "\n" + state.get("traceback", ""),
                        raw_response=raw_response
                    )}
                    
                result = state.get("result", {})
                print(f"Result keys from agent: {result.keys()}")
                
                # If no raw_response at top level, check in result
                if not raw_response and "raw_response" in result:
                    raw_response = result["raw_response"]
                    print(f"Found raw_response in result, length: {len(raw_response)}")
                
                mood = result.get("mood", state.get("normalized_mood", state.get("mood", "unknown")))
                
                # If we have a raw response of meaningful length, ensure it's used
                if raw_response and len(raw_response) > 100:
                    print(f"Using substantial raw_response of {len(raw_response)} characters")
                    return {"output": DebugResponse(
                        success=True,
                        mood=mood,
                        response=raw_response,
                        error=None,
                        raw_response=raw_response
                    )}
                
                # Otherwise follow normal processing
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

        # Define type for the state
        class AgentWorkflowState(dict):
            """Type for the state in the workflow."""
            pass

        # Create a workflow with error handling
        try:
            # Create the StateGraph
            print("Creating StateGraph...")
            workflow = StateGraph(AgentWorkflowState)
            
            # Add nodes to the graph
            print("Adding nodes to workflow...")
            workflow.add_node("preprocess", preprocess_request)
            workflow.add_node("debug", debug_with_agent)
            workflow.add_node("format", format_response)
            
            # Define the edges
            print("Setting up workflow flow...")
            workflow.set_entry_point("preprocess")
            workflow.add_edge("preprocess", "debug")
            workflow.add_edge("debug", "format")
            
            # Compile the workflow
            print("Compiling workflow...")
            app = workflow.compile()
            
            print("Workflow created and compiled successfully")
            return app
            
        except Exception as workflow_error:
            print(f"Error creating workflow: {str(workflow_error)}")
            print(traceback.format_exc())
            
            # Create a fallback simple workflow that skips langgraph
            print("Creating fallback direct workflow...")
            
            try:
                # Function to process request directly
                async def direct_process(request):
                    if isinstance(request, DebugRequest):
                        code = request.code
                        filename = request.filename
                        mood = request.mood
                        query = request.query
                    else:
                        code = request.get("code", "")
                        filename = request.get("filename", "unknown.py")
                        mood = request.get("mood", "happy")
                        query = request.get("query", "")
                    
                    print(f"Direct processing: {filename} with mood {mood}")
                    
                    # Call agent directly
                    result = agent_manager.debug_code(
                        code=code,
                        filename=filename, 
                        mood=mood,
                        user_query=query
                    )
                    
                    # Create standardized response
                    response = result.get("response", "")
                    raw_response = result.get("raw_response", response)
                    
                    return {
                        "output": DebugResponse(
                            success=True,
                            mood=mood,
                            response=response,
                            error=None,
                            raw_response=raw_response
                        )
                    }
                
                # Create a simple callable object as a fallback workflow
                class DirectWorkflow:
                    async def ainvoke(self, request):
                        return await direct_process(request)
                
                print("Created direct workflow fallback")
                return DirectWorkflow()
                
            except Exception as fallback_error:
                print(f"Failed to create fallback workflow: {str(fallback_error)}")
                print(traceback.format_exc())
                return None
                
    except Exception as e:
        print(f"ERROR creating workflow: {str(e)}")
        traceback.print_exc()
        return None

async def debug_code(
    code: str,
    filename: str,
    mood: str,
    query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Debug code using the appropriate mood-based agent.
    
    Args:
        code: The code string to debug
        filename: The name of the file
        mood: The mood to use for debugging
        query: Optional specific query about the code
        
    Returns:
        Dictionary with debugging results
    """
    try:
        # Verify API key is available
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "GOOGLE_API_KEY environment variable is not set",
                "mood": mood,
                "response": "I couldn't analyze your code because the Google API key is missing. Please set the GOOGLE_API_KEY environment variable."
            }
        
        print(f"Using API key: {api_key[:4]}...{api_key[-4:]} (length: {len(api_key)})")
        
        # Create debug request
        request = DebugRequest(
            code=code,
            filename=filename,
            mood=mood,
            query=query or ""
        )
        
        print(f"Invoking workflow with request for file: {filename}, mood: {mood}")
        
        # Create the workflow
        try:
            workflow = create_workflow()
            
            # Check if workflow was created successfully
            if workflow is None:
                print("WARNING: Workflow creation failed, falling back to direct agent call")
                # Fallback to direct agent invocation
                agent_manager = AgentManager()
                direct_result = agent_manager.debug_code(
                    code=code, 
                    filename=filename, 
                    mood=mood, 
                    user_query=query or ""
                )
                
                # Process the direct result
                if direct_result:
                    # Make sure we store any raw response
                    if "response" in direct_result and len(direct_result["response"]) > 50:
                        if "raw_response" not in direct_result:
                            direct_result["raw_response"] = direct_result["response"]
                    
                    print(f"Direct agent call returned result of type: {type(direct_result)}")
                    return direct_result
                else:
                    raise ValueError("Direct agent call failed too")
            
            # Invoke the workflow with error handling
            try:
                result = await workflow.ainvoke(request)
                
                # Check for output field (standard response structure)
                if "output" in result and isinstance(result["output"], DebugResponse):
                    debug_response = result["output"]
                    print(f"Result type: DebugResponse with keys {debug_response.__dict__.keys()}")
                    
                    # Convert Pydantic model to dict for JSON serialization
                    return {
                        "success": debug_response.success,
                        "mood": debug_response.mood,
                        "response": debug_response.response,
                        "error": debug_response.error,
                        "raw_response": debug_response.raw_response
                    }
                
                # If we have a standard result dict, return it directly
                print(f"Result type: dict with keys {result.keys()}")
                return result
                
            except Exception as invoke_error:
                print(f"Error invoking workflow: {str(invoke_error)}")
                print(traceback.format_exc())
                raise invoke_error
            
        except Exception as workflow_error:
            # Handle workflow creation/execution errors
            error_traceback = traceback.format_exc()
            error_message = f"Workflow error: {str(workflow_error)}"
            print(error_message)
            print(error_traceback)
            
            # Fall back to direct agent invocation
            try:
                # Try using agent manager directly as fallback
                agent_manager = AgentManager()
                direct_result = agent_manager.debug_code(code, filename, mood, query or "")
                
                if direct_result and isinstance(direct_result, dict) and "response" in direct_result:
                    print("Successfully used direct agent_manager as fallback")
                    return direct_result
            except Exception as agent_error:
                print(f"Fallback to direct agent also failed: {str(agent_error)}")
                
            # Return structured error response
            return {
                "success": False,
                "error": error_message,
                "details": error_traceback,
                "mood": mood,
                "response": f"I encountered an error while analyzing your {filename} file{' regarding your query: ' + repr(query) if query else ''}: {str(workflow_error)}"
            }
            
    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error during debug_code: {str(e)}"
        error_traceback = traceback.format_exc()
        print(error_message)
        print(error_traceback)
        
        # Return a structured error response
        return {
            "success": False,
            "error": error_message,
            "details": error_traceback,
            "mood": mood,
            "response": f"An unexpected error occurred while analyzing your {filename} file: {str(e)}"
        }