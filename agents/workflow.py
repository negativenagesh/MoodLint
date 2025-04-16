import os
import json
from typing import Dict, Any, List, Optional, Union, TypedDict
from pathlib import Path
import asyncio
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Updated import for LangGraph
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from .agent_manager import AgentManager

class DebugRequest(BaseModel):
    """Model for debug request."""
    code: str = Field(..., description="Source code to debug")
    filename: str = Field(..., description="Filename of the code")
    mood: str = Field(..., description="User's current mood")
    query: Optional[str] = Field(None, description="Specific user query about the code")

class DebugResponse(BaseModel):
    """Model for debug response."""
    success: bool = Field(..., description="Whether the operation was successful")
    mood: str = Field(..., description="The mood used for debugging")
    response: Optional[str] = Field(None, description="The debugging response")
    error: Optional[str] = Field(None, description="Error message if any")

# LangGraph workflow definition
def create_workflow():
    """Create the LangGraph debugging workflow."""
    # Create the agent manager
    agent_manager = AgentManager()
    
    # Define workflow components as regular functions (no decorators)
    def preprocess_request(state: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the debugging request."""
        request = state["input"]
        return {
            "code": request.code,
            "filename": request.filename,
            "mood": request.mood.lower().strip(),
            "query": request.query or "",
            "normalized_mood": agent_manager.normalize_mood(request.mood)
        }
    
    def debug_with_agent(state: Dict[str, Any]) -> Dict[str, Any]:
        """Debug code using the appropriate mood agent."""
        result = agent_manager.debug_code(
            code=state["code"],
            filename=state["filename"],
            mood=state["normalized_mood"],
            user_query=state["query"]
        )
        return {**state, "result": result}
    
    def format_response(state: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response."""
        result = state.get("result", {})
        
        if result.get("success", False):
            return {"output": DebugResponse(
                success=True,
                mood=result.get("mood", state.get("normalized_mood", "unknown")),
                response=result.get("response", "No response generated"),
                error=None
            )}
        else:
            return {"output": DebugResponse(
                success=False,
                mood=result.get("mood", state.get("normalized_mood", "unknown")),
                response=None,
                error=result.get("error", "Unknown error occurred")
            )}
    
    # Build the workflow graph using StateGraph without state_type parameter
    workflow = StateGraph(input=DebugRequest, output=DebugResponse)    
    # Add nodes to the graph
    workflow.add_node("preprocess", preprocess_request)
    workflow.add_node("debug", debug_with_agent)
    workflow.add_node("format", format_response)
    
    # Set the entry point
    workflow.set_entry_point("preprocess")
    
    # Define edges
    workflow.add_edge("preprocess", "debug")
    workflow.add_edge("debug", "format")
    
    # Compile the graph
    return workflow.compile()

# Add the missing debug_code function
async def debug_code(
    code: str,
    filename: str,
    mood: str,
    query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Debug code using the mood-aware agent workflow.
    
    Args:
        code: Source code to debug
        filename: Filename of the code
        mood: User's current mood
        query: Optional specific query about the code
        
    Returns:
        Debugging results as a dictionary
    """
    workflow = create_workflow()
    
    # Make sure we're creating the request with all required fields
    request = DebugRequest(
        code=code,
        filename=filename,
        mood=mood,
        query=query
    )
    
    # Run the workflow with the request
    try:
        result = await workflow.ainvoke({"input": request})
        
        # Extract the result from the final output
        response = result["output"]
        
        if isinstance(response, DebugResponse):
            return response.dict()
        return response
    except Exception as e:
        # Return error information for debugging
        return {
            "success": False,
            "error": f"Workflow error: {str(e)}",
            "mood": mood,
            "response": None
        }
    
    # Run the workflow with the request
    result = await workflow.ainvoke({"input": request})
    
    # Extract the result from the final output
    response = result["output"]
    
    if isinstance(response, DebugResponse):
        return response.dict()
    return response