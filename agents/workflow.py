import os
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import asyncio
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph import graph
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from agent_manager import AgentManager

# Input/output models
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
    
    # Define workflow components
    @graph.node
    def preprocess_request(request: DebugRequest) -> Dict[str, Any]:
        """Preprocess the debugging request."""
        return {
            "code": request.code,
            "filename": request.filename,
            "mood": request.mood.lower().strip(),
            "query": request.query or "",
            "normalized_mood": agent_manager.normalize_mood(request.mood)
        }
    
    @graph.node
    def debug_with_agent(state: Dict[str, Any]) -> Dict[str, Any]:
        """Debug code using the appropriate mood agent."""
        result = agent_manager.debug_code(
            code=state["code"],
            filename=state["filename"],
            mood=state["normalized_mood"],
            user_query=state["query"]
        )
        return {**state, "result": result}
    
    @graph.node
    def format_response(state: Dict[str, Any]) -> DebugResponse:
        """Format the final response."""
        result = state.get("result", {})
        
        if result.get("success", False):
            return DebugResponse(
                success=True,
                mood=result.get("mood", state.get("normalized_mood", "unknown")),
                response=result.get("response", "No response generated"),
                error=None
            )
        else:
            return DebugResponse(
                success=False,
                mood=result.get("mood", state.get("normalized_mood", "unknown")),
                response=None,
                error=result.get("error", "Unknown error occurred")
            )
    
    # Build the workflow graph
    workflow = graph.Graph()
    workflow.add_node("preprocess", preprocess_request)
    workflow.add_node("debug", debug_with_agent)
    workflow.add_node("format", format_response)
    
    # Define edges
    workflow.add_edge("preprocess", "debug")
    workflow.add_edge("debug", "format")
    
    # Compile the graph
    compiled = workflow.compile()
    
    return compiled

# API interface
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
        filename: Filename/path of the code
        mood: User's current mood
        query: Optional specific query about the code
        
    Returns:
        Debugging results as a dictionary
    """
    workflow = create_workflow()
    
    request = DebugRequest(
        code=code,
        filename=filename,
        mood=mood,
        query=query
    )
    
    # Run the workflow with the request
    config = {"artifacts": {"debug_logs": True}}
    result = await workflow.ainvoke(request, config=config)
    
    # Extract the result from the final node
    response = result[("format", 0)]
    
    return response.dict()

# Command-line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug code with mood-aware agents")
    parser.add_argument("--file", "-f", type=str, help="Path to the file to debug")
    parser.add_argument("--mood", "-m", type=str, default="focused", 
                      help="User mood (happy, frustrated, exhausted, sad, angry)")
    parser.add_argument("--query", "-q", type=str, default="", 
                      help="Specific query about the code")
    
    args = parser.parse_args()
    
    if not args.file:
        print("Error: Please provide a file path with --file")
        sys.exit(1)
    
    # Read the file
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {args.file} does not exist")
        sys.exit(1)
    
    code = file_path.read_text()
    filename = file_path.name
    
    # Run the workflow
    async def main():
        result = await debug_code(code, filename, args.mood, args.query)
        print(json.dumps(result, indent=2))
    
    asyncio.run(main())