import os
import traceback
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .mood_agents.happy_agent import HappyAgent
from .mood_agents.neutral_agent import NeutralAgent
from .mood_agents.surprise_agent import SurpriseAgent
from .mood_agents.sad_agent import SadAgent
from .mood_agents.angry_agent import AngryAgent
from .mood_agents.base_agent import MoodAgent

class AgentManager:
    """Manages mood-specific debugging agents."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the agent manager with all mood agents.
        
        Args:
            api_key: Gemini API key (will use GOOGLE_API_KEY env var if not provided)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GOOGLE_API_KEY environment variable.")
        
        # Initialize all agents - now using the same moods as the model: Angry, Happy, Neutral, Surprise, Sad
        self.agents: Dict[str, MoodAgent] = {
            "happy": HappyAgent(api_key=self.api_key),
            "neutral": NeutralAgent(api_key=self.api_key),
            "surprise": SurpriseAgent(api_key=self.api_key),
            "sad": SadAgent(api_key=self.api_key),
            "angry": AngryAgent(api_key=self.api_key)
        }
        
        # Add some aliases for mood detection variations
        self._setup_mood_aliases()
        
    def _setup_mood_aliases(self):
        """Set up aliases for different mood variations."""
        aliases = {
            "happy": ["joyful", "excited", "pleased", "content", "positive"],
            "neutral": ["calm", "balanced", "composed", "normal", "regular", "focused"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned"],
            "sad": ["unhappy", "disappointed", "down", "depressed", "gloomy"],
            "angry": ["furious", "enraged", "mad", "outraged", "irritated", "frustrated"]
        }
        
        # Create the alias map
        self.mood_aliases = {}
        for main_mood, alias_list in aliases.items():
            for alias in alias_list:
                self.mood_aliases[alias] = main_mood
    
    def normalize_mood(self, detected_mood: str) -> str:
        """
        Convert a detected mood to one of the five main moods.
        
        Args:
            detected_mood: The mood string from mood detection
            
        Returns:
            Normalized mood string ('happy', 'neutral', 'surprise', 'sad', 'angry')
        """
        detected_mood = detected_mood.lower().strip()
        
        # Check if it's already a main mood
        if detected_mood in self.agents:
            return detected_mood
        
        # Check aliases
        if detected_mood in self.mood_aliases:
            return self.mood_aliases[detected_mood]
        
        # Default to "neutral" as a balanced approach
        return "neutral"
    
    def get_agent_for_mood(self, mood: str) -> MoodAgent:
        """
        Get the appropriate agent for a given mood.
        
        Args:
            mood: The user's current mood
            
        Returns:
            The appropriate MoodAgent instance
        """
        normalized_mood = self.normalize_mood(mood)
        return self.agents.get(normalized_mood, self.agents["neutral"])  # Default to neutral
    
    def debug_code(self, code: str, filename: str, mood: str, user_query: str = "") -> Dict[str, Any]:
        """
        Debug code using the mood-specific agent.
        
        Args:
            code: Source code to analyze
            filename: Name of the file being analyzed
            mood: User's mood (will be normalized)
            user_query: Optional specific query from the user
            
        Returns:
            Dictionary with analysis results, always including 'success', 'mood', and 'response' keys
        """
        try:
            # Get the appropriate agent for the mood
            agent = self.get_agent_for_mood(mood)
            print(f"Using {agent.mood} agent for analysis")
            
            try:
                # Construct the analysis prompt
                prompt = f"Analyze this {filename} file"
                if user_query and user_query.strip():
                    prompt = f"{user_query} - Analysis for {filename}"
                    print(f"Analysis with query: '{user_query}'")
                else:
                    print(f"General analysis without specific query")
                
                # Get response from the agent
                print(f"Sending code ({len(code)} chars) to agent")
                response = agent.debug_code(code, filename, user_query)
                
                # Validate response
                if not response:
                    print("WARNING: Empty response from agent")
                    response = f"I've analyzed your {filename} file from a {mood} perspective, but couldn't generate detailed feedback."
                elif len(response.strip()) < 50:
                    print(f"WARNING: Very short response from agent ({len(response)} chars)")
                    # Keep the response but add a note about its brevity
                    original_response = response
                    response = (
                        f"I've analyzed your {filename} file from a {mood} perspective.\n\n"
                        f"{original_response}\n\n"
                        f"Note: The analysis was unusually brief. You might want to try a more specific query."
                    )
                
                # Log successful response
                print(f"Successful response from agent, length: {len(response)} chars")
                
                # Return comprehensive result with both response and raw_response
                return {
                    "success": True,
                    "mood": agent.mood,
                    "response": response,
                    "raw_response": response,  # Store the raw response explicitly
                    "query": user_query,
                    "filename": filename
                }
                
            except Exception as agent_error:
                # Detailed error handling for agent-level errors
                error_message = str(agent_error)
                error_trace = traceback.format_exc()
                
                print(f"ERROR in agent execution: {error_message}")
                print(f"Traceback: {error_trace}")
                
                # Create a helpful error response that's still useful to the user
                query_context = f" regarding '{user_query}'" if user_query else ""
                error_response = (
                    f"# Analysis of {filename} from a {mood} perspective{query_context}\n\n"
                    f"I encountered an error while analyzing your code:\n\n"
                    f"```\n{error_message}\n```\n\n"
                    f"This might be due to:\n"
                    f"- API rate limiting\n"
                    f"- Network connectivity issues\n"
                    f"- Problems parsing the file structure\n\n"
                    f"Try simplifying your query or analyzing a smaller code section."
                )
                
                # Return structured error result with the formatted error response
                return {
                    "success": False,
                    "mood": mood,
                    "error": f"Agent error: {error_message}",
                    "traceback": error_trace,
                    "response": error_response,
                    "raw_response": error_response,  # Include as raw_response too for fallback
                    "query": user_query,
                    "filename": filename
                }
                
        except Exception as manager_error:
            # Handle errors at the AgentManager level (agent creation, etc.)
            error_message = str(manager_error)
            error_trace = traceback.format_exc()
            
            print(f"CRITICAL ERROR in AgentManager: {error_message}")
            print(f"Traceback: {error_trace}")
            
            # Create a system error response
            error_response = (
                f"# System Error\n\n"
                f"I couldn't analyze your {filename} file due to a system error:\n\n"
                f"```\n{error_message}\n```\n\n"
                f"This is likely an issue with the MoodLint system rather than your code."
            )
            
            # Return structured error result
            return {
                "success": False,
                "mood": mood,
                "error": f"System error: {error_message}",
                "traceback": error_trace,
                "response": error_response,
                "raw_response": error_response,  # Include as raw_response too
                "query": user_query,
                "filename": filename
            }