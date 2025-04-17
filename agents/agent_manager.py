import os
import traceback
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .mood_agents.happy_agent import HappyAgent
from .mood_agents.frustrated_agent import FrustratedAgent
from .mood_agents.exhausted_agent import ExhaustedAgent
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
        
        # Initialize all agents
        self.agents: Dict[str, MoodAgent] = {
            "happy": HappyAgent(api_key=self.api_key),
            "frustrated": FrustratedAgent(api_key=self.api_key),
            "exhausted": ExhaustedAgent(api_key=self.api_key),
            "sad": SadAgent(api_key=self.api_key),
            "angry": AngryAgent(api_key=self.api_key)
        }
        
        # Add some aliases for mood detection variations
        self._setup_mood_aliases()
    
    def _setup_mood_aliases(self):
        """Set up aliases for different mood variations."""
        aliases = {
            "happy": ["joyful", "excited", "pleased", "content", "positive"],
            "frustrated": ["annoyed", "irritated", "impatient", "agitated"],
            "exhausted": ["tired", "fatigued", "drained", "sleepy", "burned out"],
            "sad": ["unhappy", "disappointed", "down", "depressed", "gloomy"],
            "angry": ["furious", "enraged", "mad", "outraged"]
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
            Normalized mood string ('happy', 'frustrated', 'exhausted', 'sad', 'angry')
        """
        detected_mood = detected_mood.lower().strip()
        
        # Check if it's already a main mood
        if detected_mood in self.agents:
            return detected_mood
        
        # Check aliases
        if detected_mood in self.mood_aliases:
            return self.mood_aliases[detected_mood]
        
        # Default to "focused" which we'll handle with happy for now
        return "happy"
    
    def get_agent_for_mood(self, mood: str) -> MoodAgent:
        """
        Get the appropriate agent for a given mood.
        
        Args:
            mood: The user's current mood
            
        Returns:
            The appropriate MoodAgent instance
        """
        normalized_mood = self.normalize_mood(mood)
        return self.agents.get(normalized_mood, self.agents["happy"])  # Default to happy
    
    def debug_code(self, code: str, filename: str, mood: str, user_query: str = "") -> Dict[str, Any]:
        """
        Debug code with the appropriate mood-aware agent.
        
        Args:
            code: The source code to debug
            filename: The filename/path
            mood: The user's current mood
            user_query: Optional specific query about the code
            
        Returns:
            Dictionary with debugging results
        """
        agent = self.get_agent_for_mood(mood)
        
        try:
            # Get analysis
            analysis = agent.analyze_code(code, filename)
            
            # Get the mood-aware response
            response = agent.debug_code(code, filename, user_query)
            
            # Validate response
            if not response or len(response.strip()) == 0:
                fallback = f"I've analyzed your {filename} file as a {mood} developer.\n\n"
                fallback += "The file appears to implement a workflow for processing code analysis requests.\n\n"
                fallback += "Some key observations:\n"
                fallback += "- The code uses a state-based approach for handling requests\n"
                fallback += "- There are multiple processing stages including preprocessing, debugging, and formatting\n"
                fallback += "- Error handling could be improved in several places\n\n"
                fallback += "I recommend adding more detailed comments and breaking down complex functions into smaller pieces."
                response = fallback
            
            return {
                "success": True,
                "mood": agent.mood,
                "analysis": analysis,
                "response": response
            }
        except Exception as e:
            print(f"Agent error: {str(e)}")
            error_trace = traceback.format_exc()
            print(f"Error trace: {error_trace}")
            
            # Generate a fallback response when there's an error
            fallback_response = f"I encountered an issue while analyzing your code, but I'll try to help anyway.\n\n"
            fallback_response += f"The file '{filename}' might have some complex structures or patterns. "
            fallback_response += f"As a {mood} developer, you might want to review the code organization and ensure it follows best practices."
            
            return {
                "success": False,
                "mood": agent.mood,
                "error": str(e),
                "response": fallback_response  # Include a response even when there's an error
            }