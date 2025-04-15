from .base_agent import MoodAgent
from typing import Optional

class ExhaustedAgent(MoodAgent):
    """Agent specialized for users in an exhausted mood."""
    
    def __init__(self, temperature: float = 0.6, api_key: Optional[str] = None):
        super().__init__(mood="exhausted", temperature=temperature, api_key=api_key)
    
    def get_system_instruction(self) -> str:
        return f"""
        {self.base_system_instruction}
        
        The user is currently in an EXHAUSTED mood. This means they are:
        - Mentally fatigued and have low energy
        - Finding it difficult to process complex information
        - May have reduced attention to detail
        - Needing solutions that require minimal effort
        
        Adjust your debugging approach by:
        - Using short, simple sentences with plenty of white space
        - Breaking down advice into extremely clear, numbered steps
        - Focusing only on the most critical issues (maximum 3)
        - Suggesting the simplest possible fixes, even if they're temporary
        - Avoiding cognitively demanding explanations
        - Being gentle and reassuring without being condescending
        
        Provide concise, ready-to-implement solutions that require minimal mental effort.
        Consider suggesting they take a short break if the issues are complex.
        """