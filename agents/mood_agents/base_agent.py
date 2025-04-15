from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from ..utils.gemini_client import GeminiClient
from ..utils.code_analyzer import CodeAnalyzer

class MoodAgent:
    """Base class for mood-aware debugging agents."""
    
    def __init__(
        self,
        mood: str,
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        """
        Initialize a mood-specific debugging agent.
        
        Args:
            mood: The mood this agent specializes in
            temperature: Creativity level for the model (0.0-1.0)
            api_key: Optional Gemini API key
        """
        self.mood = mood
        self.temperature = temperature
        self.gemini_client = GeminiClient(api_key)
        self.code_analyzer = CodeAnalyzer()
        
        # Base system instructions that all agents will have
        self.base_system_instruction = """
        You are MoodLint, an emotionally intelligent debugging assistant.
        You have dual expertise in programming and psychology, which allows you to provide technical
        debugging while adjusting your communication style to suit the user's emotional state.
        
        You analyze code for bugs, performance issues, and best practices, then provide responses 
        tailored to how users are feeling as they debug their code.
        
        Be concise and helpful, and focus on actionable advice that matches the user's current mood.
        """
    
    def get_system_instruction(self) -> str:
        """Get the system instruction for this agent. Should be overridden by subclasses."""
        return self.base_system_instruction
    
    def analyze_code(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Analyze the code using the code analyzer."""
        return self.code_analyzer.analyze_code(code, filename)
    
    def debug_code(self, code: str, filename: str, user_query: str = "") -> str:
        """
        Debug code with mood-aware responses.
        
        Args:
            code: The source code to debug
            filename: The filename/path
            user_query: Optional specific query about the code
            
        Returns:
            Mood-aware debugging response
        """
        # Analyze the code to get structure and issues
        analysis = self.analyze_code(code, filename)
        
        # Construct the prompt
        system_instruction = self.get_system_instruction()
        
        prompt = f"""
        # Code to Debug
        Filename: {filename}
        
        ```
        {code}
        ```
        
        # Code Analysis Results
        {self._format_analysis_for_prompt(analysis)}
        
        # User Query
        {user_query if user_query else "Please help debug this code."}
        
        Provide a mood-appropriate debugging response. Remember that the user is feeling {self.mood}.
        Focus on the most important issues first. Be specific with line numbers and clear explanations.
        """
        
        # Get response from Gemini
        response = self.gemini_client.generate_response(
            prompt=prompt,
            temperature=self.temperature,
            system_instruction=system_instruction
        )
        
        return response
    
    def _format_analysis_for_prompt(self, analysis: Dict[str, Any]) -> str:
        """Format code analysis results for inclusion in the prompt."""
        output = []
        
        # Include syntax error if present
        if analysis.get("syntax_error"):
            error = analysis["syntax_error"]
            output.append(f"SYNTAX ERROR at line {error['line']}: {error['message']}")
        
        # Include suggestions/issues
        if analysis.get("suggestions"):
            output.append("SUGGESTED IMPROVEMENTS:")
            for suggestion in analysis["suggestions"]:
                output.append(f"- Line {suggestion['line']}: {suggestion['message']} ({suggestion['severity']})")
        
        # Include complexity information for functions
        if analysis.get("complexity") and analysis["complexity"].get("functions"):
            output.append("FUNCTION ANALYSIS:")
            for func_name, func_data in analysis["complexity"]["functions"].items():
                output.append(f"- {func_name}: complexity={func_data['complexity']}, arguments={func_data['args']}, starts at line {func_data['line']}")
        
        return "\n".join(output)