from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from ..utils.openai_client import OpenAIClient
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
            api_key: Optional OpenAI API key
        """
        self.mood = mood
        self.temperature = temperature
        self.openai_client = OpenAIClient(api_key)
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
        
        # Get response from OpenAI
        response = self.openai_client.generate_response(
            prompt=prompt,
            temperature=self.temperature,
            system_instruction=system_instruction
        )
        
        return response
    
    def debug_code_with_tools(self, code: str, filename: str, user_query: str = "", output_dir: str = None, stream_callback=None) -> Dict[str, Any]:
        """
        Debug code using tools (read/edit/create files).
        """
        from ..utils.tools import ToolRegistry
        
        # Analyze code first (optional, but helpful context)
        analysis_context = ""
        if code:
            analysis = self.analyze_code(code, filename)
            analysis_context = self._format_analysis_for_prompt(analysis)
            
        system_instruction = self.get_system_instruction()
        
        # Create a detailed prompt that encourages tool usage
        prompt = f"""
        # Context
        User Mood: {self.mood.upper()}
        Filename: {filename}
        
        # User Query
        {user_query}
        
        # Code Content
        ```
        {code if code else "(No content provided)"}
        ```
        
        {f"# Analysis Results{chr(10)}{analysis_context}" if analysis_context else ""}
        
        You have access to tools to READ, EDIT, and CREATE files.
        - If the user asks to create a file, use `create_file`.
        - If the user asks to fix code, use `edit_file`.
        - If you need more context, use `read_file`.
        
        IMPORTANT:
        {f"- When creating new files, ALWAYS save them to this directory: {output_dir}" if output_dir else ""}
        - If no path is specified by the user, assume the file should be created in the directory above.
        
        Provide a helpful response that addresses the user's needs while maintaining the {self.mood} persona.
        """
        
        # Get tools definition
        tools = ToolRegistry.get_tool_definitions()
        
        # Execute with tools
        result = self.openai_client.generate_response_with_tools(
            prompt=prompt,
            tools=tools,
            system_instruction=system_instruction,
            max_iterations=10,
            stream_callback=stream_callback
        )
        
        return result
    
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