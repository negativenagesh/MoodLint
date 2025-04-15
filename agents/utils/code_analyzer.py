import ast
import re
from typing import Dict, List, Tuple, Any, Optional

class CodeAnalyzer:
    """Analyzes code to extract information and detect potential issues."""
    
    @staticmethod
    def extract_syntax_error(code: str) -> Optional[Tuple[int, str]]:
        """
        Try to parse code and return syntax error details if any.
        
        Args:
            code: The source code to analyze
            
        Returns:
            Tuple of (line_number, error_message) or None if no syntax errors
        """
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return (e.lineno, str(e))
    
    @staticmethod
    def analyze_function_complexity(code: str) -> Dict[str, Any]:
        """
        Analyze code to determine function complexity.
        
        Args:
            code: The source code to analyze
            
        Returns:
            Dictionary with complexity metrics
        """
        results = {}
        
        try:
            tree = ast.parse(code)
            
            # Find all function definitions
            functions = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Calculate cyclomatic complexity approximately
                    complexity = 1  # Base complexity
                    
                    # Count branches that increase complexity
                    for subnode in ast.walk(node):
                        if isinstance(subnode, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                            complexity += 1
                        elif isinstance(subnode, ast.BoolOp):
                            # Each and/or adds a branch
                            complexity += len(subnode.values) - 1
                    
                    functions[node.name] = {
                        "line": node.lineno,
                        "complexity": complexity,
                        "args": len(node.args.args),
                        # Check if function is too long
                        "too_long": (node.end_lineno - node.lineno) > 30 if hasattr(node, 'end_lineno') else False
                    }
            
            results["functions"] = functions
            
            # Count variables and imports
            results["variables"] = len([n for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)])
            results["imports"] = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
            
            # Identify problematic patterns
            results["issues"] = CodeAnalyzer._identify_issues(tree, code)
            
        except SyntaxError:
            # If there's a syntax error, we already catch it in extract_syntax_error
            results["error"] = "Syntax error prevents analysis"
        
        return results
    
    @staticmethod
    def _identify_issues(tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Identify potential code issues."""
        issues = []
        
        # Check for except blocks with bare except
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append({
                    "line": node.lineno,
                    "type": "bare_except",
                    "message": "Using bare 'except:' is not recommended, catch specific exceptions instead."
                })
            
            # Check for potentially unused variables
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                name = node.id
                # Skip common ok patterns
                if name.startswith('_') or name == 'self' or name == 'cls':
                    continue
                
                # Simple check for variable use
                uses = [n for n in ast.walk(tree) if 
                        isinstance(n, ast.Name) and 
                        n.id == name and 
                        isinstance(n.ctx, ast.Load)]
                
                if len(uses) <= 1:  # Only the assignment itself or no uses
                    issues.append({
                        "line": node.lineno,
                        "type": "unused_variable",
                        "message": f"Variable '{name}' might be unused."
                    })
            
            # Check for very nested code (more than 3 levels)
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                nested_depth = 0
                parent = node
                while hasattr(parent, 'parent') and parent.parent is not None:
                    if isinstance(parent.parent, (ast.If, ast.For, ast.While, ast.With)):
                        nested_depth += 1
                    parent = parent.parent
                
                if nested_depth > 3:
                    issues.append({
                        "line": node.lineno,
                        "type": "deep_nesting",
                        "message": f"Code has deep nesting (level {nested_depth}), consider refactoring."
                    })
        
        return issues

    @staticmethod
    def analyze_code(code: str, filename: str = "") -> Dict[str, Any]:
        """
        Perform a complete analysis of the code.
        
        Args:
            code: The source code to analyze
            filename: Optional filename for language-specific analysis
            
        Returns:
            Analysis results dictionary
        """
        results = {
            "filename": filename,
            "language": CodeAnalyzer._detect_language(filename),
            "syntax_error": None,
            "complexity": {},
            "suggestions": []
        }
        
        # Check for syntax errors first
        syntax_error = CodeAnalyzer.extract_syntax_error(code)
        if syntax_error:
            results["syntax_error"] = {
                "line": syntax_error[0],
                "message": syntax_error[1]
            }
            return results
        
        # If no syntax errors, perform full analysis
        complexity_results = CodeAnalyzer.analyze_function_complexity(code)
        results["complexity"] = complexity_results
        
        # Generate suggestions based on analysis
        for func_name, func_data in complexity_results.get("functions", {}).items():
            if func_data["complexity"] > 10:
                results["suggestions"].append({
                    "line": func_data["line"],
                    "message": f"Function '{func_name}' has high complexity ({func_data['complexity']}). Consider breaking it down.",
                    "severity": "warning"
                })
            
            if func_data["too_long"]:
                results["suggestions"].append({
                    "line": func_data["line"],
                    "message": f"Function '{func_name}' is too long. Consider breaking it into smaller functions.",
                    "severity": "suggestion"
                })
            
            if func_data["args"] > 5:
                results["suggestions"].append({
                    "line": func_data["line"],
                    "message": f"Function '{func_name}' has too many parameters ({func_data['args']}). Consider using a configuration object.",
                    "severity": "suggestion"
                })
        
        # Add issues as suggestions
        for issue in complexity_results.get("issues", []):
            severity = "error" if issue["type"] in ["bare_except"] else "warning"
            results["suggestions"].append({
                "line": issue["line"],
                "message": issue["message"],
                "severity": severity
            })
        
        return results
    
    @staticmethod
    def _detect_language(filename: str) -> str:
        """Detect the programming language based on file extension."""
        if filename.endswith(('.py', '.pyw')):
            return 'python'
        elif filename.endswith(('.js', '.jsx')):
            return 'javascript'
        elif filename.endswith(('.ts', '.tsx')):
            return 'typescript'
        elif filename.endswith('.html'):
            return 'html'
        elif filename.endswith('.css'):
            return 'css'
        elif filename.endswith(('.c', '.cpp', '.h', '.hpp')):
            return 'c++'
        elif filename.endswith(('.java')):
            return 'java'
        elif filename.endswith(('.go')):
            return 'go'
        else:
            return 'unknown'