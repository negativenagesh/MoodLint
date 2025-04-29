import sys
import json

def check_dependencies():
    missing_deps = []
    try:
        import langchain_core
    except ImportError:
        missing_deps.append("langchain_core")
    
    try:
        import langchain
    except ImportError:
        missing_deps.append("langchain")
    
    try:
        import langgraph
    except ImportError:
        missing_deps.append("langgraph")
    
    try:
        import google.generativeai
    except ImportError:
        missing_deps.append("google-generativeai")
    
    return missing_deps

if __name__ == "__main__":
    missing = check_dependencies()
    if missing:
        install_cmd = f"pip install {' '.join(missing)}"
        print(json.dumps({
            "status": "missing_dependencies",
            "missing": missing,
            "install_command": install_cmd
        }), flush=True)
        sys.exit(1)
    else:
        print(json.dumps({"status": "dependencies_ok"}), flush=True)
        sys.exit(0)