import git
import os
import ast
import subprocess
import shutil
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import torch
import google.generativeai as genai
import re
import numpy as np

def gemini_generate_text(prompt):
    """
    Calls the Gemini API to generate text based on a prompt.
    """
    # Check if API key is in environment variables
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 256
        }
    )
    
    # Handle potential errors in response
    if hasattr(response, "text"):
        return response.text
    elif hasattr(response, "parts"):
        return response.parts[0].text
    else:
        # Access the content more explicitly for newer Gemini API versions
        return response.candidates[0].content.parts[0].text

def clone_repo(repo_url):
    """Clones the GitHub repository from the given URL."""
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    if not os.path.exists(repo_name):
        git.Repo.clone_from(repo_url, repo_name)
    return repo_name

def extract_python_dependencies(repo_path):
    """
    Extracts Python module names and their import dependencies from the repository.
    Returns a set of modules and a list of import relationships.
    """
    modules = set()
    edges = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                module_name = os.path.relpath(file_path, repo_path).replace('/', '.').replace('.py', '')
                modules.add(module_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                edges.append((module_name, alias.name))
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                edges.append((module_name, node.module))
                except SyntaxError:
                    continue
                except UnicodeDecodeError:
                    # Skip files that can't be decoded
                    continue
    return modules, edges

def generate_dot_with_gemini(modules, edges):
    """
    Uses Gemini API to generate a DOT language specification for the dependency graph.
    """
    # Limit the number of modules and edges to prevent overwhelming Gemini
    MAX_MODULES = 30
    MAX_EDGES = 50
    
    # Sample modules and edges if there are too many
    sample_modules = list(modules)[:MAX_MODULES]
    sample_edges = edges[:MAX_EDGES]
    
    module_list = ', '.join(sample_modules)
    edge_list = ', '.join([f"{importer} imports {imported}" for importer, imported in sample_edges])
    
    prompt = f"""
Generate a DOT language specification for a directed graph based on the following modules and imports.

For example, if the modules are A, B, C and the imports are A imports B, A imports C, the DOT specification should be:
digraph G {{
  A -> B;
  A -> C;
}}

Now, for the actual data:
Modules: {module_list}
Imports: {edge_list}

Please generate ONLY the DOT specification without any markdown formatting, explanation or backticks.
Make node labels very readable with fontsize=16 and with proper spacing between nodes.
"""
    try:
        dot_spec = gemini_generate_text(prompt)
        
        # Clean the response - remove markdown code block markers and backticks
        dot_spec = dot_spec.replace("```dot", "").replace("```", "").strip()
        
        # Remove any special characters that might cause syntax errors
        dot_spec = re.sub(r'[^\w\s\-\>\{\}\[\]\(\)\;\=\,\"\.\:]', '', dot_spec)
        
        # Ensure it starts with "digraph"
        if not dot_spec.startswith("digraph"):
            dot_spec = f"digraph G {{\n"
            # Add edges manually from our data
            for importer, imported in sample_edges:
                # Sanitize node names for DOT syntax
                importer_safe = importer.replace(".", "_").replace("-", "_")
                imported_safe = imported.replace(".", "_").replace("-", "_")
                dot_spec += f'  "{importer_safe}" -> "{imported_safe}";\n'
            dot_spec += "}"
        
        # Add global attributes to make text more readable
        if "graph [" not in dot_spec:
            dot_spec = dot_spec.replace("digraph G {", """digraph G {
  graph [fontsize=16, overlap=false, splines=true];
  node [fontsize=16, shape=box, style=filled, fillcolor=lightblue];
  edge [fontsize=14];
""")
        
        return dot_spec
    except Exception as e:
        print(f"Error generating DOT with Gemini: {e}")
        # Fallback to a simple DOT specification
        dot_spec = """digraph G {
  graph [fontsize=16, overlap=false, splines=true];
  node [fontsize=16, shape=box, style=filled, fillcolor=lightblue];
  edge [fontsize=14];
"""
        for importer, imported in sample_edges:
            # Sanitize node names for DOT syntax
            importer_safe = importer.replace(".", "_").replace("-", "_")
            imported_safe = imported.replace(".", "_").replace("-", "_")
            dot_spec += f'  "{importer_safe}" -> "{imported_safe}";\n'
        dot_spec += "}"
        return dot_spec

def render_dot_to_image(dot_file, output_image):
    """Renders the DOT file into a high-resolution PNG image using Graphviz."""
    # Check if Graphviz is installed
    if not shutil.which("dot"):
        raise RuntimeError("Graphviz is not installed. Please install it with: sudo apt-get install graphviz")
    
    try:
        # Check that the DOT file exists and is valid
        if not os.path.exists(dot_file):
            raise FileNotFoundError(f"DOT file not found: {dot_file}")
            
        with open(dot_file, 'r') as f:
            dot_content = f.read()
            
        # Validate dot syntax by checking for basic structure
        if not (dot_content.startswith("digraph") and "{" in dot_content and "}" in dot_content):
            # Try to fix the DOT file
            with open(dot_file, 'w') as f:
                f.write("""digraph G {
  graph [fontsize=16, overlap=false, splines=true];
  node [fontsize=16, shape=box, style=filled, fillcolor=lightblue];
  edge [fontsize=14];
  A -> B;
}""")
            print("Warning: Invalid DOT syntax. Created a simple graph instead.")
        
        # Run the dot command with high DPI to ensure text is clear
        result = subprocess.run(["dot", "-Tpng", dot_file, "-o", output_image, "-Gdpi=300"], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: graphviz error: {result.stderr}")
            # Create a simple image as fallback
            img = Image.new('RGB', (768, 768), color='white')
            img.save(output_image)
            print(f"Created a blank image as fallback at {output_image}")
        else:
            print(f"Successfully created graph image at {output_image}")
    except Exception as e:
        print(f"Error rendering DOT: {e}")
        # Create a simple image as fallback
        img = Image.new('RGB', (768, 768), color='white')
        img.save(output_image)
        print(f"Created a blank image as fallback at {output_image}")

def artistic_transform(image_path, prompt="A futuristic neon network graph with glowing connections and nodes, ultra-detailed, maintaining all text labels clearly visible"):
    """
    Transforms the initial diagram into an artistic image using Stable Diffusion
    while preserving the text and structure of the original graph.
    """
    try:
        # Ensure image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
            
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Open and resize the image
        init_image = Image.open(image_path).convert("RGB").resize((768, 768))
        
        # Generate the artistic image with lower strength to preserve original content
        image = pipe(
            prompt=prompt, 
            image=init_image, 
            strength=0.5,  # Lower strength to preserve more of the original
            guidance_scale=7.5,
            num_inference_steps=30
        ).images[0]
        
        # Blend the original and generated images to preserve text
        enhanced_image = blend_images_for_text_preservation(init_image, image)
        return enhanced_image
    except Exception as e:
        print(f"Error in artistic transform: {e}")
        # Return the original image as fallback
        try:
            return Image.open(image_path)
        except:
            # Create a default image if even that fails
            img = Image.new('RGB', (768, 768), color='white')
            return img

def blend_images_for_text_preservation(original_image, artistic_image, blend_factor=0.3):
    """
    Blends the original graph image with the artistic version to preserve text.
    
    Args:
        original_image: The original graph image with readable text
        artistic_image: The artistically transformed image
        blend_factor: How much of the original to blend in (0.0-1.0)
    
    Returns:
        A blended image with enhanced text readability
    """
    # Convert images to numpy arrays
    orig_np = np.array(original_image)
    art_np = np.array(artistic_image)
    
    # Create a mask that emphasizes areas with text (high contrast edges)
    gray_orig = np.array(original_image.convert('L'))
    edges = np.array(original_image.convert('L').filter(ImageFilter.FIND_EDGES))
    
    # Dilate the edges to include surrounding areas
    from scipy.ndimage import binary_dilation
    edges = binary_dilation(edges > 50, iterations=3)
    
    # Create the mask as a 3-channel array
    mask = np.stack([edges, edges, edges], axis=2).astype(float)
    
    # Blend images: where mask is 1, use more of original; where 0, use more of artistic
    blend_weight = np.clip(mask * (blend_factor + 0.2), blend_factor, 0.9)
    blended = orig_np * blend_weight + art_np * (1 - blend_weight)
    
    # Convert back to PIL Image
    result_image = Image.fromarray(blended.astype(np.uint8))
    
    # Apply some enhancements to make the text pop
    enhancer = ImageEnhance.Contrast(result_image)
    result_image = enhancer.enhance(1.2)
    
    return result_image

def generate_artistic_graph(repo_url):
    """Main function to generate an artistic graph from a GitHub repository."""
    try:
        # Clone the repository
        repo_path = clone_repo(repo_url)
        print(f"Repository cloned to {repo_path}")

        # Extract dependencies
        modules, edges = extract_python_dependencies(repo_path)
        if not modules:
            print("No Python modules found in the repository.")
            return
        
        print(f"Found {len(modules)} modules and {len(edges)} dependencies")

        # Generate DOT specification using Gemini
        dot_spec = generate_dot_with_gemini(modules, edges)
        dot_file = "graph.dot"
        with open(dot_file, "w") as f:
            f.write(dot_spec)
        print(f"DOT specification saved to {dot_file}")

        # Render the initial diagram
        initial_image_path = "initial_diagram.png"
        render_dot_to_image(dot_file, initial_image_path)
        
        # Verify the initial image was created
        if not os.path.exists(initial_image_path):
            print(f"Initial diagram not created at {initial_image_path}")
            # Create a fallback image
            img = Image.new('RGB', (768, 768), color='white')
            img.save(initial_image_path)
            print(f"Created a blank image as fallback at {initial_image_path}")
        
        # Transform into an artistic image
        print("Generating artistic image (this may take a few minutes)...")
        artistic_image = artistic_transform(initial_image_path)
        output_path = "artistic_graph.png"
        artistic_image.save(output_path)
        print(f"Artistic graph saved as '{output_path}'")
    except Exception as e:
        print(f"Error generating artistic graph: {e}")

# Example usage
if __name__ == "__main__":
    # Install the recommended packages
    try:
        subprocess.check_call(["pip", "install", "accelerate scipy"])
        print("Successfully installed required packages")
    except:
        print("Could not install recommended packages. Some features may be limited.")
    
    repo_url = "https://github.com/microsoft/CodeBERT"  # Replace with your GitHub repo URL
    generate_artistic_graph(repo_url)