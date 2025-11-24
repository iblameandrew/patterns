import gradio as gr
import os
import subprocess
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from patterns.algebra import AlgebraAnalyst
from patterns.composition import Composer
from patterns.code import CodeGenerator

# --- CONSTANTS ---
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-3-pro-preview", "gemini-2.5-pro"]

def strip_think_tags(text: str) -> str:
    """Removes <think> tags."""
    if not text: return ""
    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL)

def clean_latex_formatting(text: str) -> str:
    """
    Cleans LLM output to ensure visibility and proper MathJax rendering.
    """
    if not text: return ""
    
    # 1. Remove markdown code block fences
    text = text.replace("```latex", "").replace("```markdown", "").replace("```", "")
    
    # 2. Fix nested brackets/dollars
    text = text.replace("[$", "(").replace("$]", ")")
    
    # 3. Robustly handle the "Intercalation Dynamics" footer
    # Regex Breakdown:
    # Group 1 (.*?): Capture everything (The Table) before the header
    # Non-Capture Group (?:...): Match the header "Intercalation Dynamics" with loose formatting
    # Group 2 (.*): Capture everything after the header (The Equation)
    pattern = r"(?si)(.*?)(?:[\*\#\s]*Intercalation Dynamics\s*:?[\*\#\s]*)(.*)"
    
    match = re.search(pattern, text)
    if match:
        preamble = match.group(1).strip()
        equation_raw = match.group(2).strip() # <--- FIXED: Was erroneously group(3)
        
        # Clean formatting inside the equation
        equation_raw = equation_raw.replace("**", "").replace("*", "")
        
        # Wrap in $$ if missing
        if not (equation_raw.startswith("$$") or equation_raw.startswith("\\[")):
            equation_section = f"$$\n{equation_raw}\n$$"
        else:
            equation_section = equation_raw
            
        return f"{preamble}\n\n**Intercalation Dynamics:**\n\n{equation_section}"

    # 4. Fallback: simple cleanup if regex didn't match
    text = re.sub(r'`(\$+.*?\$+)`', r'\1', text)
    
    return text

def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1: return ["llama3"] 
        return [line.split()[0] for line in lines[1:]]
    except:
        return ["Ollama Not Installed"]

def update_model_choices(provider):
    if provider == "Google Gemini":
        return gr.Dropdown(choices=GEMINI_MODELS, value=GEMINI_MODELS[0], interactive=True)
    else:
        models = get_ollama_models()
        return gr.Dropdown(choices=models, value=models[0] if models else "llama3", interactive=True)

def process_pattern(text, model_name):
    if not text.strip(): return "Please enter text.", "", ""
    if model_name.lower().startswith("gemini") and not os.getenv("GEMINI_API_KEY"):
        return "Error: GOOGLE_API_KEY missing.", "", ""

    algebraic_expr = ""
    math_report = ""
    pytorch_code = ""

    # Layer 1
    try:
        print(f"--- L1: Algebra ({model_name}) ---")
        analyst = AlgebraAnalyst(model_name=model_name)
        algebraic_expr = strip_think_tags(analyst.analyze(text))
    except Exception as e: return f"Error L1: {e}", "", ""

    # Layer 2
    try:
        print("--- L2: Composition ---")
        composer = Composer(model_name=model_name)
        composition = composer.compose(algebraic_expr)
        
        if isinstance(composition, dict):
            raw = composer.format_latex_report(composition)
            math_report = clean_latex_formatting(raw)
        else:
            math_report = f"Error parsing JSON: {composition}"
            
    except Exception as e: return algebraic_expr, f"Error L2: {e}", ""

    # Layer 3
    try:
        print("--- L3: Code Gen ---")
        coder = CodeGenerator(model_name=model_name)
        pytorch_code = strip_think_tags(coder.generate_code(composition))
    except Exception as e: return algebraic_expr, math_report, f"Error L3: {e}"

    return algebraic_expr, math_report, pytorch_code

# --- CSS: HIGH VISIBILITY ---
custom_css = """
<style>
.container { max-width: 1100px; margin: auto; }
h1 { text-align: center; color: #2d3748; }

#expr_output textarea { 
    font-family: 'Courier New', monospace; 
    font-size: 18px; 
    font-weight: bold; 
    color: #2c5282 !important; 
    background-color: #ebf8ff !important;
}

#math_output {
    background-color: #ffffff !important;
    border: 1px solid #ccc;
    padding: 20px;
    border-radius: 8px;
    --body-text-color: #000000 !important;
    --prose-body: #000000 !important;
}

#math_output * {
    color: #000000 !important;
}

#math_output table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
}
#math_output th, #math_output td {
    border: 1px solid #d1d5db !important;
    padding: 8px;
    color: #000000 !important;
    background-color: #ffffff !important;
}

.MathJax, .mjx-chtml, .mjx-char, .mjx-container {
    color: #000000 !important;
    fill: #000000 !important;
    font-size: 115% !important; 
}

#math_output code {
    background-color: transparent !important;
    color: #000000 !important;
    border: none !important;
    font-family: inherit;
    font-size: 100%;
}
</style>
"""

with gr.Blocks(title="Patterns Engine") as demo:
    
    gr.HTML(custom_css)
    
    with gr.Column(elem_classes=["container"]):
        gr.Markdown("# Patterns: Cognitive Transpiler")
        
        with gr.Row():
            txt_input = gr.Textbox(label="Context", lines=4)
            with gr.Column():
                provider = gr.Radio(["Ollama (Local)", "Google Gemini"], value="Ollama (Local)", label="Provider")
                models = get_ollama_models()
                model = gr.Dropdown(choices=models, value=models[0], label="Model")
                btn = gr.Button("Analyze", variant="primary")

        gr.Markdown("---")
        
        gr.Markdown("### Layer 1: Algebra")
        out1 = gr.Textbox(label="Algebra", elem_id="expr_output", show_label=False)
        
        gr.Markdown("### Layer 2: Harmonic Schedule")
        out2 = gr.Markdown(
            elem_id="math_output",
            latex_delimiters=[
                { "left": "$$", "right": "$$", "display": True },
                { "left": "$", "right": "$", "display": False },
                { "left": "\\[", "right": "\\]", "display": True },
                { "left": "\\(", "right": "\\)", "display": False }
            ]
        )
        
        gr.Markdown("### Layer 3: Code")
        out3 = gr.Code(language="python")

    provider.change(update_model_choices, inputs=[provider], outputs=[model])
    btn.click(process_pattern, inputs=[txt_input, model], outputs=[out1, out2, out3])

if __name__ == "__main__":
    demo.launch()