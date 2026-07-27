"""Attention Grammar — Gradio front-end.

Wires the three pipeline layers (Algebra → Composition → Spectrogram) into
a simple interface.  Layers 1–2 use LLMs; Layer 3 is deterministic.
"""

import os
import re

import gradio as gr
from dotenv import load_dotenv

from attention_algebra.algebra import AlgebraAnalyst
from attention_algebra.composition import Composer
from attention_algebra.config import (
    DEFAULT_LLAMA_CPP_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    ModelFactory,
)
from attention_algebra.spectrum import SpectrogramReader

load_dotenv()

# --- CONSTANTS -------------------------------------------------------------

PROVIDER_OPENROUTER = "OpenRouter"
PROVIDER_LLAMA_CPP = "llama.cpp (Local)"

# --- HELPERS ---------------------------------------------------------------


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks.  Kept for backward-compat."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _clean_latex_formatting(text: str) -> str:
    """Make LLM output a little more presentable in the MathJax pane."""
    if not text:
        return ""

    text = text.replace("```latex", "").replace("```markdown", "").replace("```", "")
    text = text.replace("[$", "(").replace("$]", ")")

    pattern = r"(?si)(.*?)(?:[\*\#\s]*Intercalation Dynamics\s*:?[\*\#\s]*)(.*)"
    match = re.search(pattern, text)
    if match:
        preamble = match.group(1).strip()
        equation_raw = match.group(2).strip()
        equation_raw = equation_raw.replace("**", "").replace("*", "")
        if not (equation_raw.startswith("$$") or equation_raw.startswith("\\[")):
            equation_section = f"$$\n{equation_raw}\n$$"
        else:
            equation_section = equation_raw
        return f"{preamble}\n\n**Intercalation Dynamics:**\n\n{equation_section}"

    return re.sub(r"`(\$+.*?\$+)`", r"\1", text)


def _provider_key(provider: str) -> str:
    if provider == PROVIDER_OPENROUTER:
        return "openrouter"
    return "llama.cpp"


def _resolve_openrouter_key(api_key_input: str) -> str:
    """Prefer the UI value, then fall back to the environment."""
    return (api_key_input or "").strip() or (os.getenv("OPENROUTER_API_KEY") or "").strip()


def _apply_openrouter_key(api_key_input: str) -> str:
    """Set ``OPENROUTER_API_KEY`` for this request and return the resolved key."""
    key = _resolve_openrouter_key(api_key_input)
    if key:
        os.environ["OPENROUTER_API_KEY"] = key
    return key


def _validate_provider(provider: str, openrouter_api_key: str = "") -> str | None:
    """Return an error message when credentials are missing, else ``None``."""
    if provider == PROVIDER_OPENROUTER and not _resolve_openrouter_key(openrouter_api_key):
        return (
            "Error: OpenRouter API key is required. "
            "Enter it in the field above or set OPENROUTER_API_KEY in a .env file."
        )
    return None


def _toggle_provider_fields(provider: str):
    """Update credential and model fields when the provider changes."""
    is_openrouter = provider == PROVIDER_OPENROUTER
    if is_openrouter:
        model_update = gr.update(
            label="OpenRouter Model Slug",
            placeholder="e.g. google/gemini-2.5-flash, anthropic/claude-sonnet-4",
            value=DEFAULT_OPENROUTER_MODEL,
        )
    else:
        model_update = gr.update(
            label="llama.cpp Model Name",
            placeholder="local (must match the model loaded in llama-server)",
            value=DEFAULT_LLAMA_CPP_MODEL,
        )
    return gr.update(visible=is_openrouter), model_update


# --- MODEL CACHE -----------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_SPECTROGRAM_READER = SpectrogramReader()


def _get_analyst(model_name: str, provider: str, credential: str) -> AlgebraAnalyst:
    key = (model_name, provider, "algebra", credential)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = AlgebraAnalyst(
            model_name=model_name, provider=_provider_key(provider)
        )
    return _MODEL_CACHE[key]


def _get_composer(model_name: str, provider: str, credential: str) -> Composer:
    key = (model_name, provider, "composer", credential)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = Composer(
            model_name=model_name, provider=_provider_key(provider)
        )
    return _MODEL_CACHE[key]


# --- PIPELINE --------------------------------------------------------------


def process_pattern(
    text: str,
    model_name: str,
    provider: str,
    openrouter_api_key: str = "",
):
    """Run the three-layer pipeline and return ``(algebra, math, image, report)``."""
    empty_img = None

    if not text or not text.strip():
        return "Please enter text.", "", empty_img, ""

    model_name = (model_name or "").strip()
    if not model_name:
        return "Error: enter a model slug (e.g. google/gemini-2.5-flash).", "", empty_img, ""

    provider_error = _validate_provider(provider, openrouter_api_key)
    if provider_error:
        return provider_error, "", empty_img, ""

    credential = ""
    if provider == PROVIDER_OPENROUTER:
        credential = _apply_openrouter_key(openrouter_api_key)

    algebraic_expr = ""
    math_report = ""
    composition: dict = {}

    # Layer 1 — Algebra
    try:
        print(f"--- L1: Algebra ({provider}/{model_name}) ---")
        analyst = _get_analyst(model_name, provider, credential)
        algebraic_expr = _strip_think_tags(analyst.analyze(text))
    except Exception as exc:  # noqa: BLE001 — surface the message verbatim
        return f"Error L1: {exc}", "", empty_img, ""

    # Layer 2 — Composition
    try:
        print("--- L2: Composition ---")
        composer = _get_composer(model_name, provider, credential)
        composition = composer.compose(algebraic_expr)

        if isinstance(composition, dict):
            raw_report = composer.format_latex_report(composition)
            math_report = _clean_latex_formatting(raw_report)
        else:
            math_report = f"Error parsing JSON: {composition}"
            composition = {}
    except Exception as exc:  # noqa: BLE001
        return algebraic_expr, f"Error L2: {exc}", empty_img, ""

    # Layer 3 — Spectrogram (deterministic)
    try:
        print("--- L3: Spectrogram ---")
        image, spectrum_report = _SPECTROGRAM_READER.read(composition)
    except Exception as exc:  # noqa: BLE001
        return algebraic_expr, math_report, empty_img, f"Error L3: {exc}"

    return algebraic_expr, math_report, image, spectrum_report


# --- UI --------------------------------------------------------------------


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

#math_output * { color: #000000 !important; }

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


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks app."""
    with gr.Blocks(title="Attention Grammar Engine") as demo:
        gr.HTML(custom_css)

        with gr.Column(elem_classes=["container"]):
            gr.Markdown("# Attention Grammar: Cognitive Transpiler")

            with gr.Row():
                txt_input = gr.Textbox(label="Context", lines=4)
                with gr.Column():
                    provider = gr.Radio(
                        [PROVIDER_OPENROUTER, PROVIDER_LLAMA_CPP],
                        value=PROVIDER_OPENROUTER,
                        label="Provider",
                    )
                    openrouter_api_key = gr.Textbox(
                        label="OpenRouter API Key",
                        placeholder="sk-or-v1-... (or set OPENROUTER_API_KEY in .env)",
                        type="password",
                        value=os.getenv("OPENROUTER_API_KEY", ""),
                    )
                    model = gr.Textbox(
                        label="OpenRouter Model Slug",
                        placeholder="e.g. google/gemini-2.5-flash, anthropic/claude-sonnet-4",
                        value=DEFAULT_OPENROUTER_MODEL,
                    )
                    btn = gr.Button("Analyze", variant="primary")

            gr.Markdown("---")

            gr.Markdown("### Layer 1: Algebra")
            out1 = gr.Textbox(label="Algebra", elem_id="expr_output", show_label=False)

            gr.Markdown("### Layer 2: Harmonic Schedule")
            out2 = gr.Markdown(
                elem_id="math_output",
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {"left": "\\[", "right": "\\]", "display": True},
                    {"left": "\\(", "right": "\\)", "display": False},
                ],
            )

            gr.Markdown("### Layer 3: Spectrogram")
            out3 = gr.Image(label="Cognitive Spectrum", type="numpy")
            out4 = gr.Markdown(label="Spectrum Reading")

        provider.change(
            _toggle_provider_fields,
            inputs=[provider],
            outputs=[openrouter_api_key, model],
        )
        btn.click(
            process_pattern,
            inputs=[txt_input, model, provider, openrouter_api_key],
            outputs=[out1, out2, out3, out4],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()


__all__ = [
    "ModelFactory",
    "build_ui",
    "process_pattern",
]