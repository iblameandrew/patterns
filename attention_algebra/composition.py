"""Layer 2 — the Harmonic Composer.

Reads the Cognitive-Algebra expression emitted by Layer 1 and maps it
into a **Mathematical Schedule**: a JSON document that pairs every
geometric terminal with a concrete optimisation objective, a weight
(mass), and a global frequency (acceleration).  The schedule is the
machine-readable contract that Layer 3 renders as a cognitive
spectrogram.
"""

import json
import re

from langchain_core.prompts import PromptTemplate

from .config import DEFAULT_OPENROUTER_MODEL, ModelFactory, Provider
from .utils import strip_think_tags

COMPOSER_SYSTEM_PROMPT = """
You are a Mathematical Physicist and Harmonic Composer for a Computational Psychology engine.
Your task is to translate a "Cognitive Algebra" expression into a "Mathematical Schedule" whose objectives map to spectral bands for spectrogram rendering.

The twelve terminals are language-agnostic geometric anchors (math objectives over z_t).  Use only these symbols: Im, An, Bi, Rt, Ei, Pr, Hm, Ox, Ex, Bd, Nv, Df.

### MAPPING LOGIC (Geometric Terminal -> Mathematical Objective):

1. **Functions to Objectives** (formulae are over latent state z_t):
   - **Im Impulse** -> `KineticVelocityObjective` | Math: $\\|z_t - z_{{t-1}}\\|$ (Maximize step velocity)
   - **An Anchor** -> `CentroidStabilityObjective` | Math: $-\\|z_t - \\mu_{{\\mathrm{{hist}}}}\\|$ (Minimize distance to history centroid)
   - **Bi Bifurcate** -> `TemporalDualityObjective` | Math: $\\|z_t - z_{{t-2}}\\| - 0.5\\|z_t - z_{{t-1}}\\|$ (Temporal bimodality)
   - **Rt Return** -> `CyclicRecurrenceObjective` | Math: $\\cos(z_t, z_0)$ (Maximize similarity to origin)
   - **Ei Eigen** -> `RepresentativeCentralityObjective` | Math: $\\cos(z_t, \\mu_{{\\mathrm{{batch}}}}) \\cdot \\|z_t\\|$ (Batch eigen-centrality)
   - **Pr Prune** -> `SparsePrecisionObjective` | Math: $-\\|z_t\\|_1$ (L1 compression)
   - **Hm Harmon** -> `HarmonicEquilibriumObjective` | Math: $-\\|z_t - \\mu_{{\\mathrm{{batch}}}}\\|$ (Minimize distance to batch mean)
   - **Ox Orth** -> `LatentOrthogonalityObjective` | Math: $1 - |\\cos(z_t, \\mu_{{\\mathrm{{hist}}}})|$ (Orthogonal to history surface)
   - **Ex Expand** -> `VectorExpansionObjective` | Math: $\\|z_t\\|$ (Maximize magnitude)
   - **Bd Bound** -> `StructuralConstraintObjective` | Math: $-10\\|z_t - \\mathrm{{clamp}}(z_t, -1, 1)\\|$ (Hard bounds)
   - **Nv Novel** -> `DiversityNoveltyObjective` | Math: $\\|z_t - \\mu_{{\\mathrm{{batch}}}}\\|$ (Maximize distance from batch center)
   - **Df Diffuse** -> `EntropicDiffusionObjective` | Math: $-\\max_i |z_t^{{(i)}}|$ (Flatten peaks / entropy)

2. **Sequential Operators to Scheduling Logic**:
   - **Orbit (`~`)**       -> "Orbital".  Weights oscillate using Sine/Cosine waves.
   - **Drag (`->` or `→`)** -> "Drag".  First function decays exponentially, second grows.
   - **Axis Switch (`|`)** -> "Stochastic Switching".  Same-domain sub-axis alternation.
   - **Domain Switch (`+` across TRACE/FIELD/FORM)** -> "Domain Switching".
   - **Opposition (`oo`)** -> "Adversarial".  Secondary objective has negative weight.
   - **Conjunction (`&` or `+`)** -> "Linear".  Constant weights.

3. **RNA-Inspired Operators to Scheduling Logic**:
   - **Stem Pair (`::` or dot-bracket `A(((loop)))B`)** -> "Cooperative Binding".  Stem weights coupled by distance; loop terms at reduced weight.
   - **Hairpin (`^`)** -> "Feedback Loop".  Decaying self-referential temporal consistency.
   - **Bulge (`.`)** -> "Partial Adversarial".  Stem pair with small negative bulge weight.
   - **Pseudoknot (`@`)** -> "Crossing Constraints".  Constrained adversarial optimisation.
   - **Junction (`*`)** -> "Softmax Junction".  Dynamic softmax reweighting across 3+ objectives.
   - **Stacking (`=`)** -> "Amplified Binding".  Adjacent stems amplified by factor gamma=1.5.
   - **MFE fold (`fold[]`)** -> "Global Equilibrium".  Minimum free-energy structure selection.
   - **Co-transcriptional (`>>`)** -> "Sequential Commitment".  Prefix-locked progressive folding.

4. **Physics (Coefficients)**:
   - **Mass** (integer prefix on terminal, e.g. `5An`): becomes `weight` in score.
   - **Acceleration** (integer prefix on group, e.g. `40(...)`): becomes `global_frequency`.

### TASK:
Analyze the Input Algebra.  Decompose nested/RNA structures into a `nodes` tree.  Set top-level `schedule_logic` to the **outermost** or **dominant** dynamic.  Compute `fold_energy` when `fold[]` is present (sum: stems -min(m_A,m_B), loops +0.5 per loop terminal, pseudoknots +5 per crossing).

**Input Algebra**: {algebra}

### OUTPUT FORMAT (Strict JSON):
{{
    "original_expression": "{algebra}",
    "structure_type": "flat" | "nested",
    "schedule_logic": "<dominant logic from lists above>",
    "global_frequency": <float, default 1.0>,
    "fold_energy": <float or null>,
    "nodes": [
        {{
            "operator": "StemPair" | "Orbit" | "Hairpin" | "Pseudoknot" | "Junction" | "Stacking" | "Fold" | "Sequential" | "Opposition" | "Linear",
            "schedule_logic": "<logic for this node>",
            "terminals": ["Nv", "Hm"],
            "loop": ["Pr"],
            "children": []
        }}
    ],
    "score": [
        {{
            "voice": "Voice 1 (Terminal Name)",
            "symbol": "ObjectiveClassName",
            "mass": <float>,
            "formula": "LaTeX string",
            "description": "Brief physics description",
            "role": "stem" | "loop" | "bulge" | "primary" | "secondary"
        }}
    ],
    "math_narrative": "Short sentence describing the mathematical interaction."
}}
"""

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)

SCHEDULE_EQUATIONS: dict[str, str] = {
    "Orbital": "orbital",
    "Drag": "drag",
    "Stochastic Switching": "stochastic",
    "Domain Switching": "stochastic",
    "Adversarial": "adversarial",
    "Linear": "linear",
    "Cooperative Binding": "cooperative",
    "Feedback Loop": "feedback",
    "Partial Adversarial": "partial_adversarial",
    "Crossing Constraints": "crossing",
    "Softmax Junction": "softmax",
    "Amplified Binding": "amplified",
    "Global Equilibrium": "equilibrium",
    "Sequential Commitment": "sequential",
}


def _extract_json(text: str) -> str:
    """Return the JSON body from an LLM reply, stripping ```json fences."""
    match = _FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _equation_for_logic(logic: str, score: list, freq: float) -> str:
    """Build a LaTeX master equation for a single schedule logic."""
    terms: list[str] = []
    eq_type = SCHEDULE_EQUATIONS.get(logic, "linear")

    if eq_type == "orbital":
        for i, track in enumerate(score):
            trig = "\\sin" if i % 2 == 0 else "\\cos"
            terms.append(
                f"{track['mass']} \\cdot {trig}(\\omega t) "
                f"\\cdot [{track['formula']}]"
            )
        eq = " + ".join(terms)
        return f"$$ J(\\theta) = \\sum_{{t}} ({eq}) $$" if eq else ""

    if eq_type == "drag":
        if not score:
            return ""
        t0 = score[0]
        terms.append(
            f"({t0['mass']} \\cdot e^{{-\\lambda t}}) \\cdot [{t0['formula']}]"
        )
        for track in score[1:]:
            terms.append(
                f"({track['mass']} \\cdot (1 - e^{{-\\lambda t}})) "
                f"\\cdot [{track['formula']}]"
            )
        return f"$$ J(\\theta) = \\int ({' + '.join(terms)}) dt $$"

    if eq_type == "adversarial":
        if not score:
            return ""
        terms.append(f"{score[0]['mass']} \\cdot [{score[0]['formula']}]")
        for track in score[1:]:
            terms.append(f" - {track['mass']} \\cdot [{track['formula']}]")
        return f"$$ J(\\theta) = \\max_\\pi ({''.join(terms)}) $$"

    if eq_type == "cooperative":
        stems = [t for t in score if t.get("role") in (None, "stem", "primary")]
        loops = [t for t in score if t.get("role") == "loop"]
        stem_eq = " + ".join(
            f"\\min(m) \\cdot e^{{-d/\\lambda}} \\cdot [{t['formula']}]" for t in stems
        )
        loop_eq = " + ".join(
            f"(1 - e^{{-d/\\lambda}}) \\cdot [{t['formula']}]" for t in loops
        )
        parts = [p for p in (stem_eq, loop_eq) if p]
        return f"$$ J(\\theta) = {' + '.join(parts)} $$" if parts else ""

    if eq_type == "feedback":
        if score:
            t = score[0]
            return (
                f"$$ J(\\theta) = \\sum_t {t['mass']} \\cdot e^{{-\\alpha t}} "
                f"\\cdot [{t['formula']}] + \\beta \\cdot [{t['formula']}]_{{t-1}} $$"
            )
        return ""

    if eq_type == "partial_adversarial":
        if len(score) >= 2:
            return (
                f"$$ J(\\theta) = {score[0]['mass']} \\cdot [{score[0]['formula']}] "
                f"- \\varepsilon \\cdot [{score[1]['formula']}] $$"
            )
        return ""

    if eq_type == "crossing":
        if score:
            body = " + ".join(f"{t['mass']} \\cdot [{t['formula']}]" for t in score)
            return f"$$ J(\\theta) = \\max_\\pi ({body}) \\quad \\text{{s.t.}} \\; g_1 \\cdot g_2 \\leq \\varepsilon $$"
        return ""

    if eq_type == "softmax":
        if score:
            body = ", ".join(f"m_{{i}} \\cdot [{t['formula']}]" for i, t in enumerate(score))
            return f"$$ w_i(t) = \\mathrm{{softmax}}({body}) $$"
        return ""

    if eq_type == "amplified":
        for track in score:
            terms.append(f"{track['mass']} \\cdot \\gamma \\cdot [{track['formula']}]")
        eq = " + ".join(terms)
        return f"$$ J(\\theta) = {eq} $$" if eq else ""

    if eq_type == "equilibrium":
        return "$$ S^* = \\arg\\min_S \\left( \\sum_\\text{{stems}} (-\\Delta G_\\text{{pair}}) + \\sum_\\text{{loops}} \\Delta G_\\text{{loop}} + \\sum_\\text{{pk}} \\Delta G_\\text{{pk}} \\right) $$"

    if eq_type == "sequential":
        return "$$ w_i(t) = m_i \\cdot \\mathbb{{1}}[i \\leq t_\\text{{commit}}] $$"

    # Linear / stochastic / default
    for track in score:
        terms.append(f"{track['mass']} \\cdot [{track['formula']}]")
    eq = " + ".join(terms)
    return f"$$ J(\\theta) = {eq} $$" if eq else ""


class Composer:
    """Turn a Cognitive-Algebra expression into a Mathematical Schedule."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENROUTER_MODEL,
        provider: Provider = "openrouter",
        temperature: float = 0.2,
    ):
        self.llm = ModelFactory.get_model(
            model_name=model_name,
            provider=provider,
            temperature=temperature,
        )
        self.prompt = PromptTemplate(
            template=COMPOSER_SYSTEM_PROMPT,
            input_variables=["algebra"],
        )
        self.chain = self.prompt | self.llm

    def compose(self, algebra_str: str):
        """Run the LLM chain and return a parsed dict (or an error dict)."""
        raw_output = self.chain.invoke({"algebra": algebra_str}).content
        raw_output = strip_think_tags(raw_output)
        return self._parse_json_safely(raw_output)

    @staticmethod
    def _parse_json_safely(text: str):
        """Parse ``text`` as JSON, returning an error-dict on failure."""
        cleaned = _extract_json(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return {
                "original_expression": "Error parsing",
                "structure_type": "flat",
                "schedule_logic": "Linear",
                "global_frequency": 1.0,
                "fold_energy": None,
                "nodes": [],
                "score": [],
                "math_narrative": f"Error: {exc}",
                "raw_output": cleaned,
            }

    @staticmethod
    def format_latex_report(composition: dict) -> str:
        """Render a Mathematical Schedule as a Markdown/LaTeX report."""
        logic = composition.get("schedule_logic", "Linear")
        narrative = composition.get("math_narrative", "")
        freq = composition.get("global_frequency", 1.0)
        score = composition.get("score", [])
        nodes = composition.get("nodes", [])
        fold_energy = composition.get("fold_energy")
        structure_type = composition.get("structure_type", "flat")

        report = f"### Cognitive Schedule: **{logic}**"
        if structure_type == "nested":
            report += " (nested)"
        report += "\n"
        report += f"*{narrative}* (Global Accel: $\\omega={freq}$)"
        if fold_energy is not None:
            report += f" | Fold Energy: $\\Delta G={fold_energy}$"
        report += "\n\n"

        report += "| Function | Objective Class | Math Form | Mass ($m$) | Role |\n"
        report += "| :--- | :--- | :--- | :---: | :--- |\n"

        for track in score:
            clean_formula = track["formula"].replace("|", "\\|")
            role = track.get("role", "—")
            report += (
                f"| {track['voice']} | `{track['symbol']}` | "
                f"${clean_formula}$ | {track['mass']} | {role} |\n"
            )

        report += "\n**Intercalation Dynamics:**\n"
        report += _equation_for_logic(logic, score, freq)

        if nodes:
            report += "\n\n**Structure Tree:**\n"
            for i, node in enumerate(nodes):
                op = node.get("operator", "?")
                node_logic = node.get("schedule_logic", "")
                terminals = ", ".join(node.get("terminals", []))
                loop = ", ".join(node.get("loop", []))
                report += f"- Node {i + 1}: `{op}` ({node_logic})"
                if terminals:
                    report += f" — terminals: {terminals}"
                if loop:
                    report += f", loop: {loop}"
                report += "\n"

        if not score and "raw_output" in composition:
            report += (
                "\n\n<details><summary>Raw model output</summary>\n\n"
                f"```\n{composition['raw_output']}\n```\n\n</details>"
            )

        return report
