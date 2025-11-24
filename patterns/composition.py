import json
import re
from langchain_core.prompts.prompt import PromptTemplate
from .config import ModelFactory
COMPOSER_SYSTEM_PROMPT = """
You are a Mathematical Physicist and Harmonic Composer for a Computational Psychology engine.
Your task is to translate a "Cognitive Algebra" expression into a "Mathematical Schedule" of optimization objectives for a Reinforcement Learning agent.

### MAPPING LOGIC (Cognitive -> Mathematical):

1. **Functions to Objectives**:
   - **Se (Extroverted Sensing)** -> `ExplorationObjective` | Math: $\mathcal{{H}}(\pi(a|s))$ (Maximize Entropy)
   - **Si (Introverted Sensing)** -> `GatheringObjective` | Math: $e^{{-||s - \mu||}}$ (Minimize Distance to Centroid)
   - **Ne (Extroverted Intuition)** -> `ExtrapolationObjective` | Math: $e^{{||s - \mu||}}$ (Maximize Distance / Novelty)
   - **Ni (Introverted Intuition)** -> `InterpolationObjective` | Math: $\text{{proj}}_{{\vec{{v}}}}(s)$ (Trajectory Alignment)
   - **Te (Extroverted Thinking)** -> `ExploitationObjective` | Math: $\mathbb{{E}}[V(s)]$ (Maximize Value)
   - **Ti (Introverted Thinking)** -> `ContrastObjective` | Math: $|d(s, a) - d(s, b)|$ (Maximize Discrimination)
   - **Fe (Extroverted Feeling)** -> `IntegrationObjective` | Math: $\mathcal{{H}} + \alpha V(s)$ (Balance Entropy & Value)
   - **Fi (Introverted Feeling)** -> `SelectionObjective` | Math: $e^{{-d(s, s_{{t-1}})}}$ (Temporal Consistency)

2. **Operators to Scheduling Logic**:
   - **Orbit (`~`)** -> Logic: "Orbital". The weights of the functions oscillate using Sine/Cosine waves.
   - **Drag (`→` or `->`)** -> Logic: "Drag". The first function decays exponentially, the second grows.
   - **Switching (`|`)** -> Logic: "Stochastic Switching". Objectives toggle on/off based on probability.
   - **Opposition (`oo`)** -> Logic: "Adversarial". Both objectives are active, but one has a negative weight or contrastive relationship.
   - **Standard (`+`)** -> Logic: "Linear". Constant weights.

3. **Physics (Coefficients)**:
   - **Mass** (number *inside* parens, e.g., `5Si`): Becomes the `weight` of the objective.
   - **Acceleration** (number *outside* parens, e.g., `40(...)`): Becomes the `frequency` or `rate` of the interaction.

### TASK:
Analyze the Input Algebra. Return a JSON object describing the "Score".

**Input Algebra**: {algebra}

### OUTPUT FORMAT (Strict JSON):
{{
    "original_expression": "{algebra}",
    "schedule_logic": "Orbital" | "Drag" | "Stochastic Switching" | "Linear" | "Adversarial",
    "global_frequency": <float from acceleration coefficient, default 1.0>,
    "score": [
        {{
            "voice": "Voice 1 (Function Name)",
            "symbol": "ObjectiveClassName",
            "mass": <float>,
            "formula": "LaTeX string",
            "description": "Brief physics description"
        }},
        ...
    ],
    "math_narrative": "A short sentence describing the mathematical interaction (e.g., 'Entropy maximization oscillating against temporal consistency')."
}}
"""


def strip_think_tags(text: str) -> str:
    """
    Removes <think>...</think> tags and their content from a string.

    Args:
        text: The input string that may contain think tags.

    Returns:
        A new string with all think tags and their inner content removed.
    """
    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL)

class Composer:
    def __init__(self, model_name="llama3"):
        # Temperature 0.1 ensures strict adherence to the JSON format
        self.llm = ModelFactory.get_model(model_name=model_name, temperature=0.8)
        self.prompt = PromptTemplate(
            template=COMPOSER_SYSTEM_PROMPT,
            input_variables=["algebra"]
        )
        print(self.prompt)
        self.chain = self.prompt | self.llm


    def compose(self, algebra_str):
        """
        Runs the LLM chain to generate the JSON composition from the algebra string.
        """
        raw_output = self.chain.invoke({"algebra": algebra_str}).content
        raw_output = strip_think_tags(raw_output) 
        return self._parse_json_safely(raw_output)

    def _parse_json_safely(self, text):
        """
        Cleans LLM output (removes markdown backticks) and parses JSON.
        """
        text = text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Fallback for error handling in UI
            return {
                "original_expression": "Error parsing",
                "schedule_logic": "Linear",
                "score": [],
                "math_narrative": f"Error: {str(e)}",
                "raw_output": text
            }

    def format_latex_report(self, composition):
        """
        Generates a readable Markdown/LaTeX report from the JSON composition.
        """
        logic = composition.get("schedule_logic", "Linear")
        narrative = composition.get("math_narrative", "")
        freq = composition.get("global_frequency", 1.0)
        
        report = f"### Cognitive Schedule: **{logic}**\n"
        report += f"*{narrative}* (Global Accel: $\\omega={freq}$)\n\n"
        
        report += "| Function | Objective Class | Math Form | Mass ($m$) |\n"
        report += "| :--- | :--- | :--- | :---: |\n"
        
        score = composition.get("score", [])
        for track in score:
            # Escape pipes in latex for markdown table compatibility
            clean_formula = track['formula'].replace("|", "\\|")
            report += f"| {track['voice']} | `{track['symbol']}` | ${clean_formula}$ | {track['mass']} |\n"
        
        report += "\n**Intercalation Dynamics:**\n"
        
        # Generate the master equation based on logic
        terms = []
        if logic == "Orbital":
            for i, track in enumerate(score):
                trig = "\\sin" if i % 2 == 0 else "\\cos"
                terms.append(f"{track['mass']} \\cdot {trig}(\\omega t) \\cdot [{track['formula']}]")
            eq = " + ".join(terms)
            report += f"$$ J(\\theta) = \\sum_{{t}} ({eq}) $$"
            
        elif logic == "Drag":
            # Logic: First term decays, others grow (or static)
            if len(score) > 0:
                t0 = score[0]
                terms.append(f"({t0['mass']} \\cdot e^{{-\\lambda t}}) \\cdot [{t0['formula']}]")
                for track in score[1:]:
                    terms.append(f"({track['mass']} \\cdot (1-e^{{-\\lambda t}})) \\cdot [{track['formula']}]")
            eq = " + ".join(terms)
            report += f"$$ J(\\theta) = \\int ({eq}) dt $$"
            
        elif logic == "Adversarial":
            # Logic: Subtract secondary terms from primary
            if len(score) > 0:
                terms.append(f"{score[0]['mass']} \\cdot [{score[0]['formula']}]")
                for track in score[1:]:
                    terms.append(f" - {track['mass']} \\cdot [{track['formula']}]")
            eq = "".join(terms)
            report += f"$$ J(\\theta) = \\max_\\pi ({eq}) $$"
            
        else: # Linear/Default
            for track in score:
                terms.append(f"{track['mass']} \\cdot [{track['formula']}]")
            eq = " + ".join(terms)
            report += f"$$ J(\\theta) = {eq} $$"
            
        return report