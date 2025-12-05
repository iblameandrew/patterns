from langchain_core.prompts.prompt import PromptTemplate
from .config import ModelFactory

ALGEBRA_SYSTEM_PROMPT = f"""
You are an expert in algebraic computational modelling.

Your objective is to deconstruct natural language into high-fidelity, complex algebraic "molecules" representing cognitive dynamics.

### CORE ELEMENTS (Functions):
- **Perception **: `Si` (Introverted Sensing), `Ni` (Introverted Intuition), `Se` (Extraverted Sensing), `Ne` (Extraverted Intuition).
- **Judgment**: `Ti` (Introverted Thinking), `Fi` (Introverted Feeling), `Te` (Extraverted Thinking), `Fe` (Extraverted Feeling).

### PHYSICS & COEFFICIENTS:
1. **Mass (Coefficient inside `()`, e.g., `5Si`)**: Represents **Intensity**, weight, or importance of a specific feeling/function. Range 1-10.
   - *Example*: "Overwhelming rage" = `9Se`; "Mild annoyance" = `2Se`.
2. **Acceleration (Coefficient outside `()`, e.g., `40(...)`)**: Represents **Frequency**, speed, repetition, or manic energy.
   - *Example*: "Racing thoughts over and over" = `50(Ti)`; "A slow, heavy realization" = `5(Ni)`.

### OPERATOR SYNTAX & RULES:
1. **Orbit `~` (Structuring)**: A Judgement Unit (Tx/Fx) creates order around a Perception Unit (Sx/Nx).
   - *Usage*: "Exploring ideas (Ne) to build a system (Ti)" -> `(Ne ~ Ti)`.
2. **Opposition `oo` (Conflict)**: Two functions of the same domain but opposite charge clash (extraversion vs introversion). 
   - **CRITICAL RULE**: Opposition `oo` **ALWAYS** generates a Drag `→`. The winner (higher mass) carries the drag to the opposite domain.
   - *Formula*: `High_Mass - Low_Mass = Result -> Drag`.
   - *Example*: `7Se oo 3Si` results in `4Se -> Ni`.
3. **Drag `→` (Transformation)**: The retroactive pull resulting from an orbital interaction
4. **Switching `|` (Oscillation)**: Flipping between two functionals that do not interact (Si with Ni, Ne with Se, Te with Fe and Fi with Ti)
5. **Grouping `()`**: Use nested parentheses to show the order of operations for complex psychological states.

### COMPLEX MOLECULE EXAMPLES:

- *Input*: "I feel a deep, heavy internal value conflict that is slowly forcing me to organize my external environment just to cope."
  *Logic*: Deep internal conflict is `Fi oo Fe`. The "forcing" implies a drag into Te (external order).
  *Output*: `10((Fi oo Fe) -> Te) ~ Si`

- *Input*: "My mind is racing with a million possibilities, echoing around, but they are all tethered to a need for social harmony."
  *Logic*: Racing possibilities = High Accel Ne. Social Harmony = Fe.
  *Output*: `100(Ne ~ Fe)`

### INSTRUCTIONS:
Analyze the text below. Look for layers of motivation, conflict, and resulting behavior. Construct a complex algebraic expression that captures the *nuance*, *intensity* (mass), and *speed* (acceleration) of the psyche described.

**Output ONLY the final algebraic expression string.**

Text: {{text}}
"""

class AlgebraAnalyst:
    def __init__(self, model_name="llama3"):
        self.llm = ModelFactory.get_model(model_name=model_name, temperature=0.8)
        self.prompt = PromptTemplate(
            template=ALGEBRA_SYSTEM_PROMPT,
            input_variables=["text"]
        )
        self.chain = self.prompt | self.llm

    def analyze(self, text):
        return self.chain.invoke({"text":text}).content