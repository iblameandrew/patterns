import json
from langchain_core.prompts.prompt import PromptTemplate
from .config import ModelFactory

CODING_SYSTEM_PROMPT = """
You are an expert Reinforcement Learning Engineer and Computational Psychologist.
Your task is to generate a Python file implementing a dynamic `AlgebraAgent` class based on a provided "Harmonic Schedule" JSON.

### CONTEXT:
You are extending a library called `capo.py`. 
- **Base Class**: `PPOAgent`
- **Config Class**: `ObjectiveConfig(name, enabled, weight, mode, metadata)`
- **Available Objectives**: `ExplorationObjective`, `ExploitationObjective`, `GatheringObjective`, `ExtrapolationObjective`, `InterpolationObjective`, `ContrastObjective`, `IntegrationObjective`, `SelectionObjective`.

### INPUT DATA (The Harmonic Schedule):
You will receive a JSON object containing:
1. `score`: A list of objectives. Each has a `symbol` (class name), `mass` (base weight), and `voice` (name).
2. `schedule_logic`: The dynamics type ("Orbital", "Drag", "Stochastic Switching", "Adversarial", "Linear").
3. `global_frequency`: A float representing acceleration/speed.

### INSTRUCTIONS:

1. **Imports**: Import `math`, `random`, and the necessary classes from `capo`.
2. **Class Definition**: Define `class AlgebraAgent(PPOAgent):`.

3. **__init__(self, config)**:
   - Store `self.global_frequency` from the JSON input.
   - Initialize `self.step_count = 0`.
   - Create a dictionary `self.objective_configs`.
   - Iterate through the `score` list in the JSON:
     - Create an `ObjectiveConfig` for each item.
     - `name` = item['voice']
     - `weight` = item['mass']
     - `metadata` = {{"role": item['voice'], "math_symbol": item['symbol']}} 
   - Update `config` with `self.objective_configs` and call `super().__init__(config)`.

4. **get_action(self, state)**:
   - Override this method.
   - Increment `self.step_count`.
   - **Implement the Dynamics** based on `schedule_logic`:
     - **"Orbital"**: 
       - Use `math.sin(self.step_count * self.global_frequency)` to modulate weights. 
       - If multiple objectives exist, offset their phases (e.g., `sin` for one, `cos` for another).
     - **"Drag"**: 
       - Apply exponential decay to the first objective: `weight * exp(-self.step_count * freq)`.
       - Apply growth to the second (if exists): `weight * (1 - exp(...))`.
     - **"Stochastic Switching"**: 
       - Use `random.random() < frequency` to toggle `obj.enabled` True/False.
     - **"Adversarial"**: 
       - Flip the weight of the second objective to be negative or distinct from the first.
     - **"Linear"**: Keep weights constant.
   - *Crucial*: You must update `self.objective_configs[name].weight` (or `.enabled`) dynamically before calling `super().get_action(state)`.
   - Return `super().get_action(state)`.

5. **Execution Block**:
   - Add `if __name__ == "__main__":`
   - Define a standard `config` dict (env dims, hyperparameters).
   - Instantiate `AlgebraAgent(config)`.
   - Print a message confirming the pattern loaded.

### OUTPUT RULES:
- Output **ONLY** valid Python code.
- Do not use Markdown backticks.
- Ensure logic handles cases where `score` has 1 or multiple items.

### INPUT JSON:
{schedule_json}
"""

class CodeGenerator:
    def __init__(self, model_name="llama3"):
        # Temperature 0.2 allows for creative coding solutions while sticking to the strict API
        self.llm = ModelFactory.get_model(model_name=model_name, temperature=0.8)
        self.prompt = PromptTemplate(
            template=CODING_SYSTEM_PROMPT,
            input_variables=["schedule_json"]
        )
        self.chain = self.prompt | self.llm

    def generate_code(self, composition):
        """
        Generates the PyTorch code using an LLM Chain based on the Layer 2 composition JSON.
        """
        # Serialize the complex dictionary to a JSON string
        schedule_json = json.dumps(composition, indent=2)
        
        # Run the chain
        raw_code = self.chain.invoke({"schedule_json": schedule_json}).content
        
        # Clean up output
        return raw_code 

    def _clean_output(self, text):
        """Removes markdown formatting if present."""
        text = text.strip()
        if text.startswith("```python"):
            text = text[9:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()