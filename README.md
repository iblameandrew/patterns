
# Patterns: An open source translator to convert reasoning traces, into fundamental cognitive primitives

Glad you found this. This library is meant to allow researchers and practitioners to grab the scapel, and diagnose with rigourus analytical expression the cognitive modallity their models are fundamentally thinking in. To this extent, we propose a clear roadmap to AGI as the satisfaction of a criteria. In the framework of deep psychology, eight modalities for human cognition are proposed distinguished in four dimensions: each having an extraverted and an introverted counterpart. We thus have feeling, thinking, intuition and sensing. If a model is capable of thinking in terms of all 8 modalities while intertwining the modalities in meaningful patterns, we can safely assume that the model is in fact thinking like a human being. Divided into their ambivalences the functions can be derived as following:

- Introverted Feeling (Fi)
- Extraverted Feeling (Fe)
- Introverted Intuition (Ni)
- Extraverted Intuition (Ne)
- Introverted Thinking (Ti)
- Extraverted Thinking (Te)
- Introverted Sensing (Si)
- Extraverted Sensing (Se)


Extensive research and theorycrafting has suggest us over the years, that these functions can be intertwined together into patterns, as if they were chemical compounds. Furthermore each of these functions, has a direct equivalent as a mathematical operation (Fi: Group Mean, Ti: Kolmogorov Complexity Minimization, Te: Regression,...etc). We have found GRPO to only cover a few functionals. This is not enough to achieve general reasoning capabilities as many dimensions of cognition are still missing. For this purpose we introduce *Patterns* as a powerful tool to understand everyday language as an intricate web of simple mathematical operations. This is a fundamental and groundbreaking shift into how we think about language, AI and human psychology; and hopefully an engineered way to architect reasoning modalities, obtaining complete control over the thinking patterns a model has, giving us the capacity of architecting a true personality for an LLM.

## 🏗 Architecture

The system operates in three distinct layers:

1.  **Layer 1: The Algebraic Analyst**
    *   **Input:** Natural language (e.g., "I explore impulsively but feel held back by past regrets").
    *   **Process:** Uses LLMs to parse text into the **Algebra** syntax (e.g., `Se ~ Si`).
    *   **Output:** A valid algebraic expression representing cognitive dynamics.

2.  **Layer 2: The Harmonic Composer**
    *   **Input:** Algebraic expression.
    *   **Process:** Maps cognitive functions (e.g., `Se`, `Ti`) to mathematical objectives (e.g., Entropy Maximization, Contrast). It determines the "schedule" of interaction (Orbital, Drag, Switching).
    *   **Output:** A "Mathematical Partiture" (JSON) describing the physics of the agent.

3.  **Layer 3: The Mechanic (Code Generator)**
    *   **Input:** Mathematical Partiture.
    *   **Process:** Generates a custom `AlgebraAgent` class inheriting the common structure of GRPO.
    *   **Output:** Executable PyTorch code ready to be plugged into PPO 


## 🚀 Installation & Setup

### Prerequisites
*   Python 3.9+
*   [Ollama](https://ollama.com/) (If using Local Mode)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/iblameandrew/patterns.git
    cd patterns
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

**Option A: API Mode (Google Gemini)**
1.  Get an API key from [Google AI Studio](https://aistudio.google.com/).
2.  Set the environment variable:
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    # Or create a .env file
    ```

**Option B: Local Mode (Ollama)**
1.  Install Ollama.
2.  Pull the requisite model (e.g., qwen30b-thinking):
    ```bash
    ollama run qwen3:30b-thinking
    ```

---

## 💻 Usage

Run the Gradio User Interface:

```bash
python -m patterns.app