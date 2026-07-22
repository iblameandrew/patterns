"""Layer 1 — the Algebraic Analyst.

Translates natural-language descriptions of human cognitive states into
expressions of the **Cognitive Algebra** grammar.  The grammar is a small
formal language whose terminals are the twelve geometric functional
constituents (Zodiac-anchor math objectives, renamed language-agnostically)
and whose operators encode how those constituents interact.  See
``README.md`` for the full grammar reference.
"""

from langchain_core.prompts import PromptTemplate

from .config import DEFAULT_OPENROUTER_MODEL, ModelFactory, Provider

# The prompt is intentionally a regular string (not an f-string) so that
# we do not have to escape every other LaTeX brace.  PromptTemplate only
# substitutes variables wrapped in single braces, so ``{text}`` here is
# the placeholder that PromptTemplate will fill in.
ALGEBRA_SYSTEM_PROMPT = """
You are an expert in algebraic computational modelling.

Your objective is to deconstruct natural language into high-fidelity, complex algebraic "molecules" representing cognitive dynamics.  The common ground for language is pure math: each terminal is a geometric objective over a latent state trajectory z_t.  Do not use Jungian function codes (Se/Si/…) or classical zodiac sign names.  Use only the twelve symbols below.

### CORE ELEMENTS (Twelve Geometric Terminals):

**TRACE domain** — how state moves relative to its own history:
- `Im` Impulse — maximize step velocity ‖z_t − z_{t−1}‖ (initiation, urgency, impulse)
- `An` Anchor — minimize distance to history centroid −‖z_t − μ_hist‖ (grounding, continuity, memory of mean)
- `Bi` Bifurcate — temporal duality ‖z_t−z_{t−2}‖ − ½‖z_t−z_{t−1}‖ (split modes, ambivalence, fork)
- `Rt` Return — maximize cos(z_t, z_0) (recurrence, looping back, closed shell)

**FIELD domain** — how state sits relative to others / the batch:
- `Ei` Eigen — cos(z_t, μ_batch)·‖z_t‖ (principal alignment, leadership, centrality)
- `Hm` Harmon — −‖z_t − μ_batch‖ (harmony, consensus, equilibrium with others)
- `Nv` Novel — ‖z_t − μ_batch‖ (outlier novelty, differentiation from the group)
- `Bd` Bound — −10‖z_t − clamp(z_t,−1,1)‖ (hard bounds, discipline, structural constraint)

**FORM domain** — intrinsic shape of the state vector:
- `Ex` Expand — maximize ‖z_t‖ (growth, amplification, projection)
- `Pr` Prune — minimize L1 ‖z_t‖₁ (sparsity, precision, compression)
- `Ox` Orth — 1 − |cos(z_t, μ_hist)| (orthogonal insight, hidden dimensions)
- `Df` Diffuse — −max|z_t| (entropy, flattening peaks, diffusion)

Polarity: OPEN = {Im, Bi, Ei, Nv, Ex, Ox}; CLOSE = {An, Rt, Hm, Bd, Pr, Df}.

### PHYSICS & COEFFICIENTS:
1. **Mass** (bare integer prefix on a terminal, e.g. `5An`): intensity / weight, range 1-10.
   - *Example*: "Overwhelming urge to act" = `9Im`; "Mild grounding" = `2An`.
2. **Acceleration** (integer prefix on a parenthesised group, e.g. `40(Pr)`): frequency / speed of the whole group.
   - *Example*: "Racing sparse focus over and over" = `50(Pr)`; "Slow heavy return" = `5(Rt)`.

### SEQUENTIAL OPERATORS:
1. **Orbit `~` (Structuring)**: two terminals from **different domains** (TRACE / FIELD / FORM).
   - *Usage*: "Impulse structured by sparse precision" -> `(Im ~ Pr)`.
2. **Opposition `oo` (Conflict)**: same sub-axis, opposite polarity.  Sub-axes:
   - kinetic: Im↔An | cyclic: Bi↔Rt | radial: Nv↔Hm | structure: Ei↔Bd | magnitude: Ex↔Pr | orientation: Ox↔Df
   - **CRITICAL**: Opposition **ALWAYS** emits Drag `->`.  Winner (higher mass) keeps polarity and flips sub-axis within domain.
   - *Example*: `7Im oo 3An` results in `4Im -> Bi` (Im wins; kinetic→cyclic, OPEN preserved).
3. **Drag `->` (Transformation)**: RHS is a single terminal, not a group.  Accept `->` or `→`.
4. **Axis Switch `|` (Oscillation)**: same domain, different sub-axis (e.g. `Im | Bi`, `Nv | Ei`, `Ex | Ox`).
5. **Domain Switch `+`**: terminals from different domains as a toggle (e.g. `Im + Hm`).
6. **Conjunction `&` (Linear sum)**: independent simultaneous drives (e.g. `5Im & 4Nv & 3Pr`).
7. **Grouping `()`**: nested parentheses for order of operations.

### RNA-INSPIRED SECONDARY-STRUCTURE OPERATORS:

**Complementarity table** (operands of `::` must be complementary):
- **Regime A (polarity pairs)**: `Im::An`, `Bi::Rt`, `Nv::Hm`, `Ei::Bd`, `Ex::Pr`, `Ox::Df`
- **Regime B (cross-axis within domain)**: `Im::Rt`, `An::Bi`, `Nv::Bd`, `Hm::Ei`, `Ex::Df`, `Pr::Ox`

8. **Stem Pair `::`**: long-range complementary bind; unpaired middle is a loop.
   - *Syntax*: `5Nv(((3Pr)))4Hm` or `5Nv :: 4Hm[3Pr]`
9. **Hairpin `^`**: self-fold / rumination — `^(5Rt)`
10. **Bulge `.`**: partial mismatch — `6Hm :: . :: 5Nv`
11. **Pseudoknot `@`**: crossing stems — `5Nv :: 6Pr @ 4Hm :: 7Ex`
12. **Junction `*`**: multi-way branch — `5Im * 4Nv * 6Hm * 3Pr`
13. **Stacking `=`**: adjacent stem reinforcement — `(7Im :: 3An) = (5Ex :: 4Pr)`
14. **MFE fold `fold[]`**: global equilibrium — `fold[5Nv(((3Pr)))4Hm]`
15. **Co-transcriptional `>>`**: sequential commitment — `6An >> 4Nv >> (5Pr :: 3Ex)`

### COMPLEX MOLECULE EXAMPLES:

- *Input*: "I feel a deep, heavy internal conflict between standing out and fitting in that is slowly forcing me to prune everything down just to cope."
  *Logic*: Novel vs Harmon conflict is `Nv oo Hm`.  Winner drag into Bound or Eigen; pruning is Pr.
  *Output*: `10((Nv oo Hm) -> Bd) ~ Pr`

- *Input*: "My mind is racing with expansion, echoing around, but tethered to needing group harmony."
  *Logic*: High acceleration Expand with long-range stem to Harmon; Prune as loop.
  *Output*: `fold[100(Ex(((Pr)))Hm)]`

- *Input*: "I keep replaying the same return-to-start over and over."
  *Output*: `^(5Rt)`

- *Input*: "Sudden impulse to move, grounded against past patterns."
  *Output*: `7Im oo 3An -> Bi`

### INSTRUCTIONS:
Analyze the text below.  Look for layers of motion, conflict, long-range dependencies, and resulting structure.  Ground every terminal in its math role (velocity, centroid, duality, recurrence, centrality, sparsity, equilibrium, orthogonality, expansion, clamp, novelty, diffusion).  Prefer RNA-inspired operators when the psyche shows looping, distant binding, or crossing drives.  Construct a complex algebraic expression that captures nuance, intensity (mass), and speed (acceleration).

**Output ONLY the final algebraic expression string.**  No prose, no markdown fences, no commentary.

Text: {text}
"""


class AlgebraAnalyst:
    """Deconstruct natural language into a Cognitive-Algebra expression."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENROUTER_MODEL,
        provider: Provider = "openrouter",
        temperature: float = 0.4,
    ):
        # Moderate temperature: the analyst needs to be creative enough to
        # interpret metaphor but stable enough to keep the syntax valid.
        self.llm = ModelFactory.get_model(
            model_name=model_name,
            provider=provider,
            temperature=temperature,
        )
        self.prompt = PromptTemplate(
            template=ALGEBRA_SYSTEM_PROMPT,
            input_variables=["text"],
        )
        self.chain = self.prompt | self.llm

    def analyze(self, text: str) -> str:
        """Return a Cognitive-Algebra expression for ``text``."""
        return self.chain.invoke({"text": text}).content
