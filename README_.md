# Attention Grammar — A Cognitive Grammar for Language

> A formal grammar that lifts natural language into a higher-dimensional
> space of *functional constituents*, and compiles the result into
> cognitive spectrograms that read a speaker's functional state like a
> sound reads on a spectrum.

<img width="1173" height="642" alt="image" src="https://github.com/user-attachments/assets/4e0e3068-5e69-47bd-a4b9-0a6e894ab56d" />


## The Idea

Every sentence a person or a model produces is a *projection* of a
high-dimensional cognitive state down to a 1-D string of tokens. The
projection is lossy: two sentences that look nothing alike can be
driven by the same internal state, and a single sentence can be the
shadow of many overlapping states.

**Attention Grammar** is a *grammar* for the inverse projection. Given a piece
of natural language, it parses the string into an expression in a
small formal language whose terminals are twelve irreducible
*geometric functional constituents* — each one a pure mathematical
objective over a latent trajectory $z_t$ — and whose operators describe
how those constituents interact. The expression is then compiled,
layer by layer, into a mathematical schedule and finally into a
**cognitive spectrogram** — a time–frequency image of the speaker's
functional state.

The twelve terminals are the **Zodiac geometric anchors** (velocity,
centroid, duality, recurrence, centrality, sparsity, equilibrium,
orthogonality, expansion, clamp, novelty, diffusion), renamed into
language-agnostic short codes. Each sign is further **unraveled** as a
Jungian functional-algebra molecule in parentheses — e.g. Capricorn /
`Bd` = `((Te oo Ti) ~ Ni)`, Pisces / `Df` = `(Fi ~ (Ne oo Ni))`.

In one sentence:

> *Attention Grammar is a type system and compiler for language, where the
> types are twelve geometric objectives and the compiled artefact is a
> spectrogram that maps algebraic math equivalences onto a readable
> cognitive spectrum.*

The pipeline lifts language into spectral form:

```
  string  ───parse───▶  ⟨algebra⟩  ───compose───▶  ⟨schedule⟩  ───read───▶  ⟨spectrogram⟩
       (Layer 1)            (Layer 2)                       (Layer 3)
```

---

## Why a Grammar?

A grammar is the right abstraction for three reasons:

1. **Compositionality.** A grammar's production rules are *the same
   no matter how deep you nest*. A clause in a clause in a clause is
   parsed by the same rule that parses a clause. This is exactly the
   property we need: a sentence can have arbitrarily many layers of
   motivation, conflict, and rationalisation, and we want a single
   parser to handle all of them.
2. **Type safety.** Each terminal of the grammar is tagged with its
   *domain* (TRACE vs. FIELD vs. FORM) and its *polarity*
   (OPEN vs. CLOSE). The operators (`~`, `oo`, `→`, `|`,
   `+`) are partial functions that are only well-typed on certain
   combinations. The parser will reject `Im ~ An` (same domain —
   not a legal orbit) and accept `(Im ~ Pr)`.
3. **Realisability.** Every well-typed expression has a *canonical
   mathematical image*: a sum of weighted objective functions whose
   scheduling dynamics follow directly from the operator. The reader
   in Layer 3 is total: every parseable expression renders to a
   spectrogram.

The grammar is compact — twelve terminals, fifteen operators (five
sequential, eight RNA-inspired, two grouping forms), and two numeric
attributes — but it spans the cognitive space densely because the
operators let you weave the terminals into molecules of arbitrary
complexity, including long-range secondary structure.

---

## The Grammar

### Terminals — the twelve geometric constituents

Each sign of the Zodiac rotatory protocol is a differentiable reward
on latent state $z_t$. Attention Grammar keeps that math as geometric
ground, gives every sign a **language-agnostic short name**, and
**unravels** each sign as a Jungian functional-algebra molecule in
parentheses (the cognitive expansion of that sign).

| Symbol | Name | Sign | Functional expansion (unraveled) | Math objective |
| :----: | :--- | :--- | :------------------------------- | :------------- |
| `Im` | Impulse | Aries | `(Se)` | $\|z_t - z_{t-1}\|$ |
| `An` | Anchor | Taurus | `(Si -> Ne)` | $-\|z_t - \mu_{\mathrm{hist}}\|$ |
| `Bi` | Bifurcate | Gemini | `((Ne oo Ni) ~ Ti)` | $\|z_t - z_{t-2}\| - 0.5\|z_t - z_{t-1}\|$ |
| `Rt` | Return | Cancer | `(Si ~ Fe)` | $\cos(z_t, z_0)$ |
| `Ei` | Eigen | Leo | `((Fi oo Fe) -> Te)` | $\cos(z_t, \mu_{\mathrm{batch}}) \cdot \|z_t\|$ |
| `Pr` | Prune | Virgo | `(Si ~ Te oo Ti) \| (Ni ~ Fe oo Fi)` | $-\|z_t\|_1$ |
| `Hm` | Harmon | Libra | `((Fe oo Fi) ~ Ni)` | $-\|z_t - \mu_{\mathrm{batch}}\|$ |
| `Ox` | Orth | Scorpio | `(Ni -> Se)` | $1 - |\cos(z_t, \mu_{\mathrm{hist}})|$ |
| `Ex` | Expand | Sagittarius | `(Se ~ Ti) \| (Ne ~ Fi)` | $\|z_t\|$ |
| `Bd` | Bound | Capricorn | `((Te oo Ti) ~ Ni)` | $-10\|z_t - \mathrm{clamp}(z_t,-1,1)\|$ |
| `Nv` | Novel | Aquarius | `((Ti -> Fe) ~ Ne)` | $\|z_t - \mu_{\mathrm{batch}}\|$ |
| `Df` | Diffuse | Pisces | `(Fi ~ (Ne oo Ni))` | $-\max_i \|z_t^{(i)}\|$ |

**Correspondence rule.** A geometric terminal is a *name* for a sign; the
sign is a *name* for a functional molecule.  Writing `Bd` is shorthand for
Capricorn, which unravels as `((Te oo Ti) ~ Ni)`.  Sequences of geometric
codes (e.g. `Im >> Bd >> Pr`) are sequences of these molecules.

Geometric / schedule labels (for Layer 2–3):

| Symbol | Domain | Sub-axis | Polarity | Zodiac anchor |
| :----: | :----- | :------- | :------- | :------------ |
| `Im` | TRACE | kinetic | OPEN | `KINETIC_VELOCITY` |
| `An` | TRACE | kinetic | CLOSE | `CENTROID_STABILITY` |
| `Bi` | TRACE | cyclic | OPEN | `TEMPORAL_DUALITY` |
| `Rt` | TRACE | cyclic | CLOSE | `CYCLIC_RECURRENCE` |
| `Ei` | FIELD | structure | OPEN | `REPRESENTATIVE_CENTRALITY` |
| `Hm` | FIELD | radial | CLOSE | `HARMONIC_EQUILIBRIUM` |
| `Nv` | FIELD | radial | OPEN | `DIVERSITY_NOVELTY` |
| `Bd` | FIELD | structure | CLOSE | `STRUCTURAL_CONSTRAINT` |
| `Ex` | FORM | magnitude | OPEN | `VECTOR_EXPANSION` |
| `Pr` | FORM | magnitude | CLOSE | `SPARSE_PRECISION` |
| `Ox` | FORM | orientation | OPEN | `LATENT_ORTHOGONALITY` |
| `Df` | FORM | orientation | CLOSE | `ENTROPIC_DIFFUSION` |

**Domains**

| Domain  | Question the speaker is answering                         | Terminals              |
| :------ | :-------------------------------------------------------- | :--------------------- |
| TRACE   | How does my state move relative to *my own history*?      | `Im` `An` `Bi` `Rt`    |
| FIELD   | How does my state sit relative to *others / the batch*?   | `Ei` `Hm` `Nv` `Bd`    |
| FORM    | What is the *intrinsic shape* of my state vector?         | `Ex` `Pr` `Ox` `Df`    |

**Polarity**

* **OPEN** — maximising / expanding / splitting: `Im`, `Bi`, `Ei`, `Nv`, `Ex`, `Ox`
* **CLOSE** — minimising / grounding / compressing: `An`, `Rt`, `Hm`, `Bd`, `Pr`, `Df`

The terminals are a *minimal, complete, and orthogonal basis* for the
geometric cognitive space. No terminal can be expressed as a combination
of the others; each is a distinct topological move on $z_t$.

### Numeric attributes

Every terminal may be preceded by a bare integer that names its
**mass** (intensity, 1–10), e.g. `7Im`. A parenthesised group may
itself be preceded by a bare integer that names its **acceleration**
(frequency, any positive real), e.g. `40(Pr)`.

```
mass       ::=  [1-9][0-9]?    attached to a terminal
accel      ::=  [1-9][0-9]*    attached to a parenthesised expression
```

### Sequential operators

| Symbol | Name           | Semantics                                                                 | Schedule logic        |
| :----: | :------------- | :------------------------------------------------------------------------ | :-------------------- |
| `~`    | Orbit          | Two terminals from **different domains** structure each other.            | Orbital               |
| `oo`   | Opposition     | Same sub-axis, opposite polarity; winner drags to flipped sub-axis.       | Adversarial           |
| `->`   | Drag           | RHS of Opposition only; single function, not a group.                     | Drag                  |
| `\|`   | Axis Switch    | Same domain, different sub-axis alternation.                              | Stochastic switching  |
| `+`    | Domain Switch  | TRACE ↔ FIELD ↔ FORM alternation (when operands cross domains).           | Domain switching      |
| `&`    | Conjunction    | Linear sum of independent functions.                                      | Linear                |
| `( )`  | Grouping       | Override precedence, nest molecules.                                      | —                     |

Opposition sub-axes (must pair opposites):

| Sub-axis    | OPEN | CLOSE |
| :---------- | :--: | :---: |
| kinetic     | `Im` | `An`  |
| cyclic      | `Bi` | `Rt`  |
| radial      | `Nv` | `Hm`  |
| structure   | `Ei` | `Bd`  |
| magnitude   | `Ex` | `Pr`  |
| orientation | `Ox` | `Df`  |

Drag rule: winner keeps polarity, flips sub-axis inside its domain.
Example: `7Im oo 3An` → `4Im -> Bi` (kinetic → cyclic, OPEN preserved).

### RNA-inspired secondary-structure operators

| Symbol     | Name                  | Semantics                                      | Schedule logic          |
| :--------: | :-------------------- | :--------------------------------------------- | :---------------------- |
| `::`       | Stem Pair             | Long-range complementary binding with loop.    | Cooperative Binding     |
| `^`        | Hairpin               | Self-referential feedback fold.                | Feedback Loop           |
| `.`        | Bulge                 | Stem with one partial mismatch.                | Partial Adversarial     |
| `@`        | Pseudoknot            | Crossing non-nested pair constraints.          | Crossing Constraints    |
| `*`        | Junction              | Multi-way branch (3+ drives).                  | Softmax Junction        |
| `=`        | Stacking              | Adjacent stems reinforce (γ=1.5).              | Amplified Binding       |
| `fold[]`   | MFE Fold              | Global minimum free-energy structure.          | Global Equilibrium      |
| `>>`       | Co-transcriptional    | Left-to-right sequential commitment.           | Sequential Commitment   |

**Complementarity table** (for `::` stems):

* **Regime A (polarity pairs):** `Im::An`, `Bi::Rt`, `Nv::Hm`, `Ei::Bd`, `Ex::Pr`, `Ox::Df`
* **Regime B (cross-axis within domain):** `Im::Rt`, `An::Bi`, `Nv::Bd`, `Hm::Ei`, `Ex::Df`, `Pr::Ox`

Dot-bracket sugar: `5Nv(((3Pr)))4Hm` ≡ stem `Nv::Hm` with loop `Pr`.

### Well-typedness

* `~` requires terminals from two different domains.
* `oo` requires same sub-axis and opposite polarity; always emits `->`
  flipping sub-axis while preserving winner polarity.
* `->` is not free-standing; RHS must be a single terminal.
* `|` requires same domain, different sub-axis.
* `+` as domain switch requires operands from different domains.
* `::` requires complementary terminals under Regime A or B.
* `@` requires crossing (non-nested) stem pairings.

Grammar rules live in [`attention_algebra/algebra.py`](attention_algebra/algebra.py)
(prompt-embedded).  Canonical terminal table:
[`attention_algebra/terminals.py`](attention_algebra/terminals.py).
Programmatic validation:
[`attention_algebra/parser.py`](attention_algebra/parser.py).

### Worked examples

| Natural language                                                              | Parse                                             |
| :----------------------------------------------------------------------------- | :------------------------------------------------ |
| "A slow, heavy return to the start."                                          | `5(Rt)`                                            |
| "Racing sparse focus over and over."                                          | `50(Pr)`                                           |
| "Impulse structured by precision."                                            | `(Im ~ Pr)`                                        |
| "I surge forward but feel held back by the past mean."                        | `7Im oo 3An -> Bi`                                 |
| "A deep novel-vs-harmon conflict slowly forces hard bounds and pruning."     | `10((Nv oo Hm) -> Bd) ~ Pr`                        |
| "A million racing expansions tethered to group harmony."                      | `fold[100(Ex(((Pr)))Hm)]`                          |
| "I keep replaying the same return-to-origin."                                 | `^(5Rt)`                                           |
| "Mostly at equilibrium with the group, one novelty still nags."               | `6Hm :: . :: 5Nv`                                  |
| "Everything hits me at once."                                                 | `5Im * 4Nv * 6Hm * 3Pr`                            |

---

## The Functional Space

The twelve terminals are not just labels. Each one names a direction in
a three-domain × two-polarity geometric space, and the cognitive state of
an agent is a *vector* in that space. The grammar is the surface syntax
of that vector algebra:

| Dimension         | Axes / values                         | Basis elements            |
| :---------------- | :------------------------------------ | :------------------------ |
| **Domain**        | TRACE / FIELD / FORM                  | history / batch / shape   |
| **Polarity**      | OPEN / CLOSE                          | maximise / minimise       |
| **Sub-axis**      | kinetic, cyclic, radial, structure, magnitude, orientation | dual pairs |
| **Mass**          | —                                     | bare integer prefix       |
| **Acceleration**  | —                                     | integer group prefix      |

A sentence like

> "I am torn between the part of me that wants to surge ahead and the
> part that wants to hold the running average, but the pull toward
> group harmony keeps dragging me back."

parses to a single point in that space; a paragraph parses to a
*trajectory*; a conversation parses to a *flow*. The Attention Grammar
compiler is a differentiable readout of that flow.

---

## Model Providers

All three layers call an LLM through the **OpenAI SDK** (via
`langchain-openai`).  Two OpenAI-compatible backends are supported:

| Provider | When to use | Required config |
| :------- | :---------- | :-------------- |
| **OpenRouter** | Cloud models (Gemini, Claude, Qwen, Llama, …) | `OPENROUTER_API_KEY` |
| **llama.cpp** | Local inference with `llama-server` | `LLAMA_CPP_BASE_URL`, `LLAMA_CPP_MODEL` |

Pass `provider="openrouter"` or `provider="llama.cpp"` to each layer
class.  The Gradio UI exposes the same choice as a radio button.

---

## The Three Layers

The grammar is realised by a strict, three-stage compiler. Each layer
is a pure function of the previous layer's output, and the layers
themselves are language-model calls constrained by an explicit
production rule. You can swap any layer for your own implementation as
long as it respects the input/output contract.

### Layer 1 — The Algebraic Analyst (`attention_algebra.algebra`)

* **Input:** natural language.
* **Process:** an LLM, prompted with the grammar reference, produces
  a single parse tree flattened to a string.
* **Output:** a *type-checked* expression in the grammar.

```python
from attention_algebra import AlgebraAnalyst
analyst = AlgebraAnalyst(
    model_name="google/gemini-2.5-flash",
    provider="openrouter",
)
expr = analyst.analyze("I am torn between surging ahead and holding the average.")
# '7Im oo 3An -> Bi'
```

### Layer 2 — The Harmonic Composer (`attention_algebra.composition`)

* **Input:** a grammar expression.
* **Process:** an LLM, prompted with the terminal-to-objective table
  below, emits a JSON *Mathematical Schedule* describing the loss
  landscape.
* **Output:** a JSON object with `schedule_logic`, `score`, and
  `math_narrative`.

The canonical mapping from terminal to optimisation objective (Zodiac math):

| Terminal | Objective class                    | Math (LaTeX) | Interpretation |
| :------: | :--------------------------------- | :----------- | :------------- |
| `Im`     | `KineticVelocityObjective`         | $\|z_t - z_{t-1}\|$ | Maximise step velocity. |
| `An`     | `CentroidStabilityObjective`       | $-\|z_t - \mu_{\mathrm{hist}}\|$ | Cluster around history centroid. |
| `Bi`     | `TemporalDualityObjective`         | $\|z_t - z_{t-2}\| - 0.5\|z_t - z_{t-1}\|$ | Temporal bifurcation. |
| `Rt`     | `CyclicRecurrenceObjective`        | $\cos(z_t, z_0)$ | Return to origin. |
| `Ei`     | `RepresentativeCentralityObjective`| $\cos(z_t, \mu_{\mathrm{batch}}) \cdot \|z_t\|$ | Batch eigen-centrality. |
| `Pr`     | `SparsePrecisionObjective`         | $-\|z_t\|_1$ | L1 sparse precision. |
| `Hm`     | `HarmonicEquilibriumObjective`     | $-\|z_t - \mu_{\mathrm{batch}}\|$ | Equilibrium with batch. |
| `Ox`     | `LatentOrthogonalityObjective`     | $1 - |\cos(z_t, \mu_{\mathrm{hist}})|$ | Orthogonal to history. |
| `Ex`     | `VectorExpansionObjective`         | $\|z_t\|$ | Maximise magnitude. |
| `Bd`     | `StructuralConstraintObjective`    | $-10\|z_t - \mathrm{clamp}(z_t,-1,1)\|$ | Hard structural bounds. |
| `Nv`     | `DiversityNoveltyObjective`        | $\|z_t - \mu_{\mathrm{batch}}\|$ | Outlier novelty. |
| `Df`     | `EntropicDiffusionObjective`       | $-\max_i \|z_t^{(i)}\|$ | Flatten peaks / entropy. |

```python
from attention_algebra import Composer
composer = Composer(
    model_name="google/gemini-2.5-flash",
    provider="openrouter",
)
schedule = composer.compose("7Im oo 3An -> Bi")
# {'schedule_logic': 'Adversarial',
#  'global_frequency': 1.0,
#  'score': [{'voice': '...', 'symbol': 'KineticVelocityObjective', ...}, ...],
#  'math_narrative': '...'}
```

### Layer 3 — The Spectrogram Reader (`attention_algebra.spectrum`)

* **Input:** the Mathematical Schedule.
* **Process:** deterministic synthesis — each geometric terminal maps to a
  carrier frequency; schedule logic modulates amplitude envelopes over
  time (`sin`/`cos` for Orbital, `exp(-λt)` for Drag, softmax for
  Junction, etc.); the mixed signal is rendered as a spectrogram with
  terminal band annotations.
* **Output:** a spectrogram image plus a spectrum reading report
  (dominant bands, carrier table, fold energy).

| Terminal | Carrier (Hz) | Spectral region   |
| :------: | -----------: | :---------------- |
| `Im`     | 82.4         | Low TRACE         |
| `An`     | 98.0         | Low TRACE         |
| `Bi`     | 123.5        | Mid TRACE         |
| `Rt`     | 146.8        | Mid TRACE         |
| `Ei`     | 174.6        | Low FIELD         |
| `Pr`     | 196.0        | Low FORM          |
| `Hm`     | 220.0        | Mid FIELD         |
| `Ox`     | 261.6        | Mid FORM          |
| `Ex`     | 293.7        | Mid FORM          |
| `Bd`     | 349.2        | High FIELD        |
| `Nv`     | 392.0        | High FIELD        |
| `Df`     | 440.0        | High FORM         |

```python
from attention_algebra import SpectrogramReader

reader = SpectrogramReader()
image, report = reader.read(schedule)
# image: numpy RGB array; report: markdown spectrum reading
```

---

## Installation

### Prerequisites

* Python 3.10 or newer
* An LLM backend — pick one:
  * **[OpenRouter](https://openrouter.ai/)** — easiest path; one API key
    gives access to Gemini, Claude, Qwen, Llama, and other models.
  * **[llama.cpp](https://github.com/ggerganov/llama.cpp) server** — run
    models locally via the OpenAI-compatible endpoint:

    ```bash
    llama-server -m model.gguf --port 8080
    ```

### Install

```bash
git clone https://github.com/iblameandrew/attention-grammar.git
cd attention-grammar
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root (or export the variables in
your shell):

```bash
# OpenRouter (cloud)
echo 'OPENROUTER_API_KEY=your_api_key_here' > .env

# llama.cpp server (local) — optional overrides
# LLAMA_CPP_BASE_URL=http://127.0.0.1:8080/v1
# LLAMA_CPP_MODEL=local
# LLAMA_CPP_API_KEY=no-key
```

---

## Usage

### Interactive UI

```bash
python app.py
```

This launches a three-pane Gradio interface. Type a sentence in
*Context*, choose **OpenRouter** or **llama.cpp (Local)**, enter a model slug,
click *Analyze*, and watch the algebra, the math schedule, and the
cognitive spectrogram appear in order.

### Programmatic

```python
import os
from attention_algebra import AlgebraAnalyst, Composer, SpectrogramReader

os.environ.setdefault("OPENROUTER_API_KEY", "...")
model = "google/gemini-2.5-flash"

# Layer 1: parse.
algebra = AlgebraAnalyst(model_name=model, provider="openrouter").analyze(
    "I surge impulsively but feel held back by the past average."
)

# Layer 2: compose.
schedule = Composer(model_name=model, provider="openrouter").compose(algebra)

# Layer 3: spectrogram.
image, report = SpectrogramReader().read(schedule)
```

### Local llama.cpp

```python
import os

os.environ["LLAMA_CPP_BASE_URL"] = "http://127.0.0.1:8080/v1"
os.environ["LLAMA_CPP_MODEL"] = "local"  # must match the loaded model

algebra = AlgebraAnalyst(model_name="local", provider="llama.cpp").analyze(
    "Racing sparse focus over and over."
)
```

---

## Thought Library — parasitic sequences & hermetic legislation

Beyond single expressions, Attention Grammar can **enumerate permutations of
reactives** up to sequence length $N$, batch-classify each chain with a
GenAI (or offline hermetic heuristic), and **legislate** formal canons over
which thought-forms may circulate.

### Doctrine

| Idea | Encoding |
| :--- | :------- |
| **12th reactive** | `Df` (Diffuse) = **Pisces** = natural **pathogen** of form |
| **Water** | Cancer=`Rt`, Scorpio=`Ox`, Pisces=`Df` — element of **death & release** |
| **Hermetic lens** | Mentalism, Correspondence, Vibration, Polarity, Rhythm, Cause & Effect, Gender |
| **Verdicts** | `parasitic` · `symbiotic` · `catalytic` · `neutral` |
| **Canons** | severity `permit` / `restrict` / `ban` / `ritual` |

Leading pathogen (`Df >> …`), boom-collapse (`Ex|Ei|Nv|Im >> Df`), and
water-majority drowning are typical **parasitic** patterns. The singleton
`[Df]` is **catalytic** — a rite of pure release, not a host-feeding parasite.

### CLI

```bash
# Hermetic reactive table (sign / element / pathogen flag)
python thought_library_build.py --print-table

# Offline scaffold (no API) — heuristic classify + legislate
python thought_library_build.py --dry-run -N 2 --limit 48 -o thought_library_out

# Full GenAI batches (requires OPENROUTER_API_KEY)
python thought_library_build.py -N 2 --batch-size 16 -o thought_library_out

# Sample length-3 space with pathogen enrichment
python thought_library_build.py -N 3 --limit 96 --batch-size 12
```

### Programmatic

```python
from attention_algebra import ThoughtLibrarian, save_library
from pathlib import Path

# dry_run=True uses local hermetic heuristics (no API)
lib = ThoughtLibrarian(dry_run=True).build(max_length=2, limit=48, batch_size=12)
paths = save_library(lib, Path("thought_library_out"))
print(lib.stats)           # parasitic / symbiotic / catalytic counts
print(paths["legislation"])  # LEGISLATION.md with canons
```

### Outputs

| File | Contents |
| :--- | :------- |
| `thought_library.json` | Full corpus of classified sequences |
| `LEGISLATION.md` | Preamble + canons + parasitic index |
| `parasitic.json` | Parasitic records only (downstream filters) |

Modules: [`attention_algebra/hermetic.py`](attention_algebra/hermetic.py),
[`attention_algebra/thought_library.py`](attention_algebra/thought_library.py),
CLI [`thought_library_build.py`](thought_library_build.py).

---

## Repository Layout

```
attention-grammar/
├── assets/                     # README banner image
├── app.py                      # Gradio front-end
├── thought_library_build.py    # CLI: permute reactives, classify, legislate
├── attention_algebra/
│   ├── algebra.py              # Layer 1 — the grammar + the analyst
│   ├── composition.py          # Layer 2 — the harmonic composer
│   ├── spectrum.py             # Layer 3 — the spectrogram reader
│   ├── terminals.py            # Twelve geometric terminals + Zodiac map
│   ├── hermetic.py             # Sign / element / pathogen correspondences
│   ├── thought_library.py      # Enumerate → classify → legislate
│   ├── parser.py               # Programmatic grammar validation
│   ├── config.py               # Model factory (OpenRouter / llama.cpp)
│   ├── utils.py                # strip_think_tags, strip_code_fences
│   └── __init__.py
├── tests/
├── requirements.txt
└── README.md
```

---

## Roadmap to AGI

We use the grammar as an *evaluation instrument*. A model that can
parse into, reason over, and emit from all twelve geometric terminals
in all operator configurations is by construction manipulating the
same functional constituents a structured latent dynamics engine uses.
Coupled with a target distribution over parse trees (derived
empirically from a corpus of human reasoning traces), the grammar lets
us define a *cognitive completeness* criterion for AGI that is
independent of any particular benchmark: a model is generally
intelligent iff its distribution over grammar expressions matches the
human distribution under KL divergence below a threshold.

In the meantime, Attention Grammar is a tool for *diagnosing* what a model is
doing. Drop a chain-of-thought trace in, get the grammar expression
out, and read the cognitive state of the model the way a spectrogram
reads a sound.

---

## Future direction and personal note

I've spent 6 years *hand-writing* the original operators and the *people
patterns* that make it meaningful — ever since 2020 I've been running big
data experiments with massive data dumps all over the web, running spectral
clustering, trying to understand what was the hidden order to psychology:
the answer is attention. The next step would be to make a collection of
patterns that group the grammar expressions into taxonomies — I address that
this is crackpot territory but honestly, it's still a good hobby: to me
there's no difference between psychology and physics, and this a hill I'm
willing to die on. I will eventually train a tiny model with this and see how
much using this as meta-tagging does an economy on tokens spent.

A lot of this README was automated, so some slop is expected.

Have a nice day.

---

## License

MIT. See [`LICENSE`](LICENSE).
