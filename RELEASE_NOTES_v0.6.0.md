# Attention Grammar v0.6.0

Geometric terminals + Hermetic Thought Library.

## Highlights

### Twelve geometric reactives (replaces Jungian eight)

The grammar terminals are no longer Jungian cognitive functions (`Se`…`Fi`).
They are **twelve language-agnostic geometric objectives** aligned with the
parallel-TTT geometric anchors, each a pure reward on latent $z_t$:

| Symbol | Name | Math role |
| :----: | :--- | :-------- |
| `Im` | Impulse | maximize step velocity |
| `An` | Anchor | history centroid stability |
| `Bi` | Bifurcate | temporal duality |
| `Rt` | Return | cyclic recurrence to origin |
| `Ei` | Eigen | batch representative centrality |
| `Pr` | Prune | L1 sparse precision |
| `Hm` | Harmon | batch harmonic equilibrium |
| `Ox` | Orth | latent orthogonality to history |
| `Ex` | Expand | vector magnitude expansion |
| `Bd` | Bound | structural clamp |
| `Nv` | Novel | diversity / outlier novelty |
| `Df` | Diffuse | entropic diffusion (**natural pathogen**) |

Domains: **TRACE** / **FIELD** / **FORM**. Polarity: **OPEN** / **CLOSE**.
Orbit, opposition, axis switch, and stem complementarity are retyped
accordingly. Canonical table: `attention_algebra/terminals.py`.

Spectrogram Layer 3 still works: twelve carrier bands (82.4–440 Hz), same
envelope logics, RGB image + report.

### Hermetic Thought Library

New pipeline enumerates reactive sequences up to length $N$, batch-classifies
them (GenAI or offline heuristic), and **legislates** canons.

- **Doctrine:** `Df` = natural pathogen; release triad Rt/Ox/Df = death & release of form.
- **Verdicts:** parasitic · symbiotic · catalytic · neutral.
- **Canons:** permit / restrict / ban / ritual.
- **CLI:** `python thought_library_build.py --dry-run -N 2 --limit 48`
- **API:** `ThoughtLibrarian`, `ThoughtLibrary`, `save_library`

## Migration from v0.5

```python
# Before (v0.5) — Jungian terminals
expr = "(Ne ~ Ti)"
schedule score symbols: ExtrapolationObjective, ContrastObjective

# After (v0.6) — geometric terminals
expr = "(Im ~ Pr)"
schedule score symbols: KineticVelocityObjective, SparsePrecisionObjective
```

Update any saved algebra strings, tests, and prompts that still use
`Se|Si|Ne|Ni|Te|Ti|Fe|Fi`.

## Files added

- `attention_algebra/terminals.py`
- `attention_algebra/hermetic.py`
- `attention_algebra/thought_library.py`
- `thought_library_build.py`
- `tests/test_thought_library.py`
- `RELEASE_NOTES_v0.6.0.md`

## Version

`attention_algebra.__version__` → `0.6.0`
