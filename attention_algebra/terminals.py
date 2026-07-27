"""The twelve geometric terminals of Attention Grammar.

Each terminal is a language-agnostic short code for a pure mathematical
objective over a latent state trajectory, and each is **unraveled** as a
Jungian functional-algebra expression in parentheses (the cognitive
molecule of that reactive).

Source geometry: parallel TTT anchor objectives (velocity, centroid,
duality, recurrence, centrality, sparsity, equilibrium, orthogonality,
expansion, clamp, novelty, diffusion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class TerminalSpec:
    """One functional constituent: geometric symbol + unraveled algebra."""

    symbol: str
    name: str
    objective: str
    formula: str
    description: str
    domain: str  # TRACE | FIELD | FORM
    sub_axis: str
    polarity: str  # OPEN | CLOSE
    carrier_hz: float
    # Primary Jungian functional expansion (always parenthesised form).
    functional: str
    # Optional alternate expansion when the reactive is a disjunction (|).
    functional_alt: str | None = None


# Cycle order 0..11 of the geometric rotatory protocol.
# Each reactive is unraveled as a functional-algebra molecule.
TERMINAL_SPECS: tuple[TerminalSpec, ...] = (
    TerminalSpec(
        "Im",
        "Impulse",
        "KineticVelocityObjective",
        r"\|z_t - z_{t-1}\|",
        "Maximize step velocity — initiation and impulse.",
        "TRACE",
        "kinetic",
        "OPEN",
        82.4,
        "(Se)",
    ),
    TerminalSpec(
        "An",
        "Anchor",
        "CentroidStabilityObjective",
        r"-\|z_t - \mu_{\mathrm{hist}}\|",
        "Minimize distance to history centroid — grounding.",
        "TRACE",
        "kinetic",
        "CLOSE",
        98.0,
        "(Si -> Ne)",
    ),
    TerminalSpec(
        "Bi",
        "Bifurcate",
        "TemporalDualityObjective",
        r"\|z_t - z_{t-2}\| - 0.5\|z_t - z_{t-1}\|",
        "Induce temporal bimodality — dual / split modes.",
        "TRACE",
        "cyclic",
        "OPEN",
        123.5,
        "((Ne oo Ni) ~ Ti)",
    ),
    TerminalSpec(
        "Rt",
        "Return",
        "CyclicRecurrenceObjective",
        r"\cos(z_t, z_0)",
        "Maximize similarity to origin — closed recurrence.",
        "TRACE",
        "cyclic",
        "CLOSE",
        146.8,
        "(Si ~ Fe)",
    ),
    TerminalSpec(
        "Ei",
        "Eigen",
        "RepresentativeCentralityObjective",
        r"\cos(z_t, \mu_{\mathrm{batch}}) \cdot \|z_t\|",
        "Align with batch principal axis — eigen-centrality.",
        "FIELD",
        "structure",
        "OPEN",
        174.6,
        "((Fi oo Fe) -> Te)",
    ),
    TerminalSpec(
        "Pr",
        "Prune",
        "SparsePrecisionObjective",
        r"-\|z_t\|_1",
        "Minimize L1 magnitude — sparse precision.",
        "FORM",
        "magnitude",
        "CLOSE",
        196.0,
        "(Si ~ Te oo Ti)",
        "(Ni ~ Fe oo Fi)",
    ),
    TerminalSpec(
        "Hm",
        "Harmon",
        "HarmonicEquilibriumObjective",
        r"-\|z_t - \mu_{\mathrm{batch}}\|",
        "Minimize distance to batch mean — social equilibrium.",
        "FIELD",
        "radial",
        "CLOSE",
        220.0,
        "((Fe oo Fi) ~ Ni)",
    ),
    TerminalSpec(
        "Ox",
        "Orth",
        "LatentOrthogonalityObjective",
        r"1 - |\cos(z_t, \mu_{\mathrm{hist}})|",
        "Maximize orthogonality to history surface — hidden axes.",
        "FORM",
        "orientation",
        "OPEN",
        261.6,
        "(Ni -> Se)",
    ),
    TerminalSpec(
        "Ex",
        "Expand",
        "VectorExpansionObjective",
        r"\|z_t\|",
        "Maximize vector magnitude — projection outward.",
        "FORM",
        "magnitude",
        "OPEN",
        293.7,
        "(Se ~ Ti)",
        "(Ne ~ Fi)",
    ),
    TerminalSpec(
        "Bd",
        "Bound",
        "StructuralConstraintObjective",
        r"-10\|z_t - \mathrm{clamp}(z_t, -1, 1)\|",
        "Penalize out-of-bounds state — hard structural clamp.",
        "FIELD",
        "structure",
        "CLOSE",
        349.2,
        "((Te oo Ti) ~ Ni)",
    ),
    TerminalSpec(
        "Nv",
        "Novel",
        "DiversityNoveltyObjective",
        r"\|z_t - \mu_{\mathrm{batch}}\|",
        "Maximize distance from batch center — outlier novelty.",
        "FIELD",
        "radial",
        "OPEN",
        392.0,
        "((Ti -> Fe) ~ Ne)",
    ),
    TerminalSpec(
        "Df",
        "Diffuse",
        "EntropicDiffusionObjective",
        r"-\max_i |z_t^{(i)}|",
        "Flatten vector peaks — entropic diffusion.",
        "FORM",
        "orientation",
        "CLOSE",
        440.0,
        "(Fi ~ (Ne oo Ni))",
    ),
)

TERMINALS: FrozenSet[str] = frozenset(s.symbol for s in TERMINAL_SPECS)
TERMINAL_ORDER: list[str] = [s.symbol for s in TERMINAL_SPECS]
TERMINAL_BY_SYMBOL: dict[str, TerminalSpec] = {s.symbol: s for s in TERMINAL_SPECS}

# Domain partitions
TRACE: FrozenSet[str] = frozenset(s.symbol for s in TERMINAL_SPECS if s.domain == "TRACE")
FIELD: FrozenSet[str] = frozenset(s.symbol for s in TERMINAL_SPECS if s.domain == "FIELD")
FORM: FrozenSet[str] = frozenset(s.symbol for s in TERMINAL_SPECS if s.domain == "FORM")

OPEN: FrozenSet[str] = frozenset(s.symbol for s in TERMINAL_SPECS if s.polarity == "OPEN")
CLOSE: FrozenSet[str] = frozenset(s.symbol for s in TERMINAL_SPECS if s.polarity == "CLOSE")

# Sub-axis membership
KINETIC: FrozenSet[str] = frozenset({"Im", "An"})
CYCLIC: FrozenSet[str] = frozenset({"Bi", "Rt"})
STRUCTURE: FrozenSet[str] = frozenset({"Ei", "Bd"})
RADIAL: FrozenSet[str] = frozenset({"Nv", "Hm"})
MAGNITUDE: FrozenSet[str] = frozenset({"Ex", "Pr"})
ORIENTATION: FrozenSet[str] = frozenset({"Ox", "Df"})

# Release triad — recurrence, hidden-axis transform, entropic diffusion
RELEASE: FrozenSet[str] = frozenset({"Rt", "Ox", "Df"})

# Regime A — polarity pairs (same sub-axis)
ATTITUDE_PAIRS: dict[str, str] = {
    "Im": "An",
    "An": "Im",
    "Bi": "Rt",
    "Rt": "Bi",
    "Nv": "Hm",
    "Hm": "Nv",
    "Ei": "Bd",
    "Bd": "Ei",
    "Ex": "Pr",
    "Pr": "Ex",
    "Ox": "Df",
    "Df": "Ox",
}

# Regime B — cross-axis complements within domain
CROSS_AXIS_PAIRS: dict[str, str] = {
    "Im": "Rt",
    "Rt": "Im",
    "An": "Bi",
    "Bi": "An",
    "Nv": "Bd",
    "Bd": "Nv",
    "Hm": "Ei",
    "Ei": "Hm",
    "Ex": "Df",
    "Df": "Ex",
    "Pr": "Ox",
    "Ox": "Pr",
}

SYMBOL_TO_TERMINAL: dict[str, str] = {
    s.objective: s.symbol for s in TERMINAL_SPECS
}

VOICE_NAME_MAP: dict[str, str] = {}
for _spec in TERMINAL_SPECS:
    VOICE_NAME_MAP[_spec.name.lower()] = _spec.symbol
    VOICE_NAME_MAP[_spec.symbol.lower()] = _spec.symbol
    for word in _spec.name.lower().split():
        if word not in VOICE_NAME_MAP:
            VOICE_NAME_MAP[word] = _spec.symbol

TERMINAL_FREQS: dict[str, float] = {s.symbol: s.carrier_hz for s in TERMINAL_SPECS}

TERMINAL_ALT = "|".join(TERMINAL_ORDER)

# Technical labels for the geometric rotatory anchors
GEOMETRIC_ANCHOR: dict[str, str] = {
    "Im": "KINETIC_VELOCITY",
    "An": "CENTROID_STABILITY",
    "Bi": "TEMPORAL_DUALITY",
    "Rt": "CYCLIC_RECURRENCE",
    "Ei": "REPRESENTATIVE_CENTRALITY",
    "Pr": "SPARSE_PRECISION",
    "Hm": "HARMONIC_EQUILIBRIUM",
    "Ox": "LATENT_ORTHOGONALITY",
    "Ex": "VECTOR_EXPANSION",
    "Bd": "STRUCTURAL_CONSTRAINT",
    "Nv": "DIVERSITY_NOVELTY",
    "Df": "ENTROPIC_DIFFUSION",
}

# Back-compat alias
ZODIAC_ANCHOR = GEOMETRIC_ANCHOR

SYMBOL_FUNCTIONAL: dict[str, str] = {s.symbol: s.functional for s in TERMINAL_SPECS}


def unravel(symbol: str) -> str:
    """Return the full functional display, including alternate if present.

    Examples
    --------
    >>> unravel("Im")
    '(Se)'
    >>> unravel("Pr")
    '(Si ~ Te oo Ti) | (Ni ~ Fe oo Fi)'
    """
    spec = TERMINAL_BY_SYMBOL.get(symbol)
    if spec is None:
        raise KeyError(f"Unknown terminal: {symbol!r}")
    if spec.functional_alt:
        return f"{spec.functional} | {spec.functional_alt}"
    return spec.functional


def domain_of(term: str) -> str:
    spec = TERMINAL_BY_SYMBOL.get(term)
    return spec.domain if spec else "unknown"


def sub_axis_of(term: str) -> str:
    spec = TERMINAL_BY_SYMBOL.get(term)
    return spec.sub_axis if spec else "unknown"


def polarity_of(term: str) -> str:
    spec = TERMINAL_BY_SYMBOL.get(term)
    return spec.polarity if spec else "unknown"


def are_complementary(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` form a valid stem pair (Regime A or B)."""
    return ATTITUDE_PAIRS.get(a) == b or CROSS_AXIS_PAIRS.get(a) == b


def drag_target(winner: str) -> str | None:
    """Opposition drag: flip sub-axis within domain, preserve polarity."""
    spec = TERMINAL_BY_SYMBOL.get(winner)
    if not spec:
        return None
    flip_axis = {
        "kinetic": "cyclic",
        "cyclic": "kinetic",
        "radial": "structure",
        "structure": "radial",
        "magnitude": "orientation",
        "orientation": "magnitude",
    }.get(spec.sub_axis)
    if not flip_axis:
        return None
    for other in TERMINAL_SPECS:
        if (
            other.domain == spec.domain
            and other.sub_axis == flip_axis
            and other.polarity == spec.polarity
        ):
            return other.symbol
    return None
