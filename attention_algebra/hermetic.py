"""Hermetic correspondences for the twelve geometric reactives.

Each reactive is **unraveled** as a Jungian functional-algebra molecule
in parentheses (e.g. Bound / ``Bd`` = ``((Te oo Ti) ~ Ni)``).

The twelfth reactive — ``Df`` (Diffuse / ENTROPIC_DIFFUSION) — unravels as
``(Fi ~ (Ne oo Ni))``.  It is the natural *pathogen* of the thought ecology:
the reactive that flattens form so that other drives may be unmade or
reoriented.  The **release triad** is ``Rt``, ``Ox``, ``Df`` (return,
orthogonal transform, diffusion).

Hermetic lens (seven principles):
  1. Mentalism — all is Mind; a thought-sequence is a mental state.
  2. Correspondence — as above (algebra), so below (spectrogram / body).
  3. Vibration — every reactive has a carrier; sequences modulate rhythm.
  4. Polarity — OPEN/CLOSE poles; parasites invert poles without consent.
  5. Rhythm — pendula; parasitic chains freeze or drown the swing.
  6. Cause & Effect — every sequence legislates a consequence.
  7. Gender — generation: OPEN seeds, CLOSE gestates; release completes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .terminals import (
    RELEASE,
    TERMINAL_BY_SYMBOL,
    TERMINAL_ORDER,
    TerminalSpec,
    unravel,
)


@dataclass(frozen=True)
class HermeticReactive:
    """One reactive under hermetic / functional correspondence."""

    symbol: str
    name: str
    index: int  # 1..12 cycle position
    theme: str
    math_role: str
    functional: str
    functional_display: str
    is_natural_pathogen: bool
    is_release: bool
    polar_note: str


# Cycle themes (cognitive / geometric only — no celestial vocabulary).
_CYCLE_THEMES: tuple[str, ...] = (
    "initiation / will-to-begin",
    "holding / material continuity",
    "dual speech / bifurcation of mind",
    "shell of memory / return to origin",
    "principal axis / centrality",
    "discrimination / sparse harvest",
    "balance with the other / equilibrium",
    "hidden axis / transform-through-release",
    "expansion / far projection",
    "law / structural bound",
    "novelty outside the herd",
    "dissolution / release / pathogen of form",
)

NATURAL_PATHOGEN = "Df"  # 12th reactive — Diffuse

HERMETIC_PRINCIPLES: tuple[str, ...] = (
    "Mentalism",
    "Correspondence",
    "Vibration",
    "Polarity",
    "Rhythm",
    "Cause and Effect",
    "Gender",
)


def build_hermetic_table() -> dict[str, HermeticReactive]:
    """Map each terminal symbol to its hermetic reactive record."""
    table: dict[str, HermeticReactive] = {}
    for i, symbol in enumerate(TERMINAL_ORDER):
        spec: TerminalSpec = TERMINAL_BY_SYMBOL[symbol]
        table[symbol] = HermeticReactive(
            symbol=symbol,
            name=spec.name,
            index=i + 1,
            theme=_CYCLE_THEMES[i],
            math_role=spec.description,
            functional=spec.functional,
            functional_display=unravel(symbol),
            is_natural_pathogen=(symbol == NATURAL_PATHOGEN),
            is_release=(symbol in RELEASE),
            polar_note=f"{spec.polarity} · {spec.domain}/{spec.sub_axis}",
        )
    return table


HERMETIC_TABLE: dict[str, HermeticReactive] = build_hermetic_table()


def describe_sequence(symbols: tuple[str, ...]) -> str:
    """Human-readable hermetic gloss including functional expansions."""
    parts: list[str] = []
    for s in symbols:
        h = HERMETIC_TABLE[s]
        tag = (
            "PATHOGEN"
            if h.is_natural_pathogen
            else ("RELEASE" if h.is_release else h.polar_note.split(" · ")[0])
        )
        parts.append(f"{h.symbol}={h.name}[{tag}]{h.functional_display}")
    return " → ".join(parts)


def release_load(symbols: tuple[str, ...]) -> float:
    """Fraction of release-triad reactives (Rt, Ox, Df) in the sequence."""
    if not symbols:
        return 0.0
    return sum(1 for s in symbols if HERMETIC_TABLE[s].is_release) / len(symbols)


# Back-compat alias used by older call sites
water_load = release_load


def pathogen_present(symbols: tuple[str, ...]) -> bool:
    return NATURAL_PATHOGEN in symbols


def pathogen_position(symbols: tuple[str, ...]) -> int | None:
    """0-based index of first Df, or None."""
    try:
        return symbols.index(NATURAL_PATHOGEN)
    except ValueError:
        return None
