"""Hermetic and classical-element correspondences for the twelve reactives.

The Great-Year order of geometric terminals is paired 1:1 with the tropical
zodiac.  Each sign is **unraveled** as a Jungian functional-algebra molecule
in parentheses (e.g. Capricorn = ``((Te oo Ti) ~ Ni)``).

The twelfth reactive — ``Df`` (Diffuse / ENTROPIC_DIFFUSION) — corresponds
to **Pisces** ``(Fi ~ (Ne oo Ni))``: water of death, release, and dissolution.
It is the natural *pathogen* of the thought ecology.

Hermetic lens (Kybalion):
  1. Mentalism — all is Mind; a thought-sequence is a mental state.
  2. Correspondence — as above (algebra), so below (spectrogram / body).
  3. Vibration — every reactive has a carrier; sequences modulate rhythm.
  4. Polarity — OPEN/CLOSE poles; parasites invert poles without consent.
  5. Rhythm — pendula; parasitic chains freeze or drown the swing.
  6. Cause & Effect — every sequence legislates a consequence.
  7. Gender — generation: OPEN seeds, CLOSE gestates; water releases seed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .terminals import (
    TERMINAL_BY_SYMBOL,
    TERMINAL_ORDER,
    TerminalSpec,
    unravel,
)


@dataclass(frozen=True)
class HermeticReactive:
    """One reactive under hermetic / elemental / functional correspondence."""

    symbol: str
    name: str
    index: int  # 1..12 Great-Year position
    sign: str
    element: str  # Fire | Earth | Air | Water
    modality: str  # Cardinal | Fixed | Mutable
    house_theme: str
    math_role: str
    functional: str  # primary unraveled form, e.g. "((Te oo Ti) ~ Ni)"
    functional_display: str  # full form including | alternate if any
    is_natural_pathogen: bool
    is_water: bool
    polar_note: str


# Tropical zodiac order aligned with TERMINAL_ORDER[i].
_ZODIAC_ROW: tuple[tuple[str, str, str, str], ...] = (
    ("Aries", "Fire", "Cardinal", "initiation / will-to-begin"),
    ("Taurus", "Earth", "Fixed", "holding / material continuity"),
    ("Gemini", "Air", "Mutable", "dual speech / bifurcation of mind"),
    ("Cancer", "Water", "Cardinal", "shell of memory / return to origin"),
    ("Leo", "Fire", "Fixed", "solar centrality / principal ray"),
    ("Virgo", "Earth", "Mutable", "discrimination / sparse harvest"),
    ("Libra", "Air", "Cardinal", "balance with the other / equilibrium"),
    ("Scorpio", "Water", "Fixed", "occult axis / death-as-transformation"),
    ("Sagittarius", "Fire", "Mutable", "expansion / far projection"),
    ("Capricorn", "Earth", "Cardinal", "law / structural bound"),
    ("Aquarius", "Air", "Fixed", "novelty outside the herd"),
    ("Pisces", "Water", "Mutable", "dissolution / release / pathogen of form"),
)

NATURAL_PATHOGEN = "Df"  # Pisces — 12th reactive
WATER_OF_DEATH = "Water"

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
        sign, element, modality, house = _ZODIAC_ROW[i]
        spec: TerminalSpec = TERMINAL_BY_SYMBOL[symbol]
        table[symbol] = HermeticReactive(
            symbol=symbol,
            name=spec.name,
            index=i + 1,
            sign=sign,
            element=element,
            modality=modality,
            house_theme=house,
            math_role=spec.description,
            functional=spec.functional,
            functional_display=unravel(symbol),
            is_natural_pathogen=(symbol == NATURAL_PATHOGEN),
            is_water=(element == WATER_OF_DEATH),
            polar_note=f"{spec.polarity} · {spec.domain}/{spec.sub_axis}",
        )
    return table


HERMETIC_TABLE: dict[str, HermeticReactive] = build_hermetic_table()


def describe_sequence(symbols: tuple[str, ...]) -> str:
    """Human-readable hermetic gloss including functional expansions."""
    parts: list[str] = []
    for s in symbols:
        h = HERMETIC_TABLE[s]
        tag = "PATHOGEN" if h.is_natural_pathogen else h.element.upper()
        parts.append(
            f"{h.symbol}={h.name}/{h.sign}[{tag}]{h.functional_display}"
        )
    return " → ".join(parts)


def water_load(symbols: tuple[str, ...]) -> float:
    """Fraction of water-element reactives in the sequence."""
    if not symbols:
        return 0.0
    return sum(1 for s in symbols if HERMETIC_TABLE[s].is_water) / len(symbols)


def pathogen_present(symbols: tuple[str, ...]) -> bool:
    return NATURAL_PATHOGEN in symbols


def pathogen_position(symbols: tuple[str, ...]) -> int | None:
    """0-based index of first Df, or None."""
    try:
        return symbols.index(NATURAL_PATHOGEN)
    except ValueError:
        return None
