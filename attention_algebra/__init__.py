"""Public API for the ``attention_algebra`` package (Attention Grammar).

The three layers of the Cognitive Grammar compiler are exposed as
top-level classes so that the typical user code is::

    from attention_algebra import AlgebraAnalyst, Composer, SpectrogramReader

Each geometric terminal unravels to a Jungian functional molecule via
``terminals.unravel`` (e.g. Capricorn / ``Bd`` → ``((Te oo Ti) ~ Ni)``).
"""

from .algebra import AlgebraAnalyst
from .composition import Composer
from .config import ModelFactory
from .parser import ValidationResult, are_complementary, validate_expression
from .spectrum import SpectrogramReader, SpectrumResult
from .terminals import (
    SIGN_FUNCTIONAL,
    TERMINAL_ORDER,
    TERMINAL_SPECS,
    TERMINALS,
    ZODIAC_ANCHOR,
    unravel,
)
from .thought_library import ThoughtLibrarian, ThoughtLibrary, save_library
from .utils import strip_code_fences, strip_think_tags

__all__ = [
    "AlgebraAnalyst",
    "Composer",
    "ModelFactory",
    "SpectrogramReader",
    "SpectrumResult",
    "SIGN_FUNCTIONAL",
    "TERMINALS",
    "TERMINAL_ORDER",
    "TERMINAL_SPECS",
    "ThoughtLibrarian",
    "ThoughtLibrary",
    "ValidationResult",
    "ZODIAC_ANCHOR",
    "are_complementary",
    "save_library",
    "strip_code_fences",
    "strip_think_tags",
    "unravel",
    "validate_expression",
]

__version__ = "0.6.1"