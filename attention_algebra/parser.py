"""Lightweight validator for Cognitive Algebra expressions.

The full grammar is prompt-specified; this module provides programmatic
well-typedness checks for the twelve geometric terminals, complementarity
pairs, and common operator constraints.  It is intentionally conservative —
it flags likely errors but does not attempt full parsing of nested RNA
structures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .terminals import (
    ATTITUDE_PAIRS,
    CROSS_AXIS_PAIRS,
    TERMINAL_ALT,
    TERMINALS,
    are_complementary,
    domain_of,
    polarity_of,
    sub_axis_of,
)

# Re-export for callers that imported from parser historically.
__all__ = [
    "TERMINALS",
    "ValidationResult",
    "are_complementary",
    "extract_terminals",
    "validate_expression",
    "ATTITUDE_PAIRS",
    "CROSS_AXIS_PAIRS",
]

TERMINAL_RE = re.compile(
    rf"(?<![A-Za-z])(\d+)?({TERMINAL_ALT})(?![A-Za-z])"
)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    terminals: list[tuple[str, int | None]] = field(default_factory=list)


def extract_terminals(expr: str) -> list[tuple[str, int | None]]:
    """Return ``(terminal, mass)`` tuples in left-to-right order."""
    found: list[tuple[str, int | None]] = []
    for match in TERMINAL_RE.finditer(expr):
        mass_str, term = match.groups()
        mass = int(mass_str) if mass_str else None
        found.append((term, mass))
    return found


def validate_expression(expr: str) -> ValidationResult:
    """Validate a Cognitive Algebra expression string."""
    errors: list[str] = []
    warnings: list[str] = []

    if not expr or not expr.strip():
        return ValidationResult(False, ["Expression is empty"])

    expr = expr.strip()
    terminals = extract_terminals(expr)

    if not terminals:
        errors.append(
            "No recognised terminals "
            f"({', '.join(sorted(TERMINALS))}) found"
        )
        return ValidationResult(False, errors, warnings, terminals)

    for term, mass in terminals:
        if mass is not None and not (1 <= mass <= 10):
            warnings.append(f"Mass {mass} on {term} is outside documented range 1-10")

    term_pat = TERMINAL_ALT

    # Orbit: X ~ Y requires two different domains
    for match in re.finditer(
        rf"(\d+)?({term_pat})\s*~\s*(\d+)?({term_pat})",
        expr,
    ):
        left, right = match.group(2), match.group(4)
        if domain_of(left) == domain_of(right):
            errors.append(
                f"Orbit `~` requires different domains "
                f"(TRACE / FIELD / FORM), got {left} ~ {right}"
            )

    # Opposition: same sub-axis, opposite polarity
    for match in re.finditer(
        rf"(\d+)?({term_pat})\s*oo\s*(\d+)?({term_pat})",
        expr,
    ):
        left, right = match.group(2), match.group(4)
        if sub_axis_of(left) != sub_axis_of(right):
            errors.append(
                f"Opposition `oo` requires same sub-axis, got {left} oo {right}"
            )
        elif polarity_of(left) == polarity_of(right):
            errors.append(
                f"Opposition `oo` requires opposite polarity, got {left} oo {right}"
            )

    # Axis switch | : same domain, different sub-axis
    for match in re.finditer(
        rf"(\d+)?({term_pat})\s*\|\s*(\d+)?({term_pat})",
        expr,
    ):
        left, right = match.group(2), match.group(4)
        if domain_of(left) != domain_of(right):
            errors.append(
                f"Axis switch `|` requires same domain, got {left} | {right}"
            )
        elif sub_axis_of(left) == sub_axis_of(right):
            errors.append(
                f"Axis switch `|` requires different sub-axis, got {left} | {right}"
            )

    # Ambiguous `+`: domain switch (different domains) vs conjunction
    switch_expr = re.sub(r"fold\[[^\]]*\]", "", expr)
    for match in re.finditer(
        rf"(\d+)?({term_pat})\s*\+\s*(\d+)?({term_pat})",
        switch_expr,
    ):
        left, right = match.group(2), match.group(4)
        if domain_of(left) != domain_of(right):
            pass  # valid domain switch
        elif sub_axis_of(left) == sub_axis_of(right):
            warnings.append(
                f"`{left} + {right}` may be conjunction; prefer `&` for linear sums"
            )

    # Stem pairs ::
    for match in re.finditer(
        rf"(\d+)?({term_pat})\s*::\s*(\d+)?({term_pat})",
        expr,
    ):
        left, right = match.group(2), match.group(4)
        if not are_complementary(left, right):
            errors.append(
                f"Stem pair `::` requires complementary terminals, "
                f"got {left} :: {right}"
            )

    # Drag must not target a parenthesised group
    if re.search(r"->\s*\(", expr) or re.search(r"→\s*\(", expr):
        errors.append("Drag `->` right-hand side must be a single function, not a group")

    return ValidationResult(len(errors) == 0, errors, warnings, terminals)
