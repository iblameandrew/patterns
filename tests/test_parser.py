"""Tests for Cognitive Algebra expression validation (12 geometric terminals)."""

from attention_algebra.parser import (
    are_complementary,
    extract_terminals,
    validate_expression,
)
from attention_algebra.terminals import TERMINALS, drag_target


def test_twelve_terminals():
    assert len(TERMINALS) == 12


def test_extract_terminals_with_mass():
    found = extract_terminals("7Im oo 3An -> Bi")
    assert found == [("Im", 7), ("An", 3), ("Bi", None)]


def test_complementarity_regime_a():
    assert are_complementary("Im", "An")
    assert are_complementary("Ex", "Pr")
    assert are_complementary("Nv", "Hm")


def test_complementarity_regime_b():
    assert are_complementary("Im", "Rt")
    assert are_complementary("Ex", "Df")
    assert are_complementary("Nv", "Bd")


def test_valid_orbit():
    result = validate_expression("(Im ~ Pr)")
    assert result.valid
    assert ("Im", None) in result.terminals


def test_invalid_orbit_same_domain():
    result = validate_expression("Im ~ An")
    assert not result.valid
    assert any("Orbit" in e for e in result.errors)


def test_valid_opposition():
    result = validate_expression("7Im oo 3An -> Bi")
    assert result.valid


def test_invalid_stem_pair():
    result = validate_expression("5Nv :: 3Ex")
    assert not result.valid
    assert any("Stem pair" in e for e in result.errors)


def test_valid_stem_pair_cross_axis():
    result = validate_expression("5Im :: 3Rt")
    assert result.valid


def test_valid_stem_pair_attitude():
    result = validate_expression("5Ex :: 4Pr")
    assert result.valid


def test_hairpin_expression():
    result = validate_expression("^(5Rt)")
    assert result.valid
    assert result.terminals == [("Rt", 5)]


def test_drag_target_preserves_polarity():
    # Im is OPEN kinetic TRACE → flip to cyclic OPEN = Bi
    assert drag_target("Im") == "Bi"
    # An is CLOSE kinetic TRACE → flip to cyclic CLOSE = Rt
    assert drag_target("An") == "Rt"
    # Ex is OPEN magnitude FORM → orientation OPEN = Ox
    assert drag_target("Ex") == "Ox"
