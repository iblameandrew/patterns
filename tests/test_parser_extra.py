from attention_algebra.parser import extract_terminals, validate_expression
from attention_algebra.terminals import drag_target, unravel


def test_empty_expression():
    result = validate_expression("   ")
    assert not result.valid
    assert "empty" in result.errors[0].lower()


def test_no_terminals():
    result = validate_expression("hello world")
    assert not result.valid
    assert any("No recognised" in e for e in result.errors)


def test_mass_out_of_range_is_warning():
    result = validate_expression("99Im")
    assert result.valid
    assert any("Mass 99" in w for w in result.warnings)


def test_invalid_opposition_same_polarity():
    result = validate_expression("Im oo Bi")
    assert not result.valid
    assert any("Opposition" in e for e in result.errors)


def test_invalid_axis_switch_cross_domain():
    result = validate_expression("Im | Nv")
    assert not result.valid
    assert any("Axis switch" in e for e in result.errors)


def test_valid_axis_switch():
    result = validate_expression("Im | Bi")
    assert result.valid


def test_drag_to_group_is_error():
    result = validate_expression("Im -> (Pr)")
    assert not result.valid
    assert any("Drag" in e for e in result.errors)


def test_plus_conjunction_warning():
    result = validate_expression("Im + An")
    assert result.valid
    assert any("conjunction" in w for w in result.warnings)


def test_extract_empty():
    assert extract_terminals("") == []


def test_unravel_unknown():
    import pytest

    with pytest.raises(KeyError):
        unravel("Zz")


def test_drag_target_unknown():
    assert drag_target("Zz") is None
