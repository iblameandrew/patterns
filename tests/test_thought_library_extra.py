import pytest

from attention_algebra.thought_library import (
    ThoughtLibrarian,
    ThoughtLibrary,
    ThoughtRecord,
    count_sequences,
    enumerate_sequences,
    sample_sequences,
)


def test_count_with_replacement():
    assert count_sequences(1, with_replacement=True, n_symbols=12) == 12
    assert count_sequences(2, with_replacement=True, n_symbols=2) == 2 + 4


def test_enumerate_bad_lengths():
    with pytest.raises(ValueError):
        list(enumerate_sequences(0))
    with pytest.raises(ValueError):
        list(enumerate_sequences(2, min_length=3))


def test_sample_sequences_limit():
    picked = sample_sequences(2, limit=5, seed=0)
    assert len(picked) == 5
    assert any("Df" in s for s in picked)


def test_library_parasitic_split():
    rec = ThoughtRecord(
        sequence=["Df", "Im"],
        expression="Df >> Im",
        hermetic_gloss="",
        length=2,
        release_load=0.5,
        pathogen=True,
        pathogen_index=0,
        verdict="parasitic",
        confidence=1.0,
        hermetic_analysis="",
        legislation_hint="",
        source="heuristic",
    )
    lib = ThoughtLibrary(
        created_at="now",
        max_length=2,
        with_replacement=False,
        model="local",
        pathogen_symbol="Df",
        records=[rec],
    )
    assert lib.parasitic() == [rec]
    assert lib.non_parasitic() == []


def test_librarian_dry_run_batch():
    lib = ThoughtLibrarian(dry_run=True)
    recs = lib.analyze_batch([("Im", "Pr"), ("Df",)])
    assert len(recs) == 2
    assert lib.analyze_batch([]) == []
