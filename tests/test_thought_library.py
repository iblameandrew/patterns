"""Tests for hermetic thought library (offline / dry-run)."""

from attention_algebra.hermetic import (
    HERMETIC_TABLE,
    NATURAL_PATHOGEN,
    pathogen_present,
    water_load,
)
from attention_algebra.terminals import TERMINAL_ORDER
from attention_algebra.thought_library import (
    ThoughtLibrarian,
    count_sequences,
    enumerate_sequences,
    heuristic_classify,
    save_library,
)


def test_twelve_hermetic_signs():
    assert len(HERMETIC_TABLE) == 12
    assert HERMETIC_TABLE[NATURAL_PATHOGEN].sign == "Pisces"
    assert HERMETIC_TABLE[NATURAL_PATHOGEN].is_natural_pathogen
    assert HERMETIC_TABLE[NATURAL_PATHOGEN].element == "Water"
    # 12th in Great-Year order
    assert TERMINAL_ORDER[11] == NATURAL_PATHOGEN


def test_water_triad():
    water = [s for s, h in HERMETIC_TABLE.items() if h.is_water]
    assert set(water) == {"Rt", "Ox", "Df"}  # Cancer, Scorpio, Pisces


def test_count_permutations_n2():
    # P(12,1)+P(12,2) = 12 + 132 = 144
    assert count_sequences(2, min_length=1) == 144


def test_enumerate_respects_length():
    seqs = list(enumerate_sequences(2, min_length=2))
    assert all(len(s) == 2 for s in seqs)
    assert len(seqs) == 12 * 11


def test_leading_pathogen_is_parasitic():
    rec = heuristic_classify((NATURAL_PATHOGEN, "Im"))
    assert rec.verdict == "parasitic"
    assert rec.pathogen


def test_pathogen_monad_is_catalytic():
    rec = heuristic_classify((NATURAL_PATHOGEN,))
    assert rec.verdict == "catalytic"


def test_boom_collapse_parasitic():
    rec = heuristic_classify(("Ex", NATURAL_PATHOGEN))
    assert rec.verdict == "parasitic"


def test_balanced_without_pathogen_symbiotic_or_neutral():
    rec = heuristic_classify(("Im", "Pr"))
    assert rec.verdict in ("symbiotic", "neutral")
    assert not rec.pathogen


def test_water_load():
    assert water_load(("Rt", "Ox", "Df")) == 1.0
    assert pathogen_present(("Im", "Df"))


def test_dry_run_build_and_save(tmp_path):
    lib = ThoughtLibrarian(dry_run=True).build(
        max_length=1,
        limit=None,
        batch_size=6,
    )
    assert lib.stats["analyzed"] == 12
    assert lib.canons
    assert any(c.severity == "ban" for c in lib.canons)
    paths = save_library(lib, tmp_path)
    assert paths["json"].is_file()
    assert paths["legislation"].is_file()
    assert "Pisces" in paths["legislation"].read_text(encoding="utf-8")
