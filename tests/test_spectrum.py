"""Tests for spectrogram rendering with geometric terminals."""

import numpy as np

from attention_algebra.spectrum import SpectrogramReader
from attention_algebra.terminals import TERMINAL_FREQS


def _sample_schedule(logic: str = "Orbital") -> dict:
    return {
        "original_expression": "(Im ~ Pr)",
        "schedule_logic": logic,
        "global_frequency": 2.0,
        "fold_energy": None,
        "score": [
            {
                "voice": "Impulse (Im)",
                "symbol": "KineticVelocityObjective",
                "mass": 7.0,
                "formula": "||z_t - z_{t-1}||",
                "description": "Maximize step velocity",
                "role": "primary",
            },
            {
                "voice": "Prune (Pr)",
                "symbol": "SparsePrecisionObjective",
                "mass": 5.0,
                "formula": "-||z_t||_1",
                "description": "L1 compression",
                "role": "secondary",
            },
        ],
        "math_narrative": "Orbital interplay between impulse and sparse precision.",
    }


def test_synthesize_produces_signal():
    reader = SpectrogramReader(duration=0.5)
    signal, labels, _ = reader.synthesize(_sample_schedule())
    assert len(signal) > 0
    assert labels == ["Im", "Pr"]
    assert np.max(np.abs(signal)) <= 1.0


def test_render_returns_image():
    reader = SpectrogramReader(duration=0.5)
    result = reader.render(_sample_schedule())
    assert result.image.ndim == 3
    assert result.image.shape[2] == 3
    assert result.image.shape[0] > 64
    assert "Cognitive Spectrogram" in result.report
    assert "Im" in result.report


def test_read_tuple_api():
    reader = SpectrogramReader(duration=0.5)
    image, report = reader.read(_sample_schedule("Linear"))
    assert isinstance(image, np.ndarray)
    assert "Dominant bands" in report or "Terminal" in report


def test_all_schedule_logics_smoke():
    reader = SpectrogramReader(duration=0.3)
    logics = [
        "Orbital",
        "Drag",
        "Stochastic Switching",
        "Adversarial",
        "Linear",
        "Cooperative Binding",
        "Feedback Loop",
        "Partial Adversarial",
        "Crossing Constraints",
        "Softmax Junction",
        "Amplified Binding",
        "Global Equilibrium",
        "Sequential Commitment",
    ]
    for logic in logics:
        sched = _sample_schedule(logic)
        if logic == "Global Equilibrium":
            sched["fold_energy"] = -8.0
        result = reader.render(sched)
        assert result.image.size > 0, f"Failed for {logic}"


def test_terminal_freq_ordering():
    assert TERMINAL_FREQS["Im"] < TERMINAL_FREQS["Df"]
    assert len(TERMINAL_FREQS) == 12


def test_resolve_all_objective_symbols():
    from attention_algebra.spectrum import _resolve_terminal
    from attention_algebra.terminals import TERMINAL_SPECS

    for spec in TERMINAL_SPECS:
        track = {"symbol": spec.objective, "voice": f"{spec.name} ({spec.symbol})"}
        assert _resolve_terminal(track) == spec.symbol
