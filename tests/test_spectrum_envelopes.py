import numpy as np

from attention_algebra.spectrum import SpectrogramReader, _envelopes, _normalize_mass


def test_normalize_mass():
    assert _normalize_mass(10) == 1.0
    assert _normalize_mass(-3) == 0.0


def test_envelopes_empty():
    t = np.linspace(0, 1, 8)
    assert _envelopes("Orbital", [], t, 1.0, None) == []


def test_envelopes_each_logic():
    t = np.linspace(0, 1, 16)
    masses = [0.7, 0.5]
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
        envs = _envelopes(logic, masses, t, 2.0, -4.0)
        assert len(envs) == 2, logic
        assert all(len(e) == len(t) for e in envs)


def test_render_empty_score():
    reader = SpectrogramReader(duration=0.2)
    result = reader.render(
        {
            "original_expression": "",
            "schedule_logic": "Linear",
            "global_frequency": 1.0,
            "fold_energy": None,
            "score": [],
            "math_narrative": "",
        }
    )
    assert result.image.ndim == 3
