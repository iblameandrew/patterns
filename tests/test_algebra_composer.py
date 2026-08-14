import json

from attention_algebra.algebra import AlgebraAnalyst
from attention_algebra.composition import Composer, _equation_for_logic, _extract_json
from tests.fakes import FakeLLM


def test_analyst_returns_llm_content():
    llm = FakeLLM("(Im ~ Pr)")
    analyst = AlgebraAnalyst(llm=llm)
    assert analyst.analyze("sudden impulse with sparse focus") == "(Im ~ Pr)"
    assert llm.prompts


def test_composer_parses_json():
    payload = {
        "original_expression": "(Im ~ Pr)",
        "structure_type": "flat",
        "schedule_logic": "Orbital",
        "global_frequency": 2.0,
        "fold_energy": None,
        "nodes": [],
        "score": [
            {
                "voice": "Impulse (Im)",
                "symbol": "KineticVelocityObjective",
                "mass": 7,
                "formula": "||z||",
                "description": "v",
                "role": "primary",
            }
        ],
        "math_narrative": "orbital",
    }
    llm = FakeLLM("```json\n" + json.dumps(payload) + "\n```")
    composer = Composer(llm=llm)
    result = composer.compose("(Im ~ Pr)")
    assert result["schedule_logic"] == "Orbital"
    assert result["score"][0]["mass"] == 7
    report = Composer.format_latex_report(result)
    assert "Cognitive Schedule" in report
    assert "Impulse" in report


def test_composer_invalid_json_returns_error_dict():
    composer = Composer(llm=FakeLLM("not json at all"))
    result = composer.compose("Im")
    assert result["score"] == []
    assert "Error" in result["math_narrative"]
    report = Composer.format_latex_report(result)
    assert "Raw model output" in report


def test_extract_json_plain_and_fenced():
    assert _extract_json('{"a":1}') == '{"a":1}'
    assert _extract_json('```json\n{"a":1}\n```') == '{"a":1}'


def test_equations_cover_logics():
    score = [
        {"mass": 1, "formula": "f", "role": "primary"},
        {"mass": 2, "formula": "g", "role": "loop"},
    ]
    for logic in (
        "Orbital",
        "Drag",
        "Adversarial",
        "Cooperative Binding",
        "Feedback Loop",
        "Partial Adversarial",
        "Crossing Constraints",
        "Softmax Junction",
        "Amplified Binding",
        "Global Equilibrium",
        "Sequential Commitment",
        "Linear",
    ):
        eq = _equation_for_logic(logic, score, 1.0)
        assert "$$" in eq
