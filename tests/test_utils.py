from attention_algebra.utils import strip_code_fences, strip_think_tags


def test_strip_think_tags():
    assert strip_think_tags("") == ""
    assert strip_think_tags("Hello <think>secret</think> world") == "Hello  world"


def test_strip_code_fences():
    assert strip_code_fences("") == ""
    fenced = '```json\n{"a": 1}\n```'
    assert strip_code_fences(fenced) == '{"a": 1}'
    assert strip_code_fences("plain") == "plain"
