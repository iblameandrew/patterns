import pytest

from attention_algebra.config import (
    DEFAULT_LLAMA_CPP_BASE_URL,
    chat_model_kwargs,
)


def test_openrouter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        chat_model_kwargs("google/gemini-2.5-flash", provider="openrouter")


def test_openrouter_kwargs(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    kw = chat_model_kwargs("google/gemini-2.5-flash", provider="openrouter")
    assert kw["openai_api_base"].startswith("https://openrouter.ai")
    assert kw["openai_api_key"] == "sk-or-test"
    assert kw["model"] == "google/gemini-2.5-flash"


def test_llamacpp_kwargs(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    kw = chat_model_kwargs("local", provider="llama.cpp")
    assert kw["openai_api_base"] == DEFAULT_LLAMA_CPP_BASE_URL
    assert kw["openai_api_key"] == "no-key"
    assert kw["model"] == "local"


def test_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        chat_model_kwargs("x", provider="ollama")  # type: ignore[arg-type]
