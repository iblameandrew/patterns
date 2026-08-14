"""Factory for chat-model clients used by the Attention Grammar pipeline.

Both back-ends speak the OpenAI Chat Completions API and are wired through
``langchain-openai`` (which wraps the official ``openai`` SDK):

* **OpenRouter** — cloud models via ``https://openrouter.ai/api/v1``.
  Requires ``OPENROUTER_API_KEY``.
* **llama.cpp server** — local inference via an OpenAI-compatible endpoint
  (e.g. ``llama-server --port 8080``).  Configure with ``LLAMA_CPP_BASE_URL``
  and optionally ``LLAMA_CPP_API_KEY`` (many local servers accept any string).
"""

import os
from typing import Literal

from langchain_openai import ChatOpenAI

Provider = Literal["openrouter", "llama.cpp"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080/v1"

OPENROUTER_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4",
    "qwen/qwen3-32b",
    "meta-llama/llama-3.3-70b-instruct",
]

DEFAULT_OPENROUTER_MODEL = OPENROUTER_MODELS[0]
DEFAULT_LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "local")


def chat_model_kwargs(
    model_name: str = DEFAULT_OPENROUTER_MODEL,
    *,
    provider: Provider = "openrouter",
    temperature: float = 0.7,
) -> dict:
    """Return constructor kwargs for ``ChatOpenAI`` (no network)."""
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Set it in your shell or in a .env file."
            )
        return {
            "model": model_name,
            "temperature": temperature,
            "openai_api_key": api_key,
            "openai_api_base": OPENROUTER_BASE_URL,
            "default_headers": {
                "HTTP-Referer": os.getenv(
                    "OPENROUTER_HTTP_REFERER",
                    "https://github.com/iblameandrew/attention-grammar",
                ),
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Attention Grammar"),
            },
        }

    if provider == "llama.cpp":
        return {
            "model": model_name or DEFAULT_LLAMA_CPP_MODEL,
            "temperature": temperature,
            "openai_api_key": os.getenv("LLAMA_CPP_API_KEY", "no-key"),
            "openai_api_base": os.getenv("LLAMA_CPP_BASE_URL", DEFAULT_LLAMA_CPP_BASE_URL),
        }

    raise ValueError(f"Unknown provider {provider!r}. Expected 'openrouter' or 'llama.cpp'.")


class ModelFactory:
    """Build chat-model clients on demand."""

    @staticmethod
    def get_model(
        model_name: str = DEFAULT_OPENROUTER_MODEL,
        *,
        provider: Provider = "openrouter",
        temperature: float = 0.7,
    ) -> ChatOpenAI:
        """Return a ``ChatOpenAI`` client for the given provider and model."""
        print(f"Initializing Model: {provider}/{model_name} (Temp: {temperature})")
        return ChatOpenAI(
            **chat_model_kwargs(model_name, provider=provider, temperature=temperature)
        )
