# Changelog

## 0.7.0 — 2026-08-14

- Injectable `llm=` on `AlgebraAnalyst`, `Composer`, and `ThoughtLibrarian` for offline tests.
- Fix Layer-1 prompt: literal `{...}` sets were parsed as PromptTemplate variables, so `analyze()` crashed.
- Fix Partial Adversarial spectrogram envelopes so they stay time-aligned arrays.
- `chat_model_kwargs()` builds llama.cpp / OpenRouter client config without opening a socket.
- Expanded offline pytest suite (parser, utils, config, algebra/composer, spectrum envelopes, thought library).
- `pyproject.toml`, multi-stage Docker image, compose, and GitHub Actions CI/release (GHCR).
- Gradio app honors `HOST` / `PORT` for containers.

## 0.6.2

Geometric terminals and hermetic thought library (see `RELEASE_NOTES_v0.6.0.md`).
