"""Shared helpers used across the Attention Grammar pipeline."""

import re

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and their contents from ``text``.

    Returns an empty string when ``text`` is falsy so callers can pass the
    result directly into template-rendering code without further checks.
    """
    if not text:
        return ""
    return _THINK_TAG_RE.sub("", text)


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences (``​```​``) wrapping a code block.

    The helper is intentionally tolerant: it strips a single leading
    ``​```<lang>​`` (optional language tag) and the trailing ``​```​``.
    Any language tag is preserved in the resulting string for callers that
    care about it; pass the result through ``re``/AST parsing if you only
    want the code body.
    """
    if not text:
        return ""

    stripped = text.strip()

    fence_open_re = re.compile(r"^```([a-zA-Z0-9_+\-]*)\s*\n")
    match = fence_open_re.match(stripped)
    if match:
        stripped = stripped[match.end() :]

    if stripped.endswith("```"):
        stripped = stripped[:-3]

    return stripped.strip()
