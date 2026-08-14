from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM(Runnable):
    def __init__(self, reply: str = "(Im ~ Pr)") -> None:
        self.reply = reply
        self.prompts: list[Any] = []

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> FakeMessage:
        self.prompts.append(input)
        return FakeMessage(self.reply)
