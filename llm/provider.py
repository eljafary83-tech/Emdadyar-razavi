"""Swappable LLM provider boundary."""
from __future__ import annotations
from typing import Protocol
from llm.client import ChatCompletionsClient

class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...

class GenericChatProvider:
    def __init__(self, client: ChatCompletionsClient) -> None: self.client = client
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.client.complete(system_prompt, user_prompt)
