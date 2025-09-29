"""Utility helpers for interacting with the local Ollama models."""
from __future__ import annotations

from typing import List, Dict
import ollama


def chat_once(model: str, messages: List[Dict[str, str]], *, stream: bool = False) -> str:
    """Send a list of chat ``messages`` to ``model`` and return the assistant response.

    Parameters
    ----------
    model:
        Name of the Ollama model to invoke.
    messages:
        Conversation as a list of {"role", "content"} dictionaries.
    stream:
        Whether to request a streaming response. When ``True`` the helper
        aggregates the streamed chunks into a single string.
    """
    if stream:
        response_chunks = []
        for part in ollama.chat(model=model, messages=messages, stream=True):
            msg = part.get("message") or {}
            content = msg.get("content")
            if content:
                response_chunks.append(content)
        return "".join(response_chunks)

    response = ollama.chat(model=model, messages=messages, stream=False)
    message = response.get("message") or {}
    return message.get("content", "").strip()
