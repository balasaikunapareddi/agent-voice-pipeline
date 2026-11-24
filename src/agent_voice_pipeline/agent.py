"""The voice agent: owns the conversation, one turn at a time.

The agent is deliberately thin — the pipeline stages do the hearing
and speaking; the agent holds the history, the system prompt, and the
decision of what to say next.
"""

from __future__ import annotations

from collections.abc import Iterator

from .llm import make_llm
from .prompts import SYSTEM_PROMPT


class VoiceAgent:
    """Conversational agent over a voice call.

    ``llm`` is any object with ``reply(messages)`` / ``stream_reply`` —
    see ``llm.py``. History is a plain list of ``{role, content}``
    dicts, so the whole conversation is inspectable.
    """

    def __init__(self, llm=None, system_prompt: str | None = None):
        self.llm = llm if llm is not None else make_llm("stub")
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.history: list[dict] = [{"role": "system", "content": self.system_prompt}]

    @property
    def llm_name(self) -> str:
        return getattr(self.llm, "name", type(self.llm).__name__)

    def run_turn(self, transcript: str) -> str:
        """One user turn -> the agent's reply text (blocking)."""
        self.history.append({"role": "user", "content": transcript})
        reply = self.llm.reply(self.history).strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def stream_turn(self, transcript: str) -> Iterator[str]:
        """One user turn -> reply text streamed in chunks."""
        self.history.append({"role": "user", "content": transcript})
        parts: list[str] = []
        for chunk in self.llm.stream_reply(self.history):
            parts.append(chunk)
            yield chunk
        self.history.append({"role": "assistant", "content": "".join(parts).strip()})

    def reset(self) -> None:
        self.history = [{"role": "system", "content": self.system_prompt}]
