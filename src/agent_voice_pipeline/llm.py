"""Language model stage: streaming replies (and tool calls) for the agent.

Backends
--------
``gemini`` (default, live)
    LiteLLM chat completions; model from ``LLM_MODEL`` (default
    ``gemini/gemini-3.5-flash-lite``). Needs ``GEMINI_API_KEY`` (or the
    provider key for the chosen model) — the live test supplies it
    through the run_live.py harness, never committed.
``stub`` (offline tests only)
    Deterministic canned replies keyed on the transcript. Clearly
    labeled, never used for a live claim.
``llamacpp`` (opt-in, local)
    Talks to a llama.cpp server (``LLAMA_CPP_URL``, default
    http://127.0.0.1:8080) over its OpenAI-compatible ``/v1`` API.
    No model is downloaded by this repo — run your own GGUF server.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gemini/gemini-3.5-flash-lite")


class StubLLM:
    """Deterministic stand-in. Replies are canned templates so the
    offline suite and the file demo run with no key and no network."""

    name = "stub"

    def __init__(self, model: str = "stub"):
        self.model = model

    def reply(self, messages: list[dict], **kwargs) -> str:
        return "".join(self.stream_reply(messages, **kwargs))

    def stream_reply(self, messages: list[dict], **kwargs) -> Iterator[str]:
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break
        low = user_text.lower()
        if "library" in low:
            text = ("The library opens at 9 AM today and closes at 8 PM. "
                    "Is there anything else I can help with?")
        elif "hello" in low or "hi" in low:
            text = "Hello! How can I help you today?"
        elif not user_text.strip():
            text = "I didn't catch that — could you say it again?"
        else:
            text = f"You said: {user_text}. How can I help with that?"
        # Yield in word chunks so streaming behaviour is exercised.
        for word in text.split(" "):
            yield word + " "


class LiteLLMChat:
    """Live LLM via LiteLLM (streaming). Lazy import, provider-agnostic."""

    name = "gemini"

    def __init__(self, model: str | None = None, **kwargs):
        self.model = model or DEFAULT_MODEL
        # A stalled request must not hang forever: LiteLLM has no
        # default timeout, so set one unless the caller chose otherwise.
        kwargs.setdefault("timeout", 120)
        self.kwargs = kwargs

    def _call(self, messages: list[dict], stream: bool):
        import litellm
        return litellm.completion(
            model=self.model, messages=messages, stream=stream, **self.kwargs)

    def reply(self, messages: list[dict], **kwargs) -> str:
        resp = self._call(messages, stream=False)
        try:
            return resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return str(resp)

    def stream_reply(self, messages: list[dict], **kwargs) -> Iterator[str]:
        resp = self._call(messages, stream=True)
        for chunk in resp:
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError):
                delta = None
            if delta:
                yield delta


class LlamaCppLLM:
    """Opt-in local backend: a llama.cpp server's /v1 chat API.

    Start one yourself, e.g.::
        llama-server -m qwen3.5-4b.gguf --port 8080
    then set ``LLAMA_CPP_URL`` (default http://127.0.0.1:8080).
    No GGUF is downloaded by this repo.
    """

    name = "llamacpp"

    def __init__(self, base_url: str | None = None, model: str = "local"):
        self.base_url = (base_url or os.environ.get(
            "LLAMA_CPP_URL", "http://127.0.0.1:8080")).rstrip("/")
        self.model = model

    def _post(self, messages: list[dict], stream: bool):
        import json
        import urllib.request
        payload = json.dumps({
            "model": self.model, "messages": messages, "stream": stream,
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=120)

    def reply(self, messages: list[dict], **kwargs) -> str:
        import json
        with self._post(messages, stream=False) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"] or ""

    def stream_reply(self, messages: list[dict], **kwargs) -> Iterator[str]:
        import json
        with self._post(messages, stream=True) as r:
            for line in r:
                line = line.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (KeyError, IndexError, json.JSONDecodeError):
                    delta = None
                if delta:
                    yield delta


def make_llm(name: str, **kwargs):
    """Factory: ``gemini`` (default, live via LiteLLM), ``stub``,
    ``llamacpp``."""
    name = (name or "gemini").lower()
    if name in ("gemini", "litellm", "live"):
        return LiteLLMChat(**kwargs)
    if name == "stub":
        return StubLLM(**kwargs)
    if name == "llamacpp":
        return LlamaCppLLM(**kwargs)
    raise ValueError(f"unknown LLM backend: {name!r} (gemini, stub, llamacpp)")
