"""Realtime server: event flow, in-process (no socket needed)."""

import base64

import numpy as np
import pytest

from agent_voice_pipeline.agent import VoiceAgent
from agent_voice_pipeline.audio import synth_bursts, synth_tone
from agent_voice_pipeline.llm import StubLLM
from agent_voice_pipeline.pipeline import Pipeline
from agent_voice_pipeline.server import (
    CLIENT_EVENTS,
    SERVER_EVENTS,
    RealtimeSession,
    session_created_event,
)
from agent_voice_pipeline.stt import StubSTT
from agent_voice_pipeline.tts import StubTTS
from agent_voice_pipeline.vad import EnergyVAD


def _session():
    pipe = Pipeline(vad=EnergyVAD(), stt=StubSTT(), llm=StubLLM(),
                    tts=StubTTS(), agent=VoiceAgent(llm=StubLLM()))
    return RealtimeSession(pipe)


def _pcm16_base64(samples: np.ndarray) -> str:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode()


def test_session_created_shape():
    ev = session_created_event()
    assert ev["type"] == "session.created"
    assert ev["session"]["modalities"] == ["text", "audio"]


def test_event_sets_are_documented():
    assert "session.update" in CLIENT_EVENTS
    assert "input_audio_buffer.commit" in CLIENT_EVENTS
    assert "response.create" in CLIENT_EVENTS
    assert "response.audio.delta" in SERVER_EVENTS
    assert "response.done" in SERVER_EVENTS


def test_session_update_round_trip():
    s = _session()
    out = s.handle_event({"type": "session.update",
                          "session": {"voice": "echo", "instructions": "be brief"}})
    assert out[0]["type"] == "session.updated"
    assert s.voice == "echo"
    assert s.instructions == "be brief"


def test_unknown_event_is_error():
    s = _session()
    out = s.handle_event({"type": "teleport.now"})
    assert out[0]["type"] == "error"
    assert "teleport" in out[0]["error"]["message"]


def test_audio_commit_runs_full_turn():
    s = _session()
    # 1 s of 24 kHz tone as if spoken; energy VAD will find it.
    tone24 = synth_tone(440.0, 1.0, sr=24000)
    assert s.handle_event({"type": "input_audio_buffer.append",
                           "audio": _pcm16_base64(tone24)}) == []
    out = s.handle_event({"type": "input_audio_buffer.commit"})
    types = [e["type"] for e in out]
    assert "response.created" in types
    assert "response.audio_transcript.done" in types
    assert "response.text.done" in types
    assert "response.audio.delta" in types
    assert "response.audio.done" in types
    assert "response.done" in types
    # Text deltas concatenate to the stub reply.
    text = "".join(e.get("delta", "") for e in out
                   if e["type"] == "response.text.delta")
    assert text.strip()


def test_audio_buffer_clear_drops_audio():
    s = _session()
    tone24 = synth_tone(440.0, 1.0, sr=24000)
    s.handle_event({"type": "input_audio_buffer.append",
                    "audio": _pcm16_base64(tone24)})
    s.handle_event({"type": "input_audio_buffer.clear"})
    out = s.handle_event({"type": "input_audio_buffer.commit"})
    types = [e["type"] for e in out]
    assert "error" in types  # no audio buffered


def test_conversation_item_create_text_turn():
    s = _session()
    out = s.handle_event({
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user",
                 "content": [{"type": "input_text",
                              "text": "What time does the library open?"}]}})
    types = [e["type"] for e in out]
    assert "response.done" in types
    transcript = [e for e in out if e["type"] == "response.audio_transcript.done"]
    assert transcript and "9 AM" in transcript[0]["transcript"]


def test_response_create_with_instructions():
    s = _session()
    out = s.handle_event({"type": "response.create",
                          "response": {"instructions": "Say hello briefly."}})
    assert any(e["type"] == "response.done" for e in out)


def test_bad_audio_chunk_is_error():
    s = _session()
    out = s.handle_event({"type": "input_audio_buffer.append",
                          "audio": "!!!not-base64!!!"})
    assert out[0]["type"] == "error"


def test_audio_deltas_are_valid_base64():
    s = _session()
    tone24 = synth_tone(440.0, 1.0, sr=24000)
    s.handle_event({"type": "input_audio_buffer.append",
                    "audio": _pcm16_base64(tone24)})
    out = s.handle_event({"type": "input_audio_buffer.commit"})
    deltas = [e["delta"] for e in out if e["type"] == "response.audio.delta"]
    assert deltas
    raw = b"".join(base64.b64decode(d) for d in deltas)
    assert len(raw) % 2 == 0 and len(raw) > 0


def test_failed_turn_reports_failed_not_completed():
    """Review fix: a failed turn emitted an error event but response.done
    still said 'completed' — a contradiction. It now says 'failed'."""
    s = _session()
    out = s.handle_event({"type": "input_audio_buffer.commit"})  # no audio buffered
    types = [e["type"] for e in out]
    assert "error" in types
    done = [e for e in out if e["type"] == "response.done"]
    assert done and done[0]["response"]["status"] == "failed"
    # A successful turn still reports completed.
    tone24 = synth_tone(440.0, 1.0, sr=24000)
    s.handle_event({"type": "input_audio_buffer.append",
                    "audio": _pcm16_base64(tone24)})
    out = s.handle_event({"type": "input_audio_buffer.commit"})
    done = [e for e in out if e["type"] == "response.done"]
    assert done and done[0]["response"]["status"] == "completed"


def test_sse_server_smoke_over_http():
    """The SSE transport serves real HTTP: POST an event, get SSE back.
    Exercises the shared-session + lock path in serve_sse."""
    import json
    import socket
    import threading
    import urllib.request
    from agent_voice_pipeline.server import serve_sse
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    t = threading.Thread(target=serve_sse, args=(_session().pipeline,),
                         kwargs={"host": "127.0.0.1", "port": port}, daemon=True)
    t.start()
    import time
    deadline = time.time() + 10
    body = None
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/realtime",
                data=json.dumps({"type": "session.update",
                                 "session": {"voice": "echo"}}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2) as r:
                body = r.read().decode()
            break
        except Exception:
            time.sleep(0.2)
    assert body is not None, "SSE server did not answer"
    assert "session.updated" in body
    assert "echo" in body
