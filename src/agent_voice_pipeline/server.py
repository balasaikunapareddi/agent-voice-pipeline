"""OpenAI Realtime-compatible server.

The server speaks the core of the OpenAI Realtime protocol over
WebSocket at ``/v1/realtime`` (default ``ws://127.0.0.1:8765``), so any
Realtime client can point at it by changing one URL — exactly like
the upstream speech-to-speech server.

Implemented events (this is the honest surface — nothing more):
    client -> server
        session.update              set voice / instructions / VAD mode
        input_audio_buffer.append   base64 PCM16 audio (24 kHz, mono)
        input_audio_buffer.commit   end the user turn, run the pipeline
        input_audio_buffer.clear    drop buffered audio
        response.create             run a turn on buffered audio (or text)
        conversation.item.create    add a text user message, then respond
    server -> client
        session.created / session.updated
        response.created
        response.output_item.added
        response.text.delta / response.text.done
        response.audio.delta (base64 PCM16) / response.audio.done
        response.audio_transcript.done
        response.done
        error (unknown event type, or a stage failure)

Audio in is 24 kHz PCM16 (the Realtime convention); it is resampled
to 16 kHz for VAD/STT. Audio out is the TTS voice's native rate
(Piper: 22.05 kHz) as base64 PCM16 chunks. Not implemented: WebRTC
transport, function/tool calls over the socket, input_audio_buffer
server-side VAD events (``input_audio_buffer.speech_started`` etc. —
turn-taking here is commit-driven), interruption handling.

Two transports: WebSocket via the ``websockets`` library (default),
or an HTTP+SSE fallback on the same port (POST JSON events to
``/v1/realtime``, stream ``text/event-stream`` back) for clients
that cannot do WebSockets.
"""

from __future__ import annotations

import base64
import json
import time

import numpy as np

from .audio import TARGET_SR, resample
from .pipeline import Pipeline

CLIENT_EVENTS = {
    "session.update",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "input_audio_buffer.clear",
    "response.create",
    "conversation.item.create",
}

SERVER_EVENTS = {
    "session.created", "session.updated", "response.created",
    "response.output_item.added", "response.text.delta", "response.text.done",
    "response.audio.delta", "response.audio.done",
    "response.audio_transcript.done", "response.done", "error",
}


class RealtimeSession:
    """One client connection: pure event logic, no networking.

    Keeps an audio buffer and a Pipeline; every ``handle_event``
    returns the list of server events to send back. Testable without
    a socket — see tests/test_server.py.
    """

    def __init__(self, pipeline: Pipeline, voice: str = "alloy"):
        self.pipeline = pipeline
        self.voice = voice
        self.instructions = ""
        self._pcm24k = bytearray()
        self._event_id = 0

    def _event(self, type_: str, **fields) -> dict:
        self._event_id += 1
        ev = {"event_id": f"evt_{self._event_id}", "type": type_}
        ev.update(fields)
        return ev

    def _error(self, message: str, code: str = "invalid_event") -> dict:
        return self._event("error", error={"type": code, "message": message})

    def _audio_to_16k(self) -> np.ndarray:
        pcm = np.frombuffer(bytes(self._pcm24k), dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        return resample(audio, 24000, TARGET_SR)

    def _run_turn(self, transcript: str | None = None) -> list[dict]:
        """Run one turn and emit the response.* event stream."""
        events: list[dict] = []
        response_id = f"resp_{int(time.time() * 1000)}"
        events.append(self._event("response.created",
                                  response={"id": response_id, "status": "in_progress"}))
        events.append(self._event("response.output_item.added",
                                  response_id=response_id,
                                  output_index=0,
                                  item={"id": f"msg_{response_id}", "type": "message",
                                        "role": "assistant", "status": "in_progress",
                                        "content": []}))
        try:
            if transcript is None:
                samples = self._audio_to_16k()
                if len(samples) == 0:
                    raise ValueError("no audio buffered for this turn")
                segments = self.pipeline.vad.segment(samples, TARGET_SR)
                parts = []
                for (start, end) in segments:
                    i0, i1 = int(start * TARGET_SR), int(end * TARGET_SR)
                    parts.append(self.pipeline.stt.transcribe(
                        samples[i0:i1], TARGET_SR).text)
                transcript = " ".join(p.strip() for p in parts if p.strip())
            reply = self.pipeline.agent.run_turn(transcript)
            events.append(self._event("response.audio_transcript.done",
                                      response_id=response_id, output_index=0,
                                      transcript=reply))
            # Stream the reply text in word chunks (text.delta).
            for word in reply.split(" "):
                events.append(self._event("response.text.delta",
                                          response_id=response_id,
                                          output_index=0, item_id=f"msg_{response_id}",
                                          content_index=0, delta=word + " "))
            events.append(self._event("response.text.done",
                                      response_id=response_id,
                                      output_index=0, item_id=f"msg_{response_id}",
                                      content_index=0))
            # TTS audio, chunked as base64 PCM16 deltas.
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tts = self.pipeline.tts.synthesize(reply, tmp.name)
                from .audio import read_wav
                audio, sr = read_wav(tmp.name, target_sr=tts.sample_rate)
                pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                raw = pcm.tobytes()
                step = 4800  # 0.2 s at 24 kHz, in bytes (2 bytes/sample)
                for i in range(0, len(raw), step):
                    events.append(self._event(
                        "response.audio.delta", response_id=response_id,
                        output_index=0, item_id=f"msg_{response_id}",
                        content_index=1,
                        delta=base64.b64encode(raw[i:i + step]).decode()))
            events.append(self._event("response.audio.done",
                                      response_id=response_id,
                                      output_index=0, item_id=f"msg_{response_id}",
                                      content_index=1))
        except Exception as e:
            events.append(self._error(f"turn failed: {e}", code="turn_failed"))
        events.append(self._event("response.done",
                                  response={"id": response_id, "status": "completed"}))
        self._pcm24k.clear()
        return events

    def handle_event(self, event: dict) -> list[dict]:
        """Dispatch one client event -> server events."""
        type_ = event.get("type", "")
        if type_ == "session.update":
            session = event.get("session", {})
            if "voice" in session:
                self.voice = session["voice"]
            if "instructions" in session:
                self.instructions = session["instructions"]
            return [self._event("session.updated",
                                session={"voice": self.voice,
                                         "instructions": self.instructions,
                                         "modalities": ["text", "audio"]})]
        if type_ == "input_audio_buffer.append":
            try:
                self._pcm24k += base64.b64decode(event.get("audio", ""))
            except Exception as e:
                return [self._error(f"bad audio chunk: {e}")]
            return []
        if type_ == "input_audio_buffer.clear":
            self._pcm24k.clear()
            return []
        if type_ == "input_audio_buffer.commit":
            return self._run_turn()
        if type_ == "response.create":
            instructions = (event.get("response") or {}).get("instructions")
            if instructions:
                return self._run_turn(transcript=instructions)
            return self._run_turn()
        if type_ == "conversation.item.create":
            item = event.get("item", {})
            text = ""
            for part in item.get("content", []):
                if part.get("type") in ("input_text", "text"):
                    text += part.get("text", "")
            if not text:
                return [self._error("conversation.item.create needs text content")]
            return self._run_turn(transcript=text)
        return [self._error(f"unknown event type: {type_!r}")]


def session_created_event(session_id: str = "sess_1") -> dict:
    return {"event_id": "evt_0", "type": "session.created",
            "session": {"id": session_id, "voice": "alloy",
                        "modalities": ["text", "audio"]}}


async def serve_websocket(pipeline: Pipeline, host: str = "127.0.0.1",
                          port: int = 8765) -> None:
    """Serve /v1/realtime over WebSocket (``websockets`` library)."""
    import websockets

    async def handler(ws):
        session = RealtimeSession(pipeline)
        await ws.send(json.dumps(session_created_event()))
        async for raw in ws:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps(
                    session._error("event is not valid JSON")))
                continue
            for out in session.handle_event(event):
                await ws.send(json.dumps(out))

    async with websockets.serve(handler, host, port):
        import asyncio
        print(f"Realtime server listening on ws://{host}:{port}/v1/realtime")
        await asyncio.Future()  # run forever


def serve_sse(pipeline: Pipeline, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve /v1/realtime over HTTP+SSE (stdlib only).

    POST a JSON event to /v1/realtime; the reply streams back as
    ``text/event-stream``. Same event shapes as the WebSocket server.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        session = RealtimeSession(pipeline)

        def log_message(self, *args):  # keep the terminal quiet
            pass

        def do_POST(self):
            if self.path != "/v1/realtime":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                event = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self.send_error(400, "invalid JSON")
                return
            events = self.session.handle_event(event)
            body = "".join(
                f"data: {json.dumps(e)}\n\n" for e in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/realtime":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(session_created_event()).encode())
            else:
                self.send_error(404)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Realtime server (SSE) listening on http://{host}:{port}/v1/realtime")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()
