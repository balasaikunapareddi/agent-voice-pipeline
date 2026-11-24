# agent-voice-pipeline

**Start with the speech-to-text — it's the reusable piece.** One command
transcribes any audio file with a real model, no pipeline, no server,
no API key:

```bash
pip install faster-whisper numpy
python examples/stt_example.py examples/hello.wav
# [faster-whisper/tiny] Hello, what time does the library open today?
# (3.05s audio, transcribed in 2.5s)
```

Or from your own code (copy `examples/stt_example.py` into your project):

```python
from stt_example import transcribe

result = transcribe("hello.wav")
print(result["text"])        # 'Hello, what time does the library open today?'
print(result["latency_s"])   # seconds the transcription took
print(result["segments"])    # [{"start": 0.0, "end": 3.0, "text": "..."}]
```

Same thing through the package, with a swappable backend:

```bash
python -m agent_voice_pipeline.stt --wav hello.wav            # faster-whisper tiny (real)
python -m agent_voice_pipeline.stt --wav hello.wav --model base # bigger model
python -m agent_voice_pipeline.stt --wav hello.wav --backend stub --json
```

```python
from agent_voice_pipeline.stt import transcribe_file
result = transcribe_file("hello.wav")   # real faster-whisper tiny by default
print(result.text)
```

The shipped test file `examples/hello.wav` ("Hello, what time does the
library open today?", generated locally with Piper TTS) transcribes at
**8/8 words correct (1.00 word accuracy)** with faster-whisper tiny on
CPU — verified live, see [Live test](#live-test).

---

## The problem this solves

Voice agents usually mean cloud APIs, API keys, and per-minute bills —
and every stage is a black box you can't swap. The upstream project
this is built in the spirit of, [HuggingFace speech-to-speech](https://github.com/huggingface/speech-to-speech)
(Apache 2.0), shows the other way: a voice agent as four swappable
stages, each in its own thread connected by queues, running entirely
on your own machine.

This repo is a small, readable version of that idea:

1. **Voice Activity Detection** — finds the speech, decides turn-taking.
2. **Speech to Text** — transcribes the turn (see above).
3. **Language Model** — replies, streaming text.
4. **Text to Speech** — speaks the reply back.

Every stage is swappable by name, in a YAML config or on the CLI, and
the server speaks the OpenAI Realtime protocol, so any Realtime client
can point at it by changing one URL.

## One-command demo (file-based, no microphone)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,stt,tts,vad]"
python -m agent_voice_pipeline demo --wav examples/hello.wav
```

```
backends: {'vad': 'energy', 'stt': 'faster-whisper', 'llm': 'stub', 'tts': 'piper'}

VAD segments: [(0.06, 2.85)]
You said: Hello, what time does the library open today?
Agent: The library opens at 9 AM today and closes at 8 PM. Is there anything else I can help with?
Reply audio: /tmp/agent-voice-pipeline-reply.wav
Timings (s): vad=0.24, stt=2.81, llm=0.00, tts=1.48
```

Flags: `--vad energy|silero --stt faster-whisper|stub|whisper
--llm stub|gemini|llamacpp --tts piper|stub|kokoro`, or `--config
configs/default.yaml`. With `--llm gemini` and `GEMINI_API_KEY` set,
the reply comes from a live model instead of the stub.

## The four stages

| Stage | Backends | Default | Notes |
|---|---|---|---|
| VAD | `energy`, `silero` | `energy` | Energy: deterministic, no model. Silero: real Silero VAD v5/v6 ONNX via onnxruntime — no torch. Needs `models/silero_vad.onnx` (2.2 MB). |
| STT | `faster-whisper`, `stub`, `whisper` | `faster-whisper` | Real: Faster-Whisper `tiny` on CPU (int8), ~75 MB auto-download. Stub: canned transcript, tests only. |
| LLM | `gemini`, `stub`, `llamacpp` | `stub` | Gemini: LiteLLM, model from `LLM_MODEL` (default `gemini/gemini-3.5-flash-lite`). llama.cpp: point at your own server, no GGUF downloaded here. |
| TTS | `piper`, `stub`, `kokoro` | `piper` | Real: Piper ONNX, `en_US-lessac-medium` voice (60 MB). Stub: sine-tone WAV, tests only. Kokoro: opt-in, needs torch (not installed here). |

The system prompt lives in `src/agent_voice_pipeline/prompts.py` — visible
and inspectable, never buried inline. The agent class is `VoiceAgent`
(`agent.py`); the wiring is `Pipeline` (`pipeline.py`).

### Model downloads

```bash
# Silero VAD (2.2 MB)
curl -L -o models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
# Piper voice (60 MB + config)
curl -L -o models/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o models/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

`models/` is gitignored — download, don't commit. `faster-whisper
tiny` downloads itself on first use (~75 MB, Hugging Face cache).

## Realtime server

```bash
python -m agent_voice_pipeline serve                 # ws://127.0.0.1:8765/v1/realtime
python -m agent_voice_pipeline serve --transport sse # HTTP+SSE fallback, same events
```

Implemented events — this is the honest surface, nothing more:

- Client → server: `session.update`, `input_audio_buffer.append`,
  `input_audio_buffer.commit`, `input_audio_buffer.clear`,
  `response.create`, `conversation.item.create`
- Server → client: `session.created` / `session.updated`,
  `response.created`, `response.output_item.added`,
  `response.text.delta` / `response.text.done`,
  `response.audio.delta` / `response.audio.done`,
  `response.audio_transcript.done`, `response.done`, `error`

Audio in is 24 kHz PCM16 base64 (the Realtime convention), resampled
to 16 kHz. Not implemented: WebRTC transport, tool calls over the
socket, server-side VAD events (`input_audio_buffer.speech_started`
etc. — turn-taking here is commit-driven), interruption handling.

## Live test

Run with the Gemini harness (injects the key; never committed):

```bash
python3 ~/workspace/custom-connectors/custom.gemini/bin/run_live.py \
  ~/workspace/agent-voice-pipeline .venv/bin/python scripts/live_test.py
```

Result (run firsthand, 2026-11-24):

```
[live] STT [faster-whisper/tiny]: 'Hello, what time does the library open today?'
[live] STT word accuracy vs known script: 8/8 = 1.00 (transcribed 3.05s audio in 2.89s)
[live] backends: {'vad': 'silero', 'stt': 'faster-whisper', 'llm': 'gemini', 'tts': 'piper'}
[live] VAD segments: [(0.064, 2.88)]
[live] transcript: 'Hello, what time does the library open today?'
[live] live LLM reply: "The library opens at nine o'clock this morning."
[live] timings (s): vad=0.07, stt=0.70, llm=10.25, tts=1.04
[live] reply WAV: 2.66s, sr=22050, 117292 bytes
[live] LIVE TEST PASSED
```

Offline suite: **67 tests pass** (`pytest tests/ -q`), including the
real Silero VAD and real faster-whisper checks. The WebSocket server
was smoke-tested over a real socket (full event flow, commit → audio
deltas → `response.done`).

## Honest limits

- **No microphone in the sandbox**, so the demo is file-based
  (`--wav`). The server takes audio over the socket; a mic client is
  not bundled.
- **Real STT/TTS need CPU time and downloads**: faster-whisper tiny
  ~75 MB, Piper voice ~60 MB, Silero model ~2.2 MB. First runs are
  slow; later runs are cached.
- **Silero VAD via ONNX, not torch**: the `silero-vad` package pulls a
  ~539 MB torch wheel whose download failed repeatedly in this sandbox,
  so the real Silero ONNX graph is driven directly with onnxruntime
  instead. One subtlety found while doing that: the v6 graph needs the
  previous 64 samples prepended as context to each 512-sample window —
  without it, probabilities sit near zero. That is implemented in
  `vad.py` and matches the upstream `OnnxWrapper`.
- **faster-whisper + PyAV**: faster-whisper 1.2.1 hands file paths to
  PyAV with a `metadata_errors` argument that PyAV ≥ 13 rejects, so
  this repo decodes audio itself (stdlib `wave` + numpy) and passes
  arrays to the model.
- **Kokoro and the `whisper` package need torch** — not installed here;
  they fail loudly with install instructions instead of silently
  falling back.
- The LLM reply in the live test (10.25 s) is Gemini free-tier latency,
  not the pipeline's — the local stages run in ~2 s total.

## Layout

```
src/agent_voice_pipeline/
  audio.py      WAV I/O, resampling, synthetic test audio
  vad.py        EnergyVAD + SileroVAD (ONNX)
  stt.py        STT backends + `python -m agent_voice_pipeline.stt`
  llm.py        LiteLLM / stub / llama.cpp backends
  tts.py        Piper / stub / Kokoro backends
  prompts.py    system prompt (separate file, on purpose)
  agent.py      VoiceAgent — owns the conversation
  pipeline.py   Pipeline — four stages, threads + queues
  server.py     OpenAI Realtime events + WS/SSE servers
  cli.py        demo / serve / stt / list-backends
examples/
  hello.wav         test utterance (Piper-generated)
  stt_example.py    copy-pasteable standalone STT
configs/default.yaml
scripts/live_test.py
tests/              67 offline tests
```

## License

MIT — see `LICENSE`. Upstream inspiration: HuggingFace
[speech-to-speech](https://github.com/huggingface/speech-to-speech)
(Apache 2.0). Silero VAD is MIT; Piper voices are MIT/CC; Faster-Whisper
is MIT.
