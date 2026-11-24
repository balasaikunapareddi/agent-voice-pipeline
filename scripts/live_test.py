"""Live test: the FULL pipeline end-to-end with real backends.

Run via the Gemini harness (injects GEMINI_API_KEY)::

    python3 ~/workspace/custom-connectors/custom.gemini/bin/run_live.py \\
        ~/workspace/agent-voice-pipeline .venv/bin/python scripts/live_test.py

Backends used here are the REAL ones:
  - VAD: energy (built-in) + Silero VAD v5 ONNX if the model file exists
  - STT: faster-whisper tiny on CPU (real transcription)
  - LLM: live Gemini (gemini-3.5-flash-lite) via LiteLLM
  - TTS: Piper en_US-lessac-medium if the voice exists, else stub (labeled)

Asserts:
  1. STT transcript is non-empty and matches the known script
     ("Hello, what time does the library open today?") at >= 75% word
     accuracy — the headline STT check.
  2. The live LLM reply is non-empty and on-topic (mentions the library
     or opening hours).
  3. The output WAV exists, is non-empty, and has duration > 0.
  4. Per-stage timings are reported.

Writes a JSON report to .runs/live/attempt-N/report.json and prints
"LIVE TEST PASSED" (exit 0) or raises (exit non-zero).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_voice_pipeline.audio import wav_info  # noqa: E402
from agent_voice_pipeline.pipeline import Pipeline  # noqa: E402
from agent_voice_pipeline.stt import word_accuracy  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELLO_WAV = os.path.join(ROOT, "examples", "hello.wav")
KNOWN_SCRIPT = "Hello, what time does the library open today?"
RUNS_DIR = os.path.join(ROOT, ".runs", "live")


def main() -> int:
    from agent_voice_pipeline.agent import VoiceAgent
    from agent_voice_pipeline.llm import make_llm
    from agent_voice_pipeline.stt import make_stt
    from agent_voice_pipeline.tts import PiperTTS, make_tts
    from agent_voice_pipeline.vad import SileroVAD, make_vad

    os.makedirs(RUNS_DIR, exist_ok=True)
    attempt = len([d for d in os.listdir(RUNS_DIR)
                   if d.startswith("attempt-")]) + 1
    attempt_dir = os.path.join(RUNS_DIR, f"attempt-{attempt}")
    os.makedirs(attempt_dir, exist_ok=True)
    out_wav = os.path.join(attempt_dir, "reply.wav")

    # --- Stage 1+2 first: the headline STT check, on its own. ---
    stt = make_stt("faster-whisper", model="tiny")
    stt_result = stt.transcribe_file(HELLO_WAV)
    acc = word_accuracy(stt_result.text, KNOWN_SCRIPT)
    print(f"[live] STT [{stt_result.backend}/{stt_result.model}]: "
          f"{stt_result.text!r}")
    print(f"[live] STT word accuracy vs known script: "
          f"{acc['correct']}/{acc['total']} = {acc['accuracy']:.2f} "
          f"(transcribed {stt_result.duration_s:.2f}s audio "
          f"in {stt_result.latency_s:.2f}s)")

    # --- Full pipeline with live Gemini + real TTS. ---
    silero_model = os.path.join(ROOT, "models", "silero_vad.onnx")
    vad_name = "silero" if SileroVAD.available(silero_model) else "energy"
    vad = make_vad(vad_name, model_path=silero_model) if vad_name == "silero" \
        else make_vad("energy")
    tts_name = "piper" if PiperTTS.available() else "stub"
    llm = make_llm("gemini")
    pipeline = Pipeline(vad=vad, stt=stt, llm=llm, tts=make_tts(tts_name),
                        agent=VoiceAgent(llm=llm))
    result = pipeline.process_file(HELLO_WAV, out_wav)
    print(f"[live] backends: {result.backends}")
    print(f"[live] VAD segments: {result.segments}")
    print(f"[live] transcript: {result.transcript!r}")
    print(f"[live] live LLM reply: {result.reply!r}")
    print(f"[live] timings (s): {result.timings_s}")

    info = wav_info(result.reply_wav)
    print(f"[live] reply WAV: {info['duration_s']:.2f}s, "
          f"sr={info['sample_rate']}, {os.path.getsize(result.reply_wav)} bytes")

    report = {
        "attempt": attempt,
        "stt": {**stt_result.to_dict(), "word_accuracy": acc},
        "known_script": KNOWN_SCRIPT,
        "pipeline": result.to_dict(),
        "reply_wav_info": info,
        "vad_backend": vad_name,
        "tts_backend": tts_name,
    }
    with open(os.path.join(attempt_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # --- Asserts. ---
    assert stt_result.text, "STT transcript is empty"
    assert acc["accuracy"] >= 0.75, (
        f"STT word accuracy {acc['accuracy']:.2f} < 0.75: "
        f"{stt_result.text!r} vs {KNOWN_SCRIPT!r}")
    assert result.reply.strip(), "live LLM reply is empty"
    low = result.reply.lower()
    assert "librar" in low or "open" in low or "hour" in low or "9" in low, (
        f"live LLM reply off-topic: {result.reply!r}")
    assert os.path.getsize(result.reply_wav) > 0, "reply WAV is empty"
    assert info["duration_s"] > 0, "reply WAV has zero duration"

    print("[live] LIVE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
