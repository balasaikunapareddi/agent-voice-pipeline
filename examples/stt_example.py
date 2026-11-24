"""Standalone speech-to-text example — copy this file into your own project.

It transcribes an audio file with Faster-Whisper (tiny) on CPU. No
agent_voice_pipeline imports, no pipeline, no server: just STT.

Install
-------
    pip install faster-whisper numpy

    # first run downloads the tiny model (~75 MB) from Hugging Face

Use it from your own code (10 lines)
------------------------------------
    from stt_example import transcribe

    result = transcribe("hello.wav")
    print(result["text"])        # 'Hello, what time does the library open today?'
    print(result["latency_s"])   # seconds the transcription took
    print(result["segments"])    # [{"start": 0.0, "end": 3.0, "text": "..."}]

Run it
------
    python stt_example.py hello.wav
    python stt_example.py hello.wav --model base --json

What it returns
---------------
A dict: ``text`` (the transcript), ``language``, ``segments`` (word-
timed chunks), ``duration_s`` (audio length), ``latency_s`` (how long
the model took), ``backend`` and ``model`` (so you always know what
produced the number).

Swap the backend
----------------
Pass ``--model base`` (or ``small``) for a bigger, more accurate
model. For the OpenAI ``whisper`` package instead, replace the
``WhisperModel`` import with ``import whisper`` and call
``whisper.load_model("tiny").transcribe(path)`` — the return shape
above stays the same.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave

import numpy as np


def _read_wav_16k(path: str) -> np.ndarray:
    """Read a WAV file as mono float32 at 16 kHz (stdlib only)."""
    with wave.open(path, "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {width}")
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != 16000:  # linear resample
        t_old = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)
        t_new = np.arange(0, len(audio) / sr, 1 / 16000)
        audio = np.interp(t_new, t_old, audio).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def transcribe(path: str, model: str = "tiny", language: str = "en") -> dict:
    """Transcribe ``path`` with Faster-Whisper. Returns the result dict."""
    from faster_whisper import WhisperModel  # lazy: import only when used

    t0 = time.time()
    audio = _read_wav_16k(path)
    # NOTE: audio is passed as an array, not a file path — faster-whisper
    # 1.2.1 hands file paths to PyAV with a `metadata_errors` argument
    # that PyAV >= 13 rejects, so decoding it ourselves is more robust.
    whisper = WhisperModel(model, device="cpu", compute_type="int8")
    segments, info = whisper.transcribe(audio, language=language)
    out_segments, texts = [], []
    for s in segments:
        out_segments.append({"start": round(float(s.start), 3),
                             "end": round(float(s.end), 3),
                             "text": s.text.strip()})
        texts.append(s.text.strip())
    return {
        "text": " ".join(texts).strip(),
        "language": info.language or language,
        "segments": out_segments,
        "duration_s": round(len(audio) / 16000, 3),
        "latency_s": round(time.time() - t0, 3),
        "backend": "faster-whisper",
        "model": model,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("wav", help="audio file (WAV) to transcribe")
    parser.add_argument("--model", default="tiny", help="whisper model size")
    parser.add_argument("--language", default="en")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = transcribe(args.wav, model=args.model, language=args.language)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[{result['backend']}/{result['model']}] {result['text']}")
        print(f"({result['duration_s']:.2f}s audio, "
              f"transcribed in {result['latency_s']:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
