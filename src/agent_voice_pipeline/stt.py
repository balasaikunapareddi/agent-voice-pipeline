"""Speech to Text — the reusable stage.

This module is deliberately self-contained: you can copy it (or
``examples/stt_example.py``) into your own project and transcribe audio
with three lines::

    from agent_voice_pipeline.stt import transcribe_file
    result = transcribe_file("hello.wav")          # real faster-whisper tiny
    print(result.text)  # 'Hello, what time does the library open today?'

It also runs on its own from the command line::

    python -m agent_voice_pipeline.stt --wav hello.wav
    python -m agent_voice_pipeline.stt --wav hello.wav --backend stub --json

Backends
--------
``faster-whisper`` (default, real)
    Faster-Whisper ``tiny`` on CPU (int8). ~75 MB model, downloaded
    automatically from the Hugging Face Hub on first use. On this
    sandbox it transcribes the 3 s ``examples/hello.wav`` exactly in
    ~0.8 s. Audio is decoded with the stdlib ``wave`` module, not
    PyAV — faster-whisper 1.2.1 passes a ``metadata_errors`` argument
    that PyAV >= 13 rejects, so file paths are never handed to it.
``stub`` (offline tests only)
    Returns a fixed canned transcript. Clearly labeled, never used
    for a live claim.
``whisper`` (opt-in)
    OpenAI's ``whisper`` package (torch). Not installed in the
    sandbox; import fails loudly with install instructions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .audio import TARGET_SR, read_audio

DEFAULT_MODEL = "tiny"
STUB_TRANSCRIPT = "Hello, this is a test of the voice pipeline."


@dataclass
class STTResult:
    """One transcription. ``backend`` names what actually produced it."""

    text: str
    language: str = "en"
    segments: list[dict] = field(default_factory=list)
    duration_s: float = 0.0
    latency_s: float = 0.0
    backend: str = "stub"
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "segments": self.segments,
            "duration_s": round(self.duration_s, 3),
            "latency_s": round(self.latency_s, 3),
            "backend": self.backend,
            "model": self.model,
        }


class StubSTT:
    """Offline stand-in. Returns a fixed transcript — for tests and
    for running the pipeline with no model at all. Never real."""

    name = "stub"

    def __init__(self, transcript: str = STUB_TRANSCRIPT):
        self.transcript = transcript

    def transcribe(self, samples: np.ndarray, sr: int = TARGET_SR,
                   language: str = "en") -> STTResult:
        t0 = time.time()
        duration = len(samples) / sr if sr else 0.0
        return STTResult(
            text=self.transcript,
            language=language,
            segments=[{"start": 0.0, "end": duration, "text": self.transcript}],
            duration_s=duration,
            latency_s=time.time() - t0,
            backend="stub",
            model="stub",
        )

    def transcribe_file(self, path: str, language: str = "en") -> STTResult:
        samples, sr = read_audio(path)
        return self.transcribe(samples, sr, language)


class FasterWhisperSTT:
    """Real STT: Faster-Whisper (default model ``tiny``) on CPU.

    Lazy import — importing this module never pulls in the model.
    The model downloads from the Hugging Face Hub on first use
    (~75 MB for tiny) and is then cached.
    """

    name = "faster-whisper"

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu",
                 compute_type: str = "int8"):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:  # pragma: no cover - missing dep
                raise RuntimeError(
                    "faster-whisper is not installed. Install it with: "
                    "pip install faster-whisper"
                ) from e
            self._model = WhisperModel(
                self.model_name, device=self.device,
                compute_type=self.compute_type)
        return self._model

    def transcribe(self, samples: np.ndarray, sr: int = TARGET_SR,
                   language: str = "en") -> STTResult:
        t0 = time.time()
        samples = np.asarray(samples, dtype=np.float32)
        if sr != TARGET_SR:
            from .audio import resample
            samples = resample(samples, sr, TARGET_SR)
        model = self._load()
        segments, info = model.transcribe(samples, language=language)
        out_segments = []
        texts = []
        for s in segments:
            out_segments.append({
                "start": round(float(s.start), 3),
                "end": round(float(s.end), 3),
                "text": s.text.strip(),
            })
            texts.append(s.text.strip())
        return STTResult(
            text=" ".join(texts).strip(),
            language=info.language or language,
            segments=out_segments,
            duration_s=len(samples) / TARGET_SR,
            latency_s=time.time() - t0,
            backend="faster-whisper",
            model=self.model_name,
        )

    def transcribe_file(self, path: str, language: str = "en") -> STTResult:
        samples, sr = read_audio(path)
        return self.transcribe(samples, sr, language)


class WhisperSTT:
    """Opt-in backend for OpenAI's ``whisper`` package (torch).

    Not installed in the sandbox. Import fails loudly with the
    install command instead of silently falling back.
    """

    name = "whisper"

    def __init__(self, model: str = "tiny"):
        self.model_name = model
        self._model = None

    def _load(self):  # pragma: no cover - needs torch + download
        if self._model is None:
            try:
                import whisper  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "openai-whisper is not installed. Install it with: "
                    "pip install openai-whisper (requires torch)"
                ) from e
            self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe(self, samples: np.ndarray, sr: int = TARGET_SR,
                   language: str = "en") -> STTResult:  # pragma: no cover
        raise NotImplementedError(
            "WhisperSTT file transcription goes through transcribe_file; "
            "the whisper package reads audio files itself."
        )

    def transcribe_file(self, path: str, language: str = "en") -> STTResult:  # pragma: no cover
        t0 = time.time()
        model = self._load()
        out = model.transcribe(path, language=language)
        segs = [{"start": round(float(s["start"]), 3),
                 "end": round(float(s["end"]), 3),
                 "text": s["text"].strip()} for s in out.get("segments", [])]
        return STTResult(
            text=out.get("text", "").strip(),
            language=out.get("language", language),
            segments=segs,
            duration_s=0.0,
            latency_s=time.time() - t0,
            backend="whisper",
            model=self.model_name,
        )


def make_stt(name: str, **kwargs):
    """Factory: ``faster-whisper`` (default, real), ``stub``, ``whisper``."""
    name = (name or "faster-whisper").lower()
    if name == "faster-whisper":
        return FasterWhisperSTT(**kwargs)
    if name == "stub":
        return StubSTT(**kwargs)
    if name == "whisper":
        return WhisperSTT(**kwargs)
    raise ValueError(
        f"unknown STT backend: {name!r} (faster-whisper, stub, whisper)")


def transcribe_file(path: str, backend: str = "faster-whisper",
                    language: str = "en", **kwargs) -> STTResult:
    """One call: transcribe an audio file. The reusable entry point.

    Example::

        from agent_voice_pipeline.stt import transcribe_file
        result = transcribe_file("examples/hello.wav")
        print(result.text)
    """
    stt = make_stt(backend, **kwargs)
    return stt.transcribe_file(path, language=language)


def word_accuracy(hypothesis: str, reference: str) -> dict:
    """Case-insensitive word accuracy of ``hypothesis`` vs ``reference``.

    Tokens are lowercased and stripped of surrounding punctuation, so
    "Hello," in a transcript matches "Hello" in the script. Returns
    ``{correct, total, accuracy, missing, extra}`` — the number the
    live test reports for the shipped test WAV.
    """
    def norm(text: str) -> list[str]:
        import string
        return [w.strip(string.punctuation) for w in text.lower().split()
                if w.strip(string.punctuation)]

    ref = norm(reference)
    hyp = norm(hypothesis)
    # Longest-common-subsequence count, small inputs only.
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = (dp[i][j] + 1 if ref[i] == hyp[j]
                                else max(dp[i][j + 1], dp[i + 1][j]))
    correct = dp[m][n]
    # Multiset diffs, so a repeated word counts once per occurrence:
    # an earlier version used set membership, which reported no missing
    # word when the hypothesis had one "the" but the reference had two.
    from collections import Counter
    ref_counts, hyp_counts = Counter(ref), Counter(hyp)
    missing = list((ref_counts - hyp_counts).elements())
    extra = list((hyp_counts - ref_counts).elements())
    return {
        "correct": correct,
        "total": m,
        "accuracy": (correct / m) if m else 1.0,
        "missing": missing,
        "extra": extra,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m agent_voice_pipeline.stt --wav <file>``."""
    parser = argparse.ArgumentParser(
        prog="agent_voice_pipeline.stt",
        description="Transcribe an audio file (standalone STT example).")
    parser.add_argument("--wav", required=True, help="audio file to transcribe")
    parser.add_argument("--backend", default="faster-whisper",
                        choices=["faster-whisper", "stub", "whisper"],
                        help="STT backend (default: faster-whisper)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="model size for faster-whisper (default: tiny)")
    parser.add_argument("--language", default="en")
    parser.add_argument("--json", action="store_true",
                        help="print the full result as JSON")
    args = parser.parse_args(argv)
    kwargs = {"model": args.model} if args.backend != "stub" else {}
    result = transcribe_file(args.wav, backend=args.backend,
                             language=args.language, **kwargs)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"[{result.backend}/{result.model}] {result.text}")
        print(f"({result.duration_s:.2f}s audio, "
              f"transcribed in {result.latency_s:.2f}s, lang={result.language})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
