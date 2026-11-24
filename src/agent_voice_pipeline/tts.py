"""Text to Speech: turn the agent's reply into audio.

Backends
--------
``piper`` (real, default)
    Piper TTS (ONNX, CPU) with the ``en_US-lessac-medium`` voice
    (``models/en_US-lessac-medium.onnx``, 60 MB). Synthesizes 22.05 kHz
    audio. On this sandbox the 3 s reply takes ~1 s to render.
``stub`` (offline tests only)
    Writes a deterministic sine-tone WAV of matching length. It is
    valid audio with the right sample rate — but it is not speech,
    and it is labeled as a stub everywhere it appears.
``kokoro`` (opt-in)
    Kokoro-82M. Needs torch, which is not installed in the sandbox
    (the CUDA torch wheel is ~2.5 GB); import fails loudly with the
    install command instead of silently falling back.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from .audio import TARGET_SR, synth_tone, write_wav

DEFAULT_VOICE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "en_US-lessac-medium.onnx",
)


@dataclass
class TTSResult:
    """One synthesis. ``backend`` names what actually produced it."""

    path: str
    text: str
    duration_s: float
    sample_rate: int
    latency_s: float
    backend: str
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "text": self.text,
            "duration_s": round(self.duration_s, 3),
            "sample_rate": self.sample_rate,
            "latency_s": round(self.latency_s, 3),
            "backend": self.backend,
            "model": self.model,
        }


class StubTTS:
    """Offline stand-in: writes a sine-tone WAV. Valid audio, not
    speech — for tests and for running the pipeline with no model."""

    name = "stub"

    def __init__(self, sample_rate: int = TARGET_SR):
        self.sample_rate = sample_rate

    def synthesize(self, text: str, out_path: str) -> TTSResult:
        t0 = time.time()
        # ~0.35 s of tone per word, so duration scales with the reply.
        duration = max(0.5, 0.35 * len(text.split()))
        audio = synth_tone(440.0, duration, self.sample_rate, amplitude=0.3)
        write_wav(out_path, audio, self.sample_rate)
        return TTSResult(
            path=out_path, text=text, duration_s=duration,
            sample_rate=self.sample_rate, latency_s=time.time() - t0,
            backend="stub", model="stub",
        )


class PiperTTS:
    """Real TTS: Piper (ONNX) with a local voice model."""

    name = "piper"

    def __init__(self, voice_path: str | None = None):
        self.voice_path = voice_path or os.environ.get(
            "PIPER_VOICE", DEFAULT_VOICE_PATH)
        self._voice = None
        self.sample_rate = 22050

    @classmethod
    def available(cls, voice_path: str | None = None) -> bool:
        path = voice_path or os.environ.get("PIPER_VOICE", DEFAULT_VOICE_PATH)
        if not os.path.exists(path):
            return False
        try:
            import piper  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        if self._voice is None:
            if not os.path.exists(self.voice_path):
                raise FileNotFoundError(
                    f"Piper voice not found at {self.voice_path}. Download it with: "
                    "curl -L -o models/en_US-lessac-medium.onnx "
                    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                    "en/en_US/lessac/medium/en_US-lessac-medium.onnx "
                    "and the matching .onnx.json next to it."
                )
            from piper import PiperVoice
            self._voice = PiperVoice.load(self.voice_path)
            self.sample_rate = self._voice.config.sample_rate
        return self._voice

    def synthesize(self, text: str, out_path: str) -> TTSResult:
        t0 = time.time()
        voice = self._load()
        chunks: list[np.ndarray] = []
        with open(out_path, "wb") as f:
            import wave
            w = wave.open(f, "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            for chunk in voice.synthesize(text):
                pcm = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                chunks.append(pcm)
                w.writeframes(chunk.audio_int16_bytes)
            w.close()
        total = sum(len(c) for c in chunks)
        return TTSResult(
            path=out_path, text=text,
            duration_s=total / self.sample_rate if self.sample_rate else 0.0,
            sample_rate=self.sample_rate, latency_s=time.time() - t0,
            backend="piper", model=os.path.basename(self.voice_path),
        )


class KokoroTTS:
    """Opt-in Kokoro-82M backend. Needs torch; not installed here."""

    name = "kokoro"

    def __init__(self, voice: str = "af_heart", lang: str = "en-us"):
        self.voice = voice
        self.lang = lang

    def synthesize(self, text: str, out_path: str) -> TTSResult:  # pragma: no cover
        try:
            from kokoro import KPipeline  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "kokoro is not installed (it needs torch). Install with: "
                "pip install kokoro soundfile"
            ) from e
        t0 = time.time()
        import soundfile as sf
        pipe = KPipeline(lang_code=self.lang[0])
        audio = np.concatenate([
            chunk for _, _, chunk in pipe(text, voice=self.voice)
        ])
        sf.write(out_path, audio, 24000)
        return TTSResult(
            path=out_path, text=text, duration_s=len(audio) / 24000,
            sample_rate=24000, latency_s=time.time() - t0,
            backend="kokoro", model="kokoro-82m",
        )


def make_tts(name: str, **kwargs):
    """Factory: ``piper`` (default, real), ``stub``, ``kokoro``."""
    name = (name or "piper").lower()
    if name == "piper":
        return PiperTTS(**kwargs)
    if name == "stub":
        return StubTTS(**kwargs)
    if name == "kokoro":
        return KokoroTTS(**kwargs)
    raise ValueError(f"unknown TTS backend: {name!r} (piper, stub, kokoro)")
