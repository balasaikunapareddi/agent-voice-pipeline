"""Audio I/O and helpers for agent-voice-pipeline.

Only the standard library + numpy. WAV files are read/written with the
stdlib ``wave`` module; anything else (mp3, ogg, ...) is decoded through
ffmpeg when it is available. All audio inside the pipeline is mono
float32 at 16 kHz unless a stage says otherwise.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import wave

import numpy as np

TARGET_SR = 16000


def resample(samples: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    """Linear-interpolation resample to ``sr_out`` (default 16 kHz)."""
    samples = np.asarray(samples, dtype=np.float32)
    if sr_in == sr_out or len(samples) == 0:
        return samples
    duration = len(samples) / sr_in
    n_out = max(1, int(round(duration * sr_out)))
    t_old = np.linspace(0.0, duration, len(samples), endpoint=False)
    t_new = np.arange(n_out, dtype=np.float64) / sr_out
    return np.interp(t_new, t_old, samples).astype(np.float32)


def read_wav(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Read a WAV file -> (mono float32 samples at ``target_sr``, sr)."""
    with wave.open(path, "rb") as w:
        n_frames = w.getnframes()
        sr = w.getframerate()
        n_channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(n_frames)
    if width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 3:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        vals = arr[:, 0] | (arr[:, 1] << 8) | (arr[:, 2] << 16)
        vals = np.where(vals >= 1 << 23, vals - (1 << 24), vals)
        audio = vals.astype(np.float32) / 8388608.0
    else:  # pragma: no cover - unusual widths
        raise ValueError(f"unsupported sample width: {width}")
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    audio = resample(np.ascontiguousarray(audio), sr, target_sr)
    return audio.astype(np.float32), target_sr


def read_audio(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Read any audio file. WAV goes through the stdlib; anything else
    is decoded with ffmpeg (must be on PATH)."""
    if path.lower().endswith(".wav"):
        return read_wav(path, target_sr)
    if _which("ffmpeg") is None:
        raise RuntimeError(
            f"cannot decode {path!r}: not a WAV file and ffmpeg is not installed"
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path,
             "-ac", "1", "-ar", str(target_sr), "-f", "wav", tmp_path],
            check=True,
        )
        return read_wav(tmp_path, target_sr)
    finally:
        os.unlink(tmp_path)


def write_wav(path: str, samples: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write mono float32 samples as 16-bit PCM WAV."""
    samples = np.asarray(samples, dtype=np.float32)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = np.round(pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def wav_info(path: str) -> dict:
    """Duration / sample rate / channels of a WAV file, without decoding."""
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        return {
            "path": path,
            "sample_rate": sr,
            "channels": w.getnchannels(),
            "frames": n,
            "duration_s": n / sr if sr else 0.0,
        }


def frame_audio(samples: np.ndarray, frame_ms: int = 30, sr: int = TARGET_SR):
    """Yield consecutive non-overlapping frames of ``frame_ms`` ms."""
    size = max(1, int(sr * frame_ms / 1000))
    for i in range(0, len(samples) - size + 1, size):
        yield samples[i:i + size]


def synth_tone(freq_hz: float, duration_s: float, sr: int = TARGET_SR,
               amplitude: float = 0.5) -> np.ndarray:
    """A pure sine tone — used by tests and by the stub TTS."""
    t = np.arange(int(duration_s * sr), dtype=np.float64) / sr
    return (amplitude * np.sin(2 * math.pi * freq_hz * t)).astype(np.float32)


def synth_bursts(bursts: list[tuple[float, float]], total_s: float,
                 sr: int = TARGET_SR, freq_hz: float = 220.0,
                 amplitude: float = 0.5, noise: float = 0.0,
                 seed: int = 0) -> np.ndarray:
    """Silence with sine bursts at ``(start_s, duration_s)`` — a fake
    utterance for VAD tests. Deterministic for a fixed seed."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(total_s * sr), dtype=np.float32)
    for start, dur in bursts:
        i0 = int(start * sr)
        i1 = min(len(out), int((start + dur) * sr))
        if i1 > i0:
            out[i0:i1] += synth_tone(freq_hz, (i1 - i0) / sr, sr, amplitude)
    if noise > 0:
        out = out + rng.normal(0.0, noise, size=len(out)).astype(np.float32)
    return np.clip(out, -1.0, 1.0)


def _which(cmd: str) -> str | None:
    import shutil
    return shutil.which(cmd)
