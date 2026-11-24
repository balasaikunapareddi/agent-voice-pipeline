"""Voice Activity Detection: speech boundaries and turn-taking.

Two backends share one segment API::

    vad = EnergyVAD()            # always available, no model
    vad = SileroVAD()            # real Silero VAD v5 via ONNX Runtime
    segments = vad.segment(samples)   # -> [(start_s, end_s), ...]

EnergyVAD is a deterministic energy detector (frame energy vs an
adaptive noise floor, plus hangover) — good enough for clean file
audio and for every offline test. SileroVAD runs the real Silero VAD
v5 ONNX graph (``models/silero_vad.onnx``) with no torch dependency;
it needs the ``onnxruntime`` package and the model file, both
documented in the README. Silero is the default in the upstream
speech-to-speech project; here it is opt-in via ``--vad silero``
because the model file is a separate download.
"""

from __future__ import annotations

import os

import numpy as np

from .audio import TARGET_SR, frame_audio

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "silero_vad.onnx",
)


def _segments_from_mask(mask: list[bool], frame_s: float,
                        hangover_frames: int = 10,
                        min_speech_s: float = 0.2) -> list[tuple[float, float]]:
    """Merge a per-frame speech mask into (start, end) segments."""
    # Hangover: bridge short gaps so one utterance stays one segment.
    merged = list(mask)
    n = len(merged)
    i = 0
    while i < n:
        if not merged[i]:
            j = i
            while j < n and not merged[j]:
                j += 1
            if j - i <= hangover_frames and i > 0 and j < n:
                for k in range(i, j):
                    merged[k] = True
            i = j
        else:
            i += 1
    segments: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if merged[i]:
            j = i
            while j < n and merged[j]:
                j += 1
            start, end = i * frame_s, j * frame_s
            if end - start >= min_speech_s:
                segments.append((round(start, 3), round(end, 3)))
            i = j
        else:
            i += 1
    return segments


class EnergyVAD:
    """Deterministic energy-based VAD. No model, no downloads.

    Frame energy is compared against an adaptive floor (10th percentile
    of frame energies, scaled). Works well on clean recordings and on
    the synthetic bursts used by the test suite.
    """

    name = "energy"

    def __init__(self, frame_ms: int = 30, factor: float = 10.0,
                 floor: float = 1e-4, hangover_frames: int = 10,
                 min_speech_s: float = 0.2):
        self.frame_ms = frame_ms
        self.factor = factor
        self.floor = floor
        self.hangover_frames = hangover_frames
        self.min_speech_s = min_speech_s

    def segment(self, samples: np.ndarray, sr: int = TARGET_SR) -> list[tuple[float, float]]:
        samples = np.asarray(samples, dtype=np.float32)
        frames = list(frame_audio(samples, self.frame_ms, sr))
        if not frames:
            return []
        energies = np.array([float(np.mean(f.astype(np.float64) ** 2)) for f in frames])
        noise = float(np.percentile(energies, 10)) if len(energies) else 0.0
        threshold = max(self.floor, noise * self.factor)
        mask = [e > threshold for e in energies]
        frame_s = self.frame_ms / 1000.0
        return _segments_from_mask(mask, frame_s, self.hangover_frames, self.min_speech_s)

    def is_speech(self, samples: np.ndarray, sr: int = TARGET_SR) -> bool:
        return len(self.segment(samples, sr)) > 0


class SileroVAD:
    """Real Silero VAD v5, run from its ONNX graph via onnxruntime.

    No torch needed. The model file is ``models/silero_vad.onnx``
    (2.2 MB, from github.com/snakers4/silero-vad). Raises a clear
    error if the package or the file is missing — use
    :meth:`available` to check first.
    """

    name = "silero"

    def __init__(self, model_path: str | None = None, threshold: float = 0.5,
                 hangover_frames: int = 10, min_speech_s: float = 0.2):
        self.model_path = model_path or os.environ.get(
            "SILERO_VAD_MODEL", DEFAULT_MODEL_PATH)
        self.threshold = threshold
        self.hangover_frames = hangover_frames
        self.min_speech_s = min_speech_s
        self._session = None

    @classmethod
    def available(cls, model_path: str | None = None) -> bool:
        path = model_path or os.environ.get("SILERO_VAD_MODEL", DEFAULT_MODEL_PATH)
        if not os.path.exists(path):
            return False
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        if self._session is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Silero VAD model not found at {self.model_path}. "
                    "Download it with: curl -L -o models/silero_vad.onnx "
                    "https://github.com/snakers4/silero-vad/raw/master/"
                    "src/silero_vad/data/silero_vad.onnx"
                )
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"])
        return self._session

    def segment(self, samples: np.ndarray, sr: int = TARGET_SR) -> list[tuple[float, float]]:
        if sr != TARGET_SR:
            from .audio import resample
            samples = resample(np.asarray(samples, dtype=np.float32), sr, TARGET_SR)
        samples = np.asarray(samples, dtype=np.float32)
        session = self._load()
        # Silero VAD v6 ONNX: 512-sample windows at 16 kHz, but each
        # call feeds the previous 64 samples as context (576 total) —
        # without it the model returns near-zero probabilities. This
        # mirrors the upstream OnnxWrapper in silero-vad's utils_vad.
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(64, dtype=np.float32)
        probs: list[float] = []
        win = 512
        for i in range(0, len(samples) - win + 1, win):
            chunk = np.concatenate([context, samples[i:i + win]]).reshape(1, -1)
            out = session.run(None, {
                "input": chunk,
                "state": state,
                "sr": np.array(TARGET_SR, dtype=np.int64),
            })
            probs.append(float(out[0][0][0]))
            state = out[1]
            context = chunk[0, -64:]
        mask = [p >= self.threshold for p in probs]
        frame_s = win / TARGET_SR
        return _segments_from_mask(mask, frame_s, self.hangover_frames, self.min_speech_s)

    def is_speech(self, samples: np.ndarray, sr: int = TARGET_SR) -> bool:
        return len(self.segment(samples, sr)) > 0


def make_vad(name: str, **kwargs):
    """Factory: ``energy`` (default) or ``silero``."""
    name = (name or "energy").lower()
    if name == "energy":
        return EnergyVAD(**kwargs)
    if name == "silero":
        return SileroVAD(**kwargs)
    raise ValueError(f"unknown VAD backend: {name!r} (energy, silero)")
