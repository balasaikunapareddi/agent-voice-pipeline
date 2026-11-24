"""The four-stage pipeline, faithful to huggingface/speech-to-speech.

Four stages, each in its own thread, connected by queues::

    WAV file -> [VAD] -> speech segments -> [STT] -> transcript
             -> [LLM/agent] -> reply text -> [TTS] -> reply WAV

Every stage is swappable: pick backends by name in a YAML config or
on the CLI. The pipeline is the wiring — the stages do the work, and
each stage's timings are reported per turn so you can see where the
milliseconds go.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import yaml

from .agent import VoiceAgent
from .audio import TARGET_SR, read_audio
from .llm import make_llm
from .stt import make_stt
from .tts import make_tts
from .vad import make_vad

_END = object()  # queue sentinel: stage finished


@dataclass
class TurnResult:
    """Everything one file turn produced, with per-stage timings."""

    transcript: str
    segments: list[tuple[float, float]]
    reply: str
    reply_wav: str
    timings_s: dict = field(default_factory=dict)
    backends: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "segments": [list(s) for s in self.segments],
            "reply": self.reply,
            "reply_wav": self.reply_wav,
            "timings_s": {k: round(v, 3) for k, v in self.timings_s.items()},
            "backends": self.backends,
        }


def load_config(path: str) -> dict:
    """Load a YAML pipeline config (see configs/default.yaml)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path!r} must be a YAML mapping")
    return cfg


class Pipeline:
    """VAD -> STT -> LLM -> TTS, each stage in its own thread."""

    def __init__(self, vad=None, stt=None, llm=None, tts=None,
                 agent: VoiceAgent | None = None):
        self.vad = vad or make_vad("energy")
        self.stt = stt or make_stt("stub")
        self.llm = llm or make_llm("stub")
        self.tts = tts or make_tts("stub")
        self.agent = agent or VoiceAgent(llm=self.llm)

    @classmethod
    def from_config(cls, cfg: dict) -> "Pipeline":
        vad = make_vad(cfg.get("vad", {}).get("backend", "energy"),
                       **{k: v for k, v in cfg.get("vad", {}).items()
                          if k != "backend"})
        stt = make_stt(cfg.get("stt", {}).get("backend", "faster-whisper"),
                       **{k: v for k, v in cfg.get("stt", {}).items()
                          if k != "backend"})
        llm = make_llm(cfg.get("llm", {}).get("backend", "stub"),
                       **{k: v for k, v in cfg.get("llm", {}).items()
                          if k != "backend"})
        tts = make_tts(cfg.get("tts", {}).get("backend", "piper"),
                       **{k: v for k, v in cfg.get("tts", {}).items()
                          if k != "backend"})
        return cls(vad=vad, stt=stt, llm=llm, tts=tts)

    @property
    def backends(self) -> dict:
        return {
            "vad": getattr(self.vad, "name", "?"),
            "stt": getattr(self.stt, "name", "?"),
            "llm": getattr(self.llm, "name", "?"),
            "tts": getattr(self.tts, "name", "?"),
        }

    def _run_stage(self, name: str, fn, q_in: queue.Queue,
                   q_out: queue.Queue, timings: dict) -> None:
        try:
            while True:
                item = q_in.get()
                if item is _END:
                    break
                t0 = time.time()
                out = fn(item)
                timings[name] = timings.get(name, 0.0) + (time.time() - t0)
                q_out.put(out)
        except Exception as e:  # stage errors travel downstream as data
            q_out.put({"error": f"{name}: {e}"})
        finally:
            q_out.put(_END)

    def process_file(self, wav_path: str, out_wav: str) -> TurnResult:
        """Run one audio file through all four stages.

        The file is split by VAD; each speech segment is transcribed
        and the segments are joined into one transcript for the agent.
        (Single-turn file demo: no microphone, no live turn-taking.)
        """
        samples, sr = read_audio(wav_path)
        return self.process_audio(samples, sr, out_wav)

    def process_audio(self, samples: np.ndarray, sr: int,
                      out_wav: str) -> TurnResult:
        timings: dict[str, float] = {}
        backends = self.backends

        # Stage 1 — VAD (runs on the calling thread; it is fast).
        t0 = time.time()
        segments = self.vad.segment(samples, sr)
        timings["vad"] = time.time() - t0

        # Stage 2 — STT, one thread per segment (queued, in order).
        q_stt_in: queue.Queue = queue.Queue()
        q_stt_out: queue.Queue = queue.Queue()
        for (start, end) in segments:
            i0, i1 = int(start * sr), int(end * sr)
            q_stt_in.put(samples[i0:i1])
        q_stt_in.put(_END)
        stt_thread = threading.Thread(
            target=self._run_stage,
            args=("stt", lambda chunk: self.stt.transcribe(chunk, sr),
                  q_stt_in, q_stt_out, timings),
            daemon=True)
        stt_thread.start()
        transcripts: list[str] = []
        while True:
            item = q_stt_out.get()
            if item is _END:
                break
            if isinstance(item, dict) and "error" in item:
                raise RuntimeError(item["error"])
            transcripts.append(item.text)
        stt_thread.join()
        transcript = " ".join(t.strip() for t in transcripts if t.strip())

        # Stage 3 — LLM/agent (streaming, collected here).
        t0 = time.time()
        reply = "".join(self.agent.stream_turn(transcript)).strip()
        timings["llm"] = time.time() - t0

        # Stage 4 — TTS.
        t0 = time.time()
        tts_result = self.tts.synthesize(reply, out_wav)
        timings["tts"] = time.time() - t0

        return TurnResult(
            transcript=transcript,
            segments=segments,
            reply=reply,
            reply_wav=os.path.abspath(out_wav),
            timings_s=timings,
            backends={**backends, "tts_model": tts_result.model},
        )
