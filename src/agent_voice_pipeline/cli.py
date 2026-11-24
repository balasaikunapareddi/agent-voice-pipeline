"""CLI: demo, serve, stt, list-backends.

    python -m agent_voice_pipeline demo --wav examples/hello.wav
    python -m agent_voice_pipeline serve
    python -m agent_voice_pipeline stt --wav examples/hello.wav
    python -m agent_voice_pipeline list-backends
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .pipeline import Pipeline, TurnResult, load_config


def _build_pipeline(args) -> Pipeline:
    if getattr(args, "config", None):
        cfg = load_config(args.config)
        return Pipeline.from_config(cfg)
    from .agent import VoiceAgent
    from .llm import make_llm
    from .stt import make_stt
    from .tts import make_tts
    from .vad import make_vad
    llm = make_llm(args.llm)
    return Pipeline(
        vad=make_vad(args.vad),
        stt=make_stt(args.stt),
        llm=llm,
        tts=make_tts(args.tts),
        agent=VoiceAgent(llm=llm),
    )


def _add_stage_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vad", default="energy", choices=["energy", "silero"])
    p.add_argument("--stt", default="faster-whisper",
                   choices=["faster-whisper", "stub", "whisper"])
    p.add_argument("--llm", default="stub",
                   choices=["gemini", "stub", "llamacpp"])
    p.add_argument("--tts", default="piper",
                   choices=["piper", "stub", "kokoro"])
    p.add_argument("--config", default=None,
                   help="YAML config (overrides the --vad/--stt/--llm/--tts flags)")


def cmd_demo(args) -> int:
    pipeline = _build_pipeline(args)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.wav)), "reply.wav")
    print(f"backends: {pipeline.backends}")
    result = pipeline.process_file(args.wav, out)
    print(f"\nVAD segments: {result.segments}")
    print(f"You said: {result.transcript}")
    print(f"Agent: {result.reply}")
    print(f"Reply audio: {result.reply_wav}")
    print("Timings (s): " + ", ".join(
        f"{k}={v:.2f}" for k, v in result.timings_s.items()))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_stt(args) -> int:
    from .stt import main as stt_main
    argv = ["--wav", args.wav, "--backend", args.stt, "--model", args.model,
            "--language", args.language]
    if args.json:
        argv.append("--json")
    return stt_main(argv)


def cmd_serve(args) -> int:
    pipeline = _build_pipeline(args)
    if args.transport == "ws":
        import asyncio
        from .server import serve_websocket
        asyncio.run(serve_websocket(pipeline, args.host, args.port))
    else:
        from .server import serve_sse
        serve_sse(pipeline, args.host, args.port)
    return 0


def cmd_list_backends(args) -> int:
    from .stt import FasterWhisperSTT
    from .tts import PiperTTS
    from .vad import SileroVAD
    rows = [
        ("vad", "energy", True, "built-in, no model"),
        ("vad", "silero", SileroVAD.available(), "needs models/silero_vad.onnx + onnxruntime"),
        ("stt", "faster-whisper", True, "tiny model, auto-downloads (~75 MB)"),
        ("stt", "stub", True, "canned transcript (tests only)"),
        ("stt", "whisper", False, "needs openai-whisper + torch (not installed)"),
        ("llm", "gemini", True, "LiteLLM, needs GEMINI_API_KEY for live use"),
        ("llm", "stub", True, "canned replies (tests only)"),
        ("llm", "llamacpp", False, "needs a llama.cpp server at LLAMA_CPP_URL"),
        ("tts", "piper", PiperTTS.available(), "needs models/en_US-lessac-medium.onnx"),
        ("tts", "stub", True, "sine-tone WAV (tests only)"),
        ("tts", "kokoro", False, "needs kokoro + torch (not installed)"),
    ]
    for stage, name, ok, note in rows:
        print(f"{stage:4} {name:15} {'ready' if ok else 'not installed':13} {note}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_voice_pipeline",
        description="Modular voice-agent pipeline (VAD -> STT -> LLM -> TTS).")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run one audio file through the pipeline")
    d.add_argument("--wav", required=True)
    d.add_argument("--out", default=None, help="reply WAV path")
    d.add_argument("--json", action="store_true")
    _add_stage_args(d)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("stt", help="transcribe one audio file (standalone STT)")
    s.add_argument("--wav", required=True)
    s.add_argument("--stt", default="faster-whisper",
                   choices=["faster-whisper", "stub", "whisper"])
    s.add_argument("--model", default="tiny")
    s.add_argument("--language", default="en")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stt)

    v = sub.add_parser("serve", help="OpenAI Realtime-compatible server")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--transport", default="ws", choices=["ws", "sse"])
    _add_stage_args(v)
    v.set_defaults(func=cmd_serve)

    l = sub.add_parser("list-backends", help="which backends are ready here")
    l.set_defaults(func=cmd_list_backends)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
