"""Modular voice-agent pipeline (VAD -> STT -> LLM -> TTS).

Lazy exports (PEP 562): importing the package stays light, and
``python -m agent_voice_pipeline.stt`` does not trip the double-import
warning.
"""

__version__ = "0.1.0"

__all__ = ["VoiceAgent", "Pipeline", "TurnResult", "STTResult",
           "transcribe_file", "__version__"]


def __getattr__(name):
    if name == "VoiceAgent":
        from .agent import VoiceAgent
        return VoiceAgent
    if name in ("Pipeline", "TurnResult"):
        from .pipeline import Pipeline, TurnResult
        return {"Pipeline": Pipeline, "TurnResult": TurnResult}[name]
    if name in ("STTResult", "transcribe_file"):
        from .stt import STTResult, transcribe_file
        return {"STTResult": STTResult, "transcribe_file": transcribe_file}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
