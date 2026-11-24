"""LLM + TTS stub contracts and the agent loop."""

import os

import numpy as np
import pytest

from agent_voice_pipeline.agent import VoiceAgent
from agent_voice_pipeline.audio import read_wav, wav_info
from agent_voice_pipeline.llm import LlamaCppLLM, LiteLLMChat, StubLLM, make_llm
from agent_voice_pipeline.prompts import SYSTEM_PROMPT
from agent_voice_pipeline.tts import PiperTTS, StubTTS, make_tts


def test_stub_llm_library_reply():
    llm = StubLLM()
    out = llm.reply([{"role": "user", "content": "What time does the library open?"}])
    assert "9 AM" in out


def test_stub_llm_streams_in_chunks():
    llm = StubLLM()
    chunks = list(llm.stream_reply([{"role": "user", "content": "hello"}]))
    assert len(chunks) > 1
    assert "".join(chunks).strip().startswith("Hello")


def test_stub_llm_empty_input():
    out = StubLLM().reply([{"role": "user", "content": "   "}])
    assert "didn't catch" in out


def test_make_llm_factory():
    assert make_llm("stub").name == "stub"
    assert make_llm("gemini").name == "gemini"
    assert make_llm("llamacpp").name == "llamacpp"
    with pytest.raises(ValueError):
        make_llm("nope")


def test_litellm_chat_default_model_env():
    assert "gemini" in LiteLLMChat().model or "flash" in LiteLLMChat().model


def test_llamacpp_url_default():
    assert LlamaCppLLM().base_url.startswith("http://127.0.0.1")


def test_agent_keeps_history():
    agent = VoiceAgent(llm=StubLLM())
    reply = agent.run_turn("Hello")
    assert reply
    assert agent.history[0]["role"] == "system"
    assert agent.history[1] == {"role": "user", "content": "Hello"}
    assert agent.history[2]["role"] == "assistant"
    assert agent.history[2]["content"] == reply


def test_agent_stream_turn_matches_reply():
    a1 = VoiceAgent(llm=StubLLM())
    a2 = VoiceAgent(llm=StubLLM())
    streamed = "".join(a1.stream_turn("What time does the library open?")).strip()
    direct = a2.run_turn("What time does the library open?")
    assert streamed == direct


def test_agent_uses_system_prompt():
    assert "voice assistant" in SYSTEM_PROMPT.lower()
    agent = VoiceAgent(llm=StubLLM(), system_prompt="CUSTOM PROMPT")
    assert agent.history[0]["content"] == "CUSTOM PROMPT"


def test_agent_reset():
    agent = VoiceAgent(llm=StubLLM())
    agent.run_turn("hello")
    agent.reset()
    assert len(agent.history) == 1
    assert agent.history[0]["role"] == "system"


def test_stub_tts_writes_valid_wav(tmp_path):
    out = tmp_path / "reply.wav"
    r = StubTTS().synthesize("hello world", str(out))
    assert r.backend == "stub"
    assert r.duration_s > 0
    info = wav_info(str(out))
    assert info["sample_rate"] == 16000
    assert info["duration_s"] > 0


def test_stub_tts_duration_scales_with_words(tmp_path):
    short = StubTTS().synthesize("hi", str(tmp_path / "a.wav"))
    long_ = StubTTS().synthesize("one two three four five six", str(tmp_path / "b.wav"))
    assert long_.duration_s > short.duration_s


def test_make_tts_factory():
    assert make_tts("stub").name == "stub"
    assert make_tts("piper").name == "piper"
    assert make_tts("kokoro").name == "kokoro"
    with pytest.raises(ValueError):
        make_tts("nope")


def test_piper_real_synthesis(tmp_path):
    """Real Piper TTS — skipped if the voice model is not present."""
    voice = os.path.join(os.path.dirname(__file__), "..", "models",
                         "en_US-lessac-medium.onnx")
    if not PiperTTS.available(voice):
        pytest.skip("Piper voice not downloaded")
    out = tmp_path / "reply.wav"
    r = PiperTTS(voice_path=voice).synthesize("The library opens at nine AM.", str(out))
    assert r.backend == "piper"
    assert r.sample_rate == 22050
    assert r.duration_s > 0.5
    samples, sr = read_wav(str(out), target_sr=r.sample_rate)
    assert sr == 22050
    assert float(np.max(np.abs(samples))) > 0.05
