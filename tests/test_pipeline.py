"""Pipeline queue flow, turn-taking, config, and the CLI demo."""

import os

import numpy as np
import pytest

from agent_voice_pipeline.agent import VoiceAgent
from agent_voice_pipeline.audio import synth_bursts, write_wav
from agent_voice_pipeline.llm import StubLLM
from agent_voice_pipeline.pipeline import Pipeline, load_config
from agent_voice_pipeline.stt import StubSTT
from agent_voice_pipeline.tts import StubTTS
from agent_voice_pipeline.vad import EnergyVAD


def _stub_pipeline():
    return Pipeline(vad=EnergyVAD(), stt=StubSTT(), llm=StubLLM(),
                    tts=StubTTS(), agent=VoiceAgent(llm=StubLLM()))


def test_pipeline_full_turn_stub(tmp_path):
    audio = synth_bursts([(0.3, 1.0)], total_s=2.0)
    wav = tmp_path / "in.wav"
    write_wav(str(wav), audio)
    out = tmp_path / "reply.wav"
    result = _stub_pipeline().process_file(str(wav), str(out))
    assert len(result.segments) == 1
    assert result.transcript  # stub transcript
    assert result.reply  # stub reply
    assert os.path.exists(result.reply_wav)
    assert result.timings_s["vad"] >= 0
    assert result.timings_s["stt"] >= 0
    assert result.timings_s["llm"] >= 0
    assert result.timings_s["tts"] >= 0
    assert result.backends["vad"] == "energy"
    assert result.backends["stt"] == "stub"


def test_pipeline_silence_produces_empty_transcript(tmp_path):
    audio = np.zeros(16000, dtype=np.float32)
    wav = tmp_path / "silent.wav"
    write_wav(str(wav), audio)
    result = _stub_pipeline().process_file(str(wav), str(tmp_path / "r.wav"))
    assert result.segments == []
    assert result.transcript == ""


def test_pipeline_two_turns_keep_history(tmp_path):
    audio = synth_bursts([(0.3, 1.0)], total_s=2.0)
    wav = tmp_path / "in.wav"
    write_wav(str(wav), audio)
    pipe = _stub_pipeline()
    pipe.process_file(str(wav), str(tmp_path / "r1.wav"))
    pipe.process_file(str(wav), str(tmp_path / "r2.wav"))
    # system + 2 user + 2 assistant turns
    assert len(pipe.agent.history) == 5


def test_pipeline_backends_report():
    pipe = _stub_pipeline()
    assert pipe.backends == {"vad": "energy", "stt": "stub",
                             "llm": "stub", "tts": "stub"}


def test_load_config_default():
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    cfg = load_config(path)
    assert cfg["vad"]["backend"] == "energy"
    assert cfg["stt"]["backend"] == "faster-whisper"
    assert cfg["llm"]["backend"] == "stub"
    assert cfg["tts"]["backend"] == "piper"


def test_pipeline_from_config_stub(tmp_path):
    cfg = {"vad": {"backend": "energy"}, "stt": {"backend": "stub"},
           "llm": {"backend": "stub"}, "tts": {"backend": "stub"}}
    pipe = Pipeline.from_config(cfg)
    audio = synth_bursts([(0.3, 1.0)], total_s=2.0)
    wav = tmp_path / "in.wav"
    write_wav(str(wav), audio)
    result = pipe.process_file(str(wav), str(tmp_path / "r.wav"))
    assert result.transcript
    assert result.reply


def test_load_config_rejects_non_mapping(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_config(str(bad))


def test_cli_demo_stub(tmp_path, capsys):
    from agent_voice_pipeline.cli import main
    audio = synth_bursts([(0.3, 1.0)], total_s=2.0)
    wav = tmp_path / "in.wav"
    write_wav(str(wav), audio)
    rc = main(["demo", "--wav", str(wav), "--vad", "energy",
               "--stt", "stub", "--llm", "stub", "--tts", "stub",
               "--out", str(tmp_path / "reply.wav")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "You said:" in out
    assert "Agent:" in out
    assert os.path.exists(tmp_path / "reply.wav")


def test_cli_list_backends(capsys):
    from agent_voice_pipeline.cli import main
    assert main(["list-backends"]) == 0
    out = capsys.readouterr().out
    assert "faster-whisper" in out and "piper" in out and "silero" in out
