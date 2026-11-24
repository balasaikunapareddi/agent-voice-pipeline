"""STT contracts: stub shape, factory, word accuracy, module CLI."""

import numpy as np
import pytest

from agent_voice_pipeline.audio import synth_tone
from agent_voice_pipeline.stt import (
    STUB_TRANSCRIPT,
    FasterWhisperSTT,
    StubSTT,
    make_stt,
    transcribe_file,
    word_accuracy,
)


def test_stub_returns_canned_transcript():
    audio = synth_tone(440.0, 1.0)
    r = StubSTT().transcribe(audio)
    assert r.text == STUB_TRANSCRIPT
    assert r.backend == "stub"
    assert r.duration_s == pytest.approx(1.0, abs=0.01)
    assert r.segments[0]["text"] == STUB_TRANSCRIPT


def test_stub_transcribe_file(tmp_path):
    from agent_voice_pipeline.audio import write_wav
    path = tmp_path / "tone.wav"
    write_wav(str(path), synth_tone(440.0, 0.5))
    r = StubSTT().transcribe_file(str(path))
    assert r.text == STUB_TRANSCRIPT


def test_make_stt_factory():
    assert make_stt("stub").name == "stub"
    assert make_stt("faster-whisper").name == "faster-whisper"
    assert make_stt("whisper").name == "whisper"
    with pytest.raises(ValueError):
        make_stt("nope")


def test_word_accuracy_exact():
    out = word_accuracy("Hello, what time does the library open today?",
                        "Hello, what time does the library open today?")
    assert out["accuracy"] == 1.0
    assert out["correct"] == 8
    assert out["missing"] == [] and out["extra"] == []


def test_word_accuracy_partial():
    out = word_accuracy("Hello what time library open",
                        "Hello, what time does the library open today?")
    assert out["accuracy"] == pytest.approx(5 / 8)
    assert "does" in out["missing"]


def test_word_accuracy_empty_reference():
    assert word_accuracy("anything", "")["accuracy"] == 1.0


def test_transcribe_file_helper_with_stub(tmp_path):
    from agent_voice_pipeline.audio import write_wav
    path = tmp_path / "tone.wav"
    write_wav(str(path), synth_tone(440.0, 0.5))
    r = transcribe_file(str(path), backend="stub")
    assert r.backend == "stub" and r.text == STUB_TRANSCRIPT


def test_stt_cli_stub(tmp_path, capsys):
    from agent_voice_pipeline.audio import write_wav
    from agent_voice_pipeline.stt import main
    path = tmp_path / "tone.wav"
    write_wav(str(path), synth_tone(440.0, 0.5))
    assert main(["--wav", str(path), "--backend", "stub"]) == 0
    out = capsys.readouterr().out
    assert STUB_TRANSCRIPT in out


def test_stt_cli_json_stub(tmp_path, capsys):
    import json
    from agent_voice_pipeline.audio import write_wav
    from agent_voice_pipeline.stt import main
    path = tmp_path / "tone.wav"
    write_wav(str(path), synth_tone(440.0, 0.5))
    assert main(["--wav", str(path), "--backend", "stub", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["backend"] == "stub"
    assert data["text"] == STUB_TRANSCRIPT


def test_faster_whisper_real_transcription():
    """Real backend on the shipped WAV — the headline check. Skipped
    only if faster-whisper is not installed."""
    import os
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        pytest.skip("faster-whisper not installed")
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "hello.wav")
    r = FasterWhisperSTT().transcribe_file(path)
    assert r.backend == "faster-whisper"
    assert r.text, "empty transcript from real STT"
    acc = word_accuracy(r.text, "Hello, what time does the library open today?")
    assert acc["accuracy"] >= 0.75, f"low word accuracy: {acc} / {r.text!r}"
