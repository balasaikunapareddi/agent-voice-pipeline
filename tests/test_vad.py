"""VAD: energy segmentation on synthetic audio + Silero smoke test."""

import numpy as np
import pytest

from agent_voice_pipeline.audio import synth_bursts
from agent_voice_pipeline.vad import EnergyVAD, SileroVAD, make_vad

SR = 16000


def test_energy_vad_finds_two_bursts():
    audio = synth_bursts([(0.5, 0.5), (2.0, 0.5)], total_s=3.0)
    vad = EnergyVAD()
    segs = vad.segment(audio, SR)
    assert len(segs) == 2
    assert abs(segs[0][0] - 0.5) < 0.1
    assert abs(segs[0][1] - 1.0) < 0.1
    assert abs(segs[1][0] - 2.0) < 0.1
    assert abs(segs[1][1] - 2.5) < 0.1


def test_energy_vad_silence_is_empty():
    audio = np.zeros(SR * 2, dtype=np.float32)
    assert EnergyVAD().segment(audio, SR) == []


def test_energy_vad_short_burst_dropped():
    audio = synth_bursts([(0.5, 0.05)], total_s=1.0)  # 50 ms < 200 ms min
    assert EnergyVAD().segment(audio, SR) == []


def test_energy_vad_hangover_bridges_small_gap():
    # Two bursts 0.15 s apart: hangover (10 frames = 0.3 s) merges them.
    audio = synth_bursts([(0.5, 0.3), (0.95, 0.3)], total_s=2.0)
    segs = EnergyVAD().segment(audio, SR)
    assert len(segs) == 1


def test_energy_vad_is_speech():
    assert EnergyVAD().is_speech(synth_bursts([(0.2, 0.5)], total_s=1.0), SR)
    assert not EnergyVAD().is_speech(np.zeros(SR, dtype=np.float32), SR)


def test_make_vad_factory():
    assert make_vad("energy").name == "energy"
    assert make_vad("silero").name == "silero"
    with pytest.raises(ValueError):
        make_vad("nope")


def test_silero_vad_on_real_audio():
    """Real Silero VAD v5 on the shipped test WAV — skipped if the
    model file is not present (it is gitignored)."""
    import os
    from agent_voice_pipeline.audio import read_audio
    model = os.path.join(os.path.dirname(__file__), "..", "models", "silero_vad.onnx")
    if not SileroVAD.available(model):
        pytest.skip("Silero model not downloaded")
    samples, sr = read_audio(os.path.join(
        os.path.dirname(__file__), "..", "examples", "hello.wav"))
    segs = SileroVAD(model_path=model).segment(samples, sr)
    assert len(segs) >= 1
    # The whole file is one utterance: one segment covering most of it.
    assert segs[0][0] < 0.5
    assert segs[-1][1] > 2.0
