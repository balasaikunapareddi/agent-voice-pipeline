"""Audio helpers: WAV I/O, resampling, synthetic test audio."""

import numpy as np

from agent_voice_pipeline.audio import (
    TARGET_SR,
    frame_audio,
    read_audio,
    read_wav,
    resample,
    synth_bursts,
    synth_tone,
    wav_info,
    write_wav,
)


def test_target_sr_is_16k():
    assert TARGET_SR == 16000


def test_wav_round_trip(tmp_path):
    tone = synth_tone(440.0, 0.5)
    path = tmp_path / "tone.wav"
    write_wav(str(path), tone)
    back, sr = read_wav(str(path))
    assert sr == 16000
    assert len(back) == len(tone)
    assert np.allclose(back, tone, atol=1 / 32767 + 1e-6)


def test_wav_info(tmp_path):
    path = tmp_path / "tone.wav"
    write_wav(str(path), synth_tone(220.0, 1.0))
    info = wav_info(str(path))
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert abs(info["duration_s"] - 1.0) < 0.01


def test_resample_down_and_up():
    tone = synth_tone(440.0, 1.0, sr=16000)
    down = resample(tone, 16000, 8000)
    assert len(down) == 8000
    up = resample(down, 8000, 16000)
    assert len(up) == 16000
    # A 440 Hz tone survives a down/up round trip recognisably.
    assert float(np.max(np.abs(up))) > 0.3


def test_resample_same_rate_is_identity():
    tone = synth_tone(440.0, 0.2)
    assert np.array_equal(resample(tone, 16000, 16000), tone)


def test_read_audio_rejects_missing_file():
    import pytest
    with pytest.raises(Exception):
        read_audio("/nonexistent/nope.wav")


def test_frame_audio_sizes():
    samples = np.zeros(16000, dtype=np.float32)
    frames = list(frame_audio(samples, frame_ms=30))
    assert len(frames) == 33  # 1000 ms / 30 ms, truncated
    assert all(len(f) == 480 for f in frames)


def test_synth_bursts_placement():
    audio = synth_bursts([(0.5, 0.5), (2.0, 0.5)], total_s=3.0)
    assert len(audio) == 48000
    assert float(np.max(np.abs(audio[:8000]))) == 0.0  # silence first
    assert float(np.max(np.abs(audio[8000:16000]))) > 0.4  # burst 1
    assert float(np.max(np.abs(audio[16000:32000]))) == 0.0  # gap
    assert float(np.max(np.abs(audio[32000:40000]))) > 0.4  # burst 2
