from __future__ import annotations

import math
import re

import numpy as np

from app.analyze import SPECTROGRAM_BANDS_HZ, SPECTRUM_BANDS_HZ
from tests.conftest import noise, sine, write


def post(client, path):
    with open(path, "rb") as f:
        return client.post("/analyze", files={"file": ("x.wav", f, "audio/wav")})


def numbers(value):
    """Every number anywhere in a JSON body."""
    if isinstance(value, dict):
        for item in value.values():
            yield from numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from numbers(item)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        yield value


def band_mean(spectrum, key, lo, hi):
    bands = np.array(spectrum["bands_hz"])
    values = np.array(spectrum[key])
    return values[(bands >= lo) & (bands <= hi)].mean()


def test_shape_and_lengths(client, tmp_path):
    response = post(client, write(tmp_path / "n.wav", noise(3, 0.5, brightness=0.5, seed=1)))
    assert response.status_code == 200
    body = response.json()

    assert body["analysis_version"] == 1
    assert body["format"] == {"sample_rate": 48000, "channels": 2, "duration_s": 3.0}
    assert isinstance(body["integrated_lufs"], float)

    wave = body["waveform"]
    assert wave["points"] == 1000
    assert len(wave["peak"]) == len(wave["rms"]) == 1000
    assert math.isclose(wave["seconds_per_point"], 0.003)
    assert all(0 <= r <= p <= 1 for p, r in zip(wave["peak"], wave["rms"], strict=True))

    spectrum = body["spectrum"]
    assert spectrum["bands_hz"] == [round(f, 2) for f in SPECTRUM_BANDS_HZ]
    assert spectrum["bands_hz"][0] >= 20 and spectrum["bands_hz"][-1] <= 20_000
    for key in ("mid_db", "side_db", "relative_mid_db", "relative_side_db"):
        assert len(spectrum[key]) == len(spectrum["bands_hz"])

    gram = body["spectrogram"]
    assert gram["frames"] == len(gram["db"]) == 1 + (3 * 48_000 - 8192) // 4096
    assert gram["bands_hz"] == [round(f, 2) for f in SPECTROGRAM_BANDS_HZ]
    assert all(len(row) == len(gram["bands_hz"]) for row in gram["db"])
    assert all(isinstance(v, int) and v >= -120 for row in gram["db"] for v in row)

    assert -1 <= body["stereo"]["correlation"] <= 1
    assert not re.search(r"-0\.0[,\]}]", response.text)  # a rounded zero is unsigned
    assert isinstance(body["stereo"]["side_to_mid_db"], float)
    assert all(math.isfinite(v) for v in numbers(body))


def test_relative_curve_is_level_minus_loudness(client, tmp_path):
    body = post(client, write(tmp_path / "n.wav", noise(3, 0.5, brightness=0.5, seed=2))).json()
    spectrum, lufs = body["spectrum"], body["integrated_lufs"]

    for absolute, rel in zip(spectrum["mid_db"], spectrum["relative_mid_db"], strict=True):
        assert abs(absolute - lufs - rel) <= 0.011


def test_bright_noise_has_more_relative_top_end_than_dark(client, tmp_path):
    dark = post(client, write(tmp_path / "d.wav", noise(3, 0.5, brightness=0.1, seed=3))).json()
    bright = post(client, write(tmp_path / "b.wav", noise(3, 0.5, brightness=0.9, seed=4))).json()

    top = "relative_mid_db"
    assert band_mean(bright["spectrum"], top, 4000, 16000) > (
        band_mean(dark["spectrum"], top, 4000, 16000) + 3
    )


def test_digital_silence_does_not_bias_the_spectrum(client, tmp_path):
    """Three seconds of silence either side would pull every band down about
    5 dB if it were averaged in. The gaps are a whole number of hops long so
    both files see the noise through the same frames: a frame shift alone
    moves single-bin bands of random noise by more than the silence should."""
    music = noise(3, 0.5, brightness=0.5, seed=5)
    gap = np.zeros((36 * 4096, 2))
    plain = post(client, write(tmp_path / "plain.wav", music)).json()
    padded = post(client, write(tmp_path / "pad.wav", np.vstack([gap, music, gap]))).json()

    for key in ("mid_db", "side_db"):
        difference = np.subtract(plain["spectrum"][key], padded["spectrum"][key])
        assert np.abs(difference).max() <= 0.5


def test_all_silent_file_still_answers(client, tmp_path):
    body = post(client, write(tmp_path / "s.wav", np.zeros((48_000, 2)))).json()

    assert body["integrated_lufs"] is None
    assert body["spectrum"]["relative_mid_db"] is None
    assert body["spectrum"]["relative_side_db"] is None
    assert all(math.isfinite(v) for v in body["spectrum"]["mid_db"])
    assert all(v == -120 for row in body["spectrogram"]["db"] for v in row)
    assert body["stereo"] == {"correlation": None, "side_to_mid_db": None}


def test_mono_has_no_side(client, tmp_path):
    mono = noise(3, 0.5, brightness=0.5, seed=6)[:, 0]
    body = post(client, write(tmp_path / "m.wav", mono)).json()

    assert body["format"]["channels"] == 1
    assert body["spectrum"]["side_db"] is None
    assert body["spectrum"]["relative_side_db"] is None
    assert body["stereo"] == {"correlation": None, "side_to_mid_db": None}
    assert len(body["spectrum"]["mid_db"]) == len(SPECTRUM_BANDS_HZ)


def test_waveform_peak_of_a_minus_6_dbfs_sine(client, tmp_path):
    body = post(client, write(tmp_path / "s.wav", sine(3, -6.0))).json()

    assert abs(max(body["waveform"]["peak"]) - 0.501) <= 0.001
    # A sine's RMS is its peak over root two.
    assert abs(np.median(body["waveform"]["rms"]) - 0.501 / math.sqrt(2)) <= 0.002
    # Dual mono: perfectly correlated, no side at all.
    assert body["stereo"] == {"correlation": 1.0, "side_to_mid_db": None}


def test_same_tone_reads_the_same_at_any_sample_rate(client, tmp_path):
    readings = []
    for rate in (44_100, 96_000):
        path = write(tmp_path / f"{rate}.wav", sine(3, -12.0, rate=rate), rate=rate)
        readings.append(post(client, path).json()["spectrum"]["mid_db"])
    one_k = SPECTRUM_BANDS_HZ.tolist().index(1000.0)

    assert abs(readings[0][one_k] - readings[1][one_k]) <= 1.0


def test_long_files_are_capped(client, tmp_path):
    body = post(client, write(tmp_path / "long.wav", sine(30, -20.0))).json()

    assert body["waveform"]["points"] == 1000
    assert body["spectrogram"]["frames"] == len(body["spectrogram"]["db"]) == 300
    assert math.isclose(body["spectrogram"]["seconds_per_frame"], 0.1)


def test_very_short_file(client, tmp_path):
    body = post(client, write(tmp_path / "short.wav", sine(0.01, -6.0))).json()

    assert body["waveform"]["points"] == 480
    assert body["spectrogram"]["frames"] == 1
    assert body["integrated_lufs"] is None  # under one 400 ms gating block


def test_unreadable_upload_is_422(client, tmp_path):
    path = tmp_path / "junk.wav"
    path.write_bytes(b"not audio at all")
    response = post(client, path)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unreadable_audio"


def test_oversized_upload_is_413(client, tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 100)
    response = post(client, write(tmp_path / "big.wav", sine(1, -20.0)))

    assert response.status_code == 413
