from __future__ import annotations

import io
import json

import numpy as np
import soundfile as sf

from app.analyze import SPECTRUM_BANDS_HZ, analyze
from tests.conftest import noise, sine, write


def curve_of(path) -> dict:
    """A target curve exactly as a caller would take it from /analyze."""
    spectrum = analyze(str(path))["spectrum"]
    return {
        "bands_hz": spectrum["bands_hz"],
        "mid_db": spectrum["relative_mid_db"],
        "side_db": spectrum["relative_side_db"],
    }


def post(client, path, curve, **form):
    data = {"target_curve": curve if isinstance(curve, str) else json.dumps(curve)}
    data |= {k: str(v) for k, v in form.items()}
    with open(path, "rb") as f:
        return client.post("/eq_match", files={"file": ("x.wav", f, "audio/wav")}, data=data)


def saved(response, tmp_path, name="out.wav") -> str:
    out = tmp_path / name
    out.write_bytes(response.content)
    return str(out)


def error_100_to_10k(path, target) -> float:
    bands = SPECTRUM_BANDS_HZ
    own = np.array(analyze(str(path))["spectrum"]["relative_mid_db"])
    in_range = (bands >= 100) & (bands <= 10_000)
    return float(np.abs(own - np.array(target["mid_db"]))[in_range].mean())


def test_dark_noise_moves_toward_a_bright_curve(client, tmp_path):
    dark = write(tmp_path / "dark.wav", noise(4, 0.3, brightness=0.2, seed=1))
    target = curve_of(write(tmp_path / "bright.wav", noise(4, 0.5, brightness=0.8, seed=2)))
    response = post(client, dark, target, max_gain_db=12)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-analysis-version"] == "2"
    out = saved(response, tmp_path)
    assert error_100_to_10k(out, target) <= 0.5 * error_100_to_10k(dark, target)


def test_output_keeps_length_rate_and_alignment(client, tmp_path):
    source = noise(3, 0.3, brightness=0.3, seed=3)
    path = write(tmp_path / "in.wav", source)
    target = curve_of(write(tmp_path / "ref.wav", noise(3, 0.5, brightness=0.7, seed=4)))
    out = saved(post(client, path, target), tmp_path)

    result, rate = sf.read(out, always_2d=True)
    assert rate == 48_000
    assert result.shape == source.shape
    assert sf.info(out).subtype == "PCM_24"
    # Linear phase with the delay removed: the output lines up with the input.
    a, b = source[:, 0], result[:, 0]
    lags = range(-50, 51)
    best = max(lags, key=lambda lag: float(np.dot(a[100:-100], np.roll(b, -lag)[100:-100])))
    assert best == 0


def test_strength_zero_returns_the_input(client, tmp_path):
    source = noise(2, 0.3, brightness=0.3, seed=5)
    path = write(tmp_path / "in.wav", source)
    target = curve_of(write(tmp_path / "ref.wav", noise(2, 0.5, brightness=0.8, seed=6)))
    response = post(client, path, target, strength=0)

    original, _ = sf.read(path, always_2d=True)
    result, _ = sf.read(saved(response, tmp_path), always_2d=True)
    assert np.abs(result - original).max() < 1e-3
    applied = json.loads(response.headers["x-applied-curve"])
    assert set(applied["mid_db"]) == {0.0}
    assert response.headers["x-output-gain-db"] == "0.0"


def test_applied_curve_never_boosts_the_extremes(client, tmp_path):
    path = write(tmp_path / "in.wav", noise(2, 0.3, brightness=0.5, seed=7))
    flat = curve_of(path)
    boost_all = {**flat, "mid_db": [v + 20 for v in flat["mid_db"]], "side_db": None}
    response = post(client, path, boost_all, max_gain_db=12)

    applied = json.loads(response.headers["x-applied-curve"])
    assert applied["bands_hz"] == flat["bands_hz"]
    bands = np.array(applied["bands_hz"])
    for key in ("mid_db", "side_db"):  # stereo: side falls back to the mid curve
        gains = np.array(applied[key])
        assert np.all(gains[bands < 30] <= 0)
        assert np.all(gains[bands > 16_000] <= 0)
        assert np.all(gains <= 12)
    # +20 dB everywhere is clamped to +12 in the middle of the band.
    assert np.array(applied["mid_db"])[(bands > 100) & (bands < 10_000)].min() >= 11.5


def test_overs_are_scaled_down_and_reported(client, tmp_path):
    path = write(tmp_path / "in.wav", noise(2, 0.9, brightness=0.5, seed=8))
    flat = curve_of(path)
    louder = {**flat, "mid_db": [v + 6 for v in flat["mid_db"]]}
    response = post(client, path, louder)

    gain = float(response.headers["x-output-gain-db"])
    assert gain < 0
    result, _ = sf.read(saved(response, tmp_path))
    assert np.abs(result).max() <= 0.99 + 1e-6


def test_mono_input_works(client, tmp_path):
    path = write(tmp_path / "mono.wav", noise(2, 0.3, brightness=0.2, seed=9)[:, 0])
    target = curve_of(write(tmp_path / "ref.wav", noise(2, 0.5, brightness=0.8, seed=10)))
    response = post(client, path, target, bit_depth=16)

    assert response.status_code == 200
    info = sf.info(io.BytesIO(response.content))
    assert (info.channels, info.subtype, info.frames) == (1, "PCM_16", 2 * 48_000)
    assert json.loads(response.headers["x-applied-curve"])["side_db"] is None


def test_invalid_curves_are_422(client, tmp_path):
    path = write(tmp_path / "in.wav", noise(1, 0.3, brightness=0.5, seed=11))
    good = curve_of(path)
    shifted = {**good, "bands_hz": [f * 1.02 for f in good["bands_hz"]]}
    short = {**good, "mid_db": good["mid_db"][:-1]}
    infinite = {**good, "mid_db": [1e400] * len(good["mid_db"])}
    side_short = {**good, "side_db": [0.0]}

    for curve in ("not json", "[]", shifted, short, infinite, side_short):
        response = post(client, path, curve)
        assert response.status_code == 422, curve
        assert response.json()["error"]["code"] == "invalid_curve"
    with open(path, "rb") as f:
        missing = client.post("/eq_match", files={"file": ("x.wav", f, "audio/wav")})
    assert missing.json()["error"]["code"] == "invalid_curve"


def test_out_of_range_parameters_are_422(client, tmp_path):
    path = write(tmp_path / "in.wav", noise(1, 0.3, brightness=0.5, seed=12))
    good = curve_of(path)

    for form in ({"strength": 1.5}, {"max_gain_db": 13}, {"bit_depth": 32}, {"strength": "x"}):
        response = post(client, path, good, **form)
        assert response.status_code == 422, form
        assert response.json()["error"]["code"] == "invalid_parameter"


def test_silent_or_unreadable_audio_is_422(client, tmp_path):
    good = curve_of(write(tmp_path / "ref.wav", sine(1, -20.0)))
    silent = write(tmp_path / "silent.wav", np.zeros((48_000, 2)))
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio at all")

    for path in (silent, junk):
        response = post(client, path, good)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "unreadable_audio"


def test_oversized_upload_is_413(client, tmp_path, monkeypatch):
    path = write(tmp_path / "in.wav", sine(1, -20.0))
    good = curve_of(path)
    monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 100)

    assert post(client, path, good).status_code == 413
