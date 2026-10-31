from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from tests.conftest import RATE, sine, write

GLUE = {"threshold_db": -20, "ratio": 4, "attack_ms": 10, "release_ms": 100, "mix": 1}


def post(client, path, **settings):
    data = {k: str(v) for k, v in {**GLUE, **settings}.items()}
    with open(path, "rb") as f:
        return client.post("/compress", files={"file": ("x.wav", f, "audio/wav")}, data=data)


def saved(response, tmp_path):
    out = tmp_path / "out.wav"
    out.write_bytes(response.content)
    return sf.read(str(out), always_2d=True)


def quiet_then_loud(tmp_path):
    """One second at -30 dBFS (under the threshold), then one at -6 dBFS
    (14 dB over it), with the right channel at half the left's level."""
    audio = np.concatenate([sine(1.0, -30), sine(1.0, -6)])
    audio[:, 1] *= 0.5
    return write(tmp_path / "in.wav", audio), audio


def test_turns_down_only_what_is_over_the_threshold(client, tmp_path):
    path, source = quiet_then_loud(tmp_path)

    response = post(client, path)
    out, rate = saved(response, tmp_path)

    assert response.status_code == 200
    assert rate == RATE and out.shape == source.shape
    assert np.allclose(
        out[: RATE // 2], source[: RATE // 2], atol=1e-4
    )  # the quiet second is untouched
    tail = slice(int(1.5 * RATE), 2 * RATE)
    reduction = 20 * np.log10(np.max(np.abs(source[tail, 0])) / np.max(np.abs(out[tail, 0])))
    assert 8 < reduction < 12  # 14 dB over at 4:1 leaves 3.5 dB over: ~10.5 dB down
    details = json.loads(response.headers["X-Compression"])
    assert details["max_reduction_db"] > 8 and 0 < details["reduced_share"] < 1


def test_moves_both_channels_together(client, tmp_path):
    path, _ = quiet_then_loud(tmp_path)

    out, _ = saved(post(client, path), tmp_path)

    loud = out[int(1.5 * RATE) :]
    assert np.allclose(loud[:, 1], loud[:, 0] * 0.5, atol=1e-4)  # the image is kept


def test_mix_zero_returns_the_input(client, tmp_path):
    path, source = quiet_then_loud(tmp_path)

    out, _ = saved(post(client, path, mix=0), tmp_path)

    assert np.allclose(out, source, atol=1e-6)


def test_refuses_a_setting_out_of_range(client, tmp_path):
    path, _ = quiet_then_loud(tmp_path)

    response = post(client, path, ratio=40)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_parameter"
    assert "ratio" in response.json()["error"]["message"]
