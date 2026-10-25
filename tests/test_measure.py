from __future__ import annotations

import numpy as np

from tests.conftest import sine, write


def post(client, path):
    with open(path, "rb") as f:
        return client.post("/measure", files={"file": ("x.wav", f, "audio/wav")})


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["matchering_version"] == "2.0.6" or body["matchering_version"].count(".") == 2
    assert body["pyloudnorm_version"] != "unknown"
    assert body["libebur128_version"].count(".") == 2


def test_ebu_tech_3341_case_1_reads_minus_23(client, tmp_path):
    """Tech 3341 test 1: 1 kHz stereo sine at -23 dBFS reads -23.0 +/-0.1 LUFS."""
    body = post(client, write(tmp_path / "t1.wav", sine(20, -23.0))).json()

    assert abs(body["libebur128"]["integrated_lufs"] - -23.0) <= 0.1
    assert abs(body["pyloudnorm"]["integrated_lufs"] - -23.0) <= 0.1
    assert body["libebur128"]["version"].count(".") == 2  # "1.2.6", not "1...2...6"
    assert all(abs(p - -23.0) <= 0.2 for p in body["libebur128"]["true_peak_dbtp"])
    assert body["format"] == {
        "sample_rate": 48000,
        "channels": 2,
        "subtype": "PCM_24",
        "duration_s": 20.0,
    }


def test_true_peak_sees_between_samples(client, tmp_path):
    """Tech 3341 test 16: fs/4 at 45 degrees — sample peak about -9 dB, true peak -6.

    Written as float only because it is a measurement oracle; nothing this
    service outputs is ever wider than 24-bit."""
    t = np.arange(48_000 * 2) / 48_000
    tone = 0.5 * np.sin(2 * np.pi * 12_000 * t + np.pi / 4)
    body = post(
        client, write(tmp_path / "t16.wav", np.column_stack([tone, tone]), subtype="FLOAT")
    ).json()

    assert all(-6.4 <= p <= -5.8 for p in body["libebur128"]["true_peak_dbtp"])
    assert all(p < -8.5 for p in body["libebur128"]["sample_peak_dbfs"])


def test_silence_is_null_not_infinity(client, tmp_path):
    body = post(client, write(tmp_path / "s.wav", np.zeros((48_000 * 2, 2)))).json()

    assert body["libebur128"]["integrated_lufs"] is None
    assert body["pyloudnorm"]["integrated_lufs"] is None


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
