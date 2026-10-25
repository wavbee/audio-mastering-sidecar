from __future__ import annotations

import io

import soundfile as sf

from app.measure import measure
from tests.conftest import noise, write


def post(client, target, reference, **form):
    with open(target, "rb") as t, open(reference, "rb") as r:
        return client.post(
            "/match",
            files={"target": ("t.wav", t, "audio/wav"), "reference": ("r.wav", r, "audio/wav")},
            data={k: str(v) for k, v in form.items()},
        )


def loudness(path):
    return measure(str(path))["libebur128"]["integrated_lufs"]


def test_target_moves_toward_the_reference(client, tmp_path):
    target = write(tmp_path / "target.wav", noise(20, 0.08, brightness=0.1, seed=1))
    reference = write(tmp_path / "reference.wav", noise(20, 0.9, brightness=0.9, seed=2))
    response = post(client, target, reference)

    assert response.status_code == 200
    assert response.headers["x-output-sample-rate"] == "48000"
    out = tmp_path / "out.wav"
    out.write_bytes(response.content)
    info = sf.info(str(out))
    assert (info.samplerate, info.subtype) == (48000, "PCM_24")
    # Matched: the output sits much closer to the reference's loudness than
    # the target did.
    gap_before = abs(loudness(target) - loudness(reference))
    gap_after = abs(loudness(out) - loudness(reference))
    assert gap_after < gap_before / 2


def test_sixteen_bit_output_on_request(client, tmp_path):
    target = write(tmp_path / "target.wav", noise(20, 0.1, brightness=0.2, seed=3))
    reference = write(tmp_path / "reference.wav", noise(20, 0.8, brightness=0.8, seed=4))
    response = post(client, target, reference, bit_depth=16)

    assert sf.info(io.BytesIO(response.content)).subtype == "PCM_16"


def test_thirty_two_bit_output_is_refused(client, tmp_path):
    target = write(tmp_path / "target.wav", noise(20, 0.1, brightness=0.2, seed=5))
    reference = write(tmp_path / "reference.wav", noise(20, 0.8, brightness=0.8, seed=6))
    response = post(client, target, reference, bit_depth=32)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "match_refused"


def test_matchering_refusals_surface_as_422(client, tmp_path):
    same = write(tmp_path / "same.wav", noise(20, 0.5, brightness=0.5, seed=7))
    response = post(client, same, same)  # Matchering refuses target == reference

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "match_refused"


def test_oversized_reference_is_413(client, tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 100)
    target = write(tmp_path / "target.wav", noise(1, 0.1, brightness=0.2, seed=8))
    response = post(client, target, target)

    assert response.status_code == 413
