from __future__ import annotations

import math
import re

import numpy as np

from app.analyze import SPECTRUM_BANDS_HZ
from tests.conftest import RATE, noise, sine, write


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

    assert body["analysis_version"] == 2
    assert body["format"] == {"sample_rate": 48000, "channels": 2, "duration_s": 3.0}
    assert isinstance(body["integrated_lufs"], float)

    wave = body["waveform"]
    assert wave["points"] == 4000
    assert len(wave["peak"]) == len(wave["rms"]) == len(wave["true_peak"]) == 4000
    assert math.isclose(wave["seconds_per_point"], 0.00075)
    assert all(0 <= r <= p <= 1 for p, r in zip(wave["peak"], wave["rms"], strict=True))
    assert all(t >= p for p, t in zip(wave["peak"], wave["true_peak"], strict=True))

    spectrum = body["spectrum"]
    assert spectrum["bands_hz"] == [round(f, 2) for f in SPECTRUM_BANDS_HZ]
    assert spectrum["bands_hz"][0] >= 20 and spectrum["bands_hz"][-1] <= 20_000
    for key in ("mid_db", "side_db", "relative_mid_db", "relative_side_db"):
        assert len(spectrum[key]) == len(spectrum["bands_hz"])

    gram = body["spectrogram"]
    assert gram["frames"] == len(gram["db"]) == 1 + (3 * 48_000 - 8192) // 4096
    assert gram["bands_hz"] == spectrum["bands_hz"]
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


def test_true_peak_reads_an_inter_sample_peak_over_full_scale(client, tmp_path):
    """A sine at fs/4 with a 45 degree phase lands every sample at 0.707 of
    its amplitude. Sampled at -0.5 dBFS, its true peak is 3 dB higher:
    about +2.5 dBTP, which only oversampling can see."""
    n = np.arange(3 * RATE)
    amplitude = 10 ** (-0.5 / 20) / math.sin(math.pi / 4)
    tone = amplitude * np.sin(np.pi / 2 * n + np.pi / 4)
    wave = post(client, write(tmp_path / "isp.wav", np.column_stack([tone, tone]))).json()[
        "waveform"
    ]

    assert abs(max(wave["peak"]) - 10 ** (-0.5 / 20)) <= 0.001
    inner = wave["true_peak"][10:-10]  # away from the edges' ringing
    assert all(abs(20 * math.log10(t) - 2.51) <= 0.05 for t in inner)
    assert all(t > 1.0 >= p for p, t in zip(wave["peak"], wave["true_peak"], strict=True))


def test_true_peak_does_not_depend_on_block_edges(client, tmp_path, monkeypatch):
    path = write(tmp_path / "n.wav", noise(2, 0.9, brightness=0.9, seed=7), subtype="FLOAT")
    whole = post(client, path).json()["waveform"]["true_peak"]
    monkeypatch.setattr("app.analyze._BLOCK", 1000)

    assert post(client, path).json()["waveform"]["true_peak"] == whole


def test_same_tone_reads_the_same_at_any_sample_rate(client, tmp_path):
    readings = []
    for rate in (44_100, 96_000):
        path = write(tmp_path / f"{rate}.wav", sine(3, -12.0, rate=rate), rate=rate)
        readings.append(post(client, path).json()["spectrum"]["mid_db"])
    one_k = SPECTRUM_BANDS_HZ.tolist().index(1000.0)

    assert abs(readings[0][one_k] - readings[1][one_k]) <= 1.0


def test_long_files_are_capped(client, tmp_path):
    body = post(client, write(tmp_path / "long.wav", sine(30, -20.0))).json()

    assert body["waveform"]["points"] == 4000
    # 30 s at 48 kHz is 350 STFT frames: under the cap, so all of them.
    assert body["spectrogram"]["frames"] == len(body["spectrogram"]["db"]) == 350


def test_spectrogram_is_capped_at_1200_frames(client, tmp_path):
    """1201 hops of 4096 plus one frame is the first length over the cap; a
    low-rate mono file keeps that quick."""
    rate = 8_000
    tone = sine((8192 + 1201 * 4096) / rate, -20.0, freq=500.0, rate=rate)[:, 0]
    gram = post(client, write(tmp_path / "long.wav", tone, rate=rate, subtype="PCM_16")).json()[
        "spectrogram"
    ]

    assert gram["frames"] == len(gram["db"]) == 1200
    assert gram["bands_hz"] == [round(f, 2) for f in SPECTRUM_BANDS_HZ]
    assert len(gram["bands_hz"]) == 59
    assert all(len(row) == 59 for row in gram["db"])


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


# Clipping --------------------------------------------------------------------


def clip_at(data, runs, rate=RATE):
    """Write full-scale runs into a copy: (channel, start second, length)."""
    out = data.copy()
    for channel, second, length in runs:
        start = int(second * rate)
        out[start : start + length, channel] = 1.0
    return out


def test_clean_file_has_no_clipping(client, tmp_path):
    clipping = post(client, write(tmp_path / "c.wav", sine(2, -1.0))).json()["clipping"]

    assert clipping == {
        "full_scale": 0.9999,
        "min_run": 3,
        "events": [],
        "events_total": 0,
        "clipped_samples": 0,
    }


def test_overdriven_sine_saturating_24_bit_pcm(client, tmp_path):
    """A 5 Hz sine driven to 2/sqrt(3) of full scale, saturated by the 24-bit
    format (2^23 - 1 reads 0.99999988), for half a second in the middle of a
    quiet file. |sine| >= sqrt(3)/2 from 60 to 120 degrees of each half
    cycle: a 33.3 ms flat top around each crest, 66.7 ms from the next, on
    both channels — so one two-channel event per crest."""
    seconds, burst = 2.0, (0.7, 1.2)
    t = np.arange(int(seconds * RATE)) / RATE
    signal = 0.1 * np.sin(2 * np.pi * 5 * t)
    inside = (t >= burst[0]) & (t < burst[1])
    drive = 2 / math.sqrt(3)
    signal[inside] = np.clip(drive * np.sin(2 * np.pi * 5 * t[inside]), -1.0, 1.0)
    clipping = post(client, write(tmp_path / "o.wav", np.column_stack([signal, signal]))).json()[
        "clipping"
    ]

    events = clipping["events"]
    assert clipping["events_total"] == len(events) == 5  # crests at 0.75, 0.85 .. 1.15 s
    for k, event in enumerate(events):
        assert event["channels"] == [0, 1]
        assert abs(event["start_s"] - (0.75 + 0.1 * k - 1 / 60)) <= 0.001
        assert abs(event["end_s"] - (0.75 + 0.1 * k + 1 / 60)) <= 0.001
        assert abs(event["samples"] - 2 * RATE / 30) <= 2 * 48
        assert event["peak_dbfs"] == 0.0
    assert clipping["clipped_samples"] == sum(e["samples"] for e in events)


def test_two_full_scale_samples_are_not_a_clip(client, tmp_path):
    quiet = sine(1, -20.0)
    clipping = post(client, write(tmp_path / "t.wav", clip_at(quiet, [(0, 0.5, 2)]))).json()[
        "clipping"
    ]

    assert clipping["events"] == [] and clipping["clipped_samples"] == 0


def test_runs_closer_than_50_ms_merge(client, tmp_path):
    quiet = sine(2, -20.0)
    runs = [(0, 0.2, 3), (1, 0.23, 4), (0, 1.0, 3), (0, 1.08, 5)]
    clipping = post(client, write(tmp_path / "m.wav", clip_at(quiet, runs))).json()["clipping"]

    assert [(e["start_s"], e["channels"], e["samples"]) for e in clipping["events"]] == [
        (0.2, [0, 1], 7),  # 30 ms apart, different channels: one event
        (1.0, [0], 3),  # 80 ms apart: two
        (1.08, [0], 5),
    ]
    assert clipping["events"][0]["end_s"] == round(0.23 + 4 / RATE, 4)
    assert clipping["clipped_samples"] == 15


def test_run_across_a_block_edge_counts_once(client, tmp_path, monkeypatch):
    monkeypatch.setattr("app.analyze._BLOCK", 4800)  # a block every 100 ms
    quiet = sine(1, -20.0)
    edge = 5 * 4800
    quiet[edge - 2 : edge + 3, 1] = -1.0  # five samples, two before the edge
    quiet[2 * 4800 - 1 : 2 * 4800 + 1, 0] = 1.0  # two samples astride an edge: no clip
    quiet[-3:, 0] = 1.0  # a run ending with the file
    clipping = post(client, write(tmp_path / "e.wav", quiet)).json()["clipping"]

    assert [(e["start_s"], e["channels"], e["samples"]) for e in clipping["events"]] == [
        (round((edge - 2) / RATE, 4), [1], 5),
        (round((RATE - 3) / RATE, 4), [0], 3),
    ]
    assert clipping["events"][-1]["end_s"] == 1.0


def test_only_the_right_channel_clips(client, tmp_path):
    quiet = sine(1, -20.0)
    clipping = post(client, write(tmp_path / "r.wav", clip_at(quiet, [(1, 0.4, 10)]))).json()[
        "clipping"
    ]

    assert [e["channels"] for e in clipping["events"]] == [[1]]
    assert clipping["events"][0]["samples"] == 10


def test_events_are_capped_at_500_largest(client, tmp_path):
    """520 runs 100 ms apart; the 20 shortest are dropped from the list but
    not from the count, and the rest stay in time order."""
    rate = 8_000
    quiet = sine(53, -20.0, rate=rate)[:, :1]
    lengths = [3 if k % 26 == 0 else 4 for k in range(520)]
    runs = [(0, 0.1 * k + 0.05, n) for k, n in enumerate(lengths)]
    path = write(tmp_path / "cap.wav", clip_at(quiet, runs, rate=rate), rate=rate)
    clipping = post(client, path).json()["clipping"]

    assert clipping["events_total"] == 520
    assert len(clipping["events"]) == 500
    assert all(e["samples"] == 4 for e in clipping["events"])
    starts = [e["start_s"] for e in clipping["events"]]
    assert starts == sorted(starts)
    assert clipping["clipped_samples"] == sum(lengths)
