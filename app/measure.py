"""Independent ITU-R BS.1770 / EBU R128 measurement.

Two implementations that share no code with each other or with ffmpeg:
  * libebur128 (via pyebur128) — integrated loudness, loudness range, and
    per-channel 4x-oversampled true peak;
  * pyloudnorm — a pure-numpy BS.1770-4 integrated-loudness meter.

Returned as raw numbers only. A caller that wants to know whether two meters
agree compares them itself; nothing here says what "agree" means.
"""

from __future__ import annotations

import math

import numpy as np
import pyebur128 as ebur
import pyloudnorm
import soundfile as sf

_MODES = (
    ebur.MeasurementMode.MODE_I
    | ebur.MeasurementMode.MODE_LRA
    | ebur.MeasurementMode.MODE_TRUE_PEAK
    | ebur.MeasurementMode.MODE_SAMPLE_PEAK
)


def _db(linear: float) -> float | None:
    return 20 * math.log10(linear) if linear > 0 else None


def _finite(value: float) -> float | None:
    """JSON has no infinity: an unmeasurable (silent) reading is null."""
    return round(value, 3) if value is not None and math.isfinite(value) else None


def measure(path: str) -> dict:
    data, rate = sf.read(path, dtype="float64", always_2d=True)
    info = sf.info(path)
    frames, channels = data.shape

    state = ebur.R128State(channels, rate, _MODES)
    state.add_frames(np.ascontiguousarray(data).ravel(), frames)
    true_peaks = [ebur.get_true_peak(state, c) for c in range(channels)]
    sample_peaks = [ebur.get_sample_peak(state, c) for c in range(channels)]

    return {
        "format": {
            "sample_rate": rate,
            "channels": channels,
            "subtype": info.subtype,
            "duration_s": round(frames / rate, 6),
        },
        "libebur128": {
            "version": str(ebur.get_libebur128_version()),
            "integrated_lufs": _finite(ebur.get_loudness_global(state)),
            "lra_lu": _finite(ebur.get_loudness_range(state)),
            "true_peak_dbtp": [_finite(_db(p)) for p in true_peaks],
            "sample_peak_dbfs": [_finite(_db(p)) for p in sample_peaks],
        },
        "pyloudnorm": {
            "integrated_lufs": _finite(pyloudnorm.Meter(rate).integrated_loudness(data)),
        },
    }
