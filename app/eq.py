"""Static EQ toward a caller-supplied tonal-balance curve.

The caller sends a target in the same convention ``/analyze`` reports
(``relative_*_db``: 1/6-octave band level minus integrated loudness). Where
that target comes from — a reference master, a genre average, a hand-drawn
curve — is the caller's business; this module only moves the file toward it
and says exactly how far it moved.

Choices that are deliberate rather than policy:
  * The file's own curve is measured by the same code as ``/analyze``, so a
    target taken from ``/analyze`` means the same thing here.
  * The difference is smoothed across three bands. A 1/6-octave measurement
    of real music has band-to-band ripple that is not tonal balance, and
    chasing it would be a comb of narrow boosts and cuts.
  * Nothing is ever boosted below 30 Hz or above 16 kHz. Lifting rumble or
    the band where codecs and resamplers roll off amplifies what is least
    likely to be music. Cuts there are allowed.
  * Gains are bounded by the caller's ``max_gain_db`` and scaled by
    ``strength``; the gains actually applied are returned, rounded exactly as
    they were used.
  * The filter is linear-phase FIR, with its delay removed, so the output is
    sample-aligned with the input and the same length — a caller can null it
    against the original. Mid and side are filtered separately for stereo.
  * If the result would exceed 0.99 sample peak it is scaled down as a whole
    (never limited) and the gain is reported. Loudness and true-peak policy
    stay with the caller.

The audio is streamed through in blocks (overlap-add), so long high-rate
files never sit in memory whole.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import soundfile as sf
from scipy.signal import firwin2, oaconvolve

from app.analyze import SPECTRUM_BANDS_HZ, relative, scan

NO_BOOST_BELOW_HZ = 30.0
NO_BOOST_ABOVE_HZ = 16_000.0
MAX_GAIN_DB_LIMIT = 12.0
PEAK_CEILING = 0.99
SUBTYPES = {16: "PCM_16", 24: "PCM_24"}

# 4097 taps at 44.1 kHz is about 10.8 Hz of resolution; the tap count scales
# with the rate so 96 kHz keeps the same resolution.
_BASE_TAPS = 4097
_BASE_RATE = 44_100
_BAND_TOLERANCE = 0.01
_BLOCK = 1 << 18


class CurveError(ValueError):
    """The target curve is malformed or not on the analysis band grid."""


class ParameterError(ValueError):
    """strength, max_gain_db or bit_depth is out of range."""


class SilentError(ValueError):
    """The file has no integrated loudness, so it has no relative curve."""


def _levels(value, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != len(SPECTRUM_BANDS_HZ):
        raise CurveError(f"{name} must be a list of {len(SPECTRUM_BANDS_HZ)} numbers")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in value):
        raise CurveError(f"{name} must hold numbers only")
    levels = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(levels)):
        raise CurveError(f"{name} must hold finite numbers only")
    return levels


def parse_curve(raw: str | None) -> dict:
    """{"mid": array, "side": array | None} from the ``target_curve`` field."""
    if not raw:
        raise CurveError("target_curve is required")
    try:
        curve = json.loads(raw)
    except ValueError as error:
        raise CurveError(f"target_curve is not JSON: {error}") from error
    if not isinstance(curve, dict):
        raise CurveError("target_curve must be an object")

    bands = _levels(curve.get("bands_hz"), "bands_hz")
    if np.any(np.abs(bands / SPECTRUM_BANDS_HZ - 1) > _BAND_TOLERANCE):
        raise CurveError("bands_hz must be the /analyze 1/6-octave grid (within 1%)")
    side = curve.get("side_db")
    return {
        "mid": _levels(curve.get("mid_db"), "mid_db"),
        "side": None if side is None else _levels(side, "side_db"),
    }


def parse_parameters(strength: str, max_gain_db: str, bit_depth: str) -> tuple[float, float, int]:
    try:
        strength_value, max_gain_value = float(strength), float(max_gain_db)
        depth = int(bit_depth)
    except ValueError as error:
        raise ParameterError(str(error)) from error
    if not 0.0 <= strength_value <= 1.0:
        raise ParameterError(f"strength must be within 0..1, got {strength}")
    if not 0.0 <= max_gain_value <= MAX_GAIN_DB_LIMIT:
        raise ParameterError(
            f"max_gain_db must be within 0..{MAX_GAIN_DB_LIMIT:g}, got {max_gain_db}"
        )
    if depth not in SUBTYPES:
        raise ParameterError(f"bit_depth must be 16 or 24, got {bit_depth}")
    return strength_value, max_gain_value, depth


def gains(target: np.ndarray, own: np.ndarray, max_gain_db: float, strength: float) -> np.ndarray:
    """Per-band gain in dB, rounded to the 2 dp that is both used and reported."""
    diff = target - own
    window = np.ones(3)
    # Centred 3-band average; the end bands average the two values they have.
    smooth = np.convolve(diff, window, "same") / np.convolve(np.ones_like(diff), window, "same")
    clamped = np.clip(smooth, -max_gain_db, max_gain_db)
    guarded = (SPECTRUM_BANDS_HZ < NO_BOOST_BELOW_HZ) | (SPECTRUM_BANDS_HZ > NO_BOOST_ABOVE_HZ)
    clamped[guarded] = np.minimum(clamped[guarded], 0.0)
    return np.round(clamped * strength, 2) + 0.0  # + 0.0: no "-0.0" in the header


def numtaps(rate: int) -> int:
    taps = round(_BASE_TAPS * rate / _BASE_RATE)
    return taps if taps % 2 else taps + 1


def design(gains_db: np.ndarray, rate: int) -> np.ndarray:
    """Linear-phase FIR whose response follows ``gains_db`` interpolated in
    log-frequency between band centres and held flat beyond the end bands."""
    nyquist = rate / 2
    freq = np.concatenate(([0.0], np.geomspace(10.0, nyquist, 2048)))
    freq[-1] = nyquist
    curve_db = np.interp(np.log2(np.maximum(freq, 10.0)), np.log2(SPECTRUM_BANDS_HZ), gains_db)
    return firwin2(numtaps(rate), freq, 10 ** (curve_db / 20), fs=rate)


def _convolve(block: np.ndarray, mid_fir: np.ndarray, side_fir: np.ndarray | None) -> np.ndarray:
    if side_fir is None:  # mono (or more than two channels): one curve for all
        return oaconvolve(block, mid_fir[:, None], axes=0)
    mid = oaconvolve((block[:, 0] + block[:, 1]) / 2, mid_fir)
    side = oaconvolve((block[:, 0] - block[:, 1]) / 2, side_fir)
    return np.column_stack([mid + side, mid - side])


def _filter(path: str, scratch: str, mid_fir: np.ndarray, side_fir: np.ndarray | None) -> float:
    """Filter ``path`` into a float ``scratch`` file; return its sample peak."""
    info = sf.info(path)
    overlap = len(mid_fir) - 1
    delay = overlap // 2
    tail = np.zeros((overlap, info.channels))
    state = {"skip": delay, "left": info.frames, "peak": 0.0}

    with (
        sf.SoundFile(path) as source,
        sf.SoundFile(scratch, "w", info.samplerate, info.channels, "FLOAT") as out,
    ):

        def emit(samples: np.ndarray) -> None:
            # Drop the filter's group delay, then stop at the input's length.
            cut = min(state["skip"], len(samples))
            state["skip"] -= cut
            samples = samples[cut : cut + state["left"]]
            if len(samples):
                state["left"] -= len(samples)
                state["peak"] = max(state["peak"], float(np.abs(samples).max()))
                out.write(samples)

        for block in source.blocks(blocksize=_BLOCK, dtype="float64", always_2d=True):
            filtered = _convolve(block, mid_fir, side_fir)
            filtered[:overlap] += tail
            tail = filtered[len(block) :].copy()
            emit(filtered[: len(block)])
        emit(tail)
    return state["peak"]


def _write(scratch: str, output: str, scale: float, bit_depth: int) -> None:
    info = sf.info(scratch)
    with (
        sf.SoundFile(scratch) as source,
        sf.SoundFile(output, "w", info.samplerate, info.channels, SUBTYPES[bit_depth]) as out,
    ):
        for block in source.blocks(blocksize=_BLOCK, dtype="float64", always_2d=True):
            out.write(block * scale)


def eq_match(
    path: str,
    output: str,
    curve: dict,
    strength: float = 1.0,
    max_gain_db: float = 6.0,
    bit_depth: int = 24,
) -> dict:
    acc, lufs = scan(path)
    if lufs is None:
        raise SilentError("the file has no integrated loudness (silent or under 400 ms)")
    own_mid, own_side = (relative(levels, lufs) for levels in acc.spectrum_db())

    mid_gains = gains(curve["mid"], own_mid, max_gain_db, strength)
    side_gains = None
    if acc.channels == 2:
        side_gains = (
            gains(curve["side"], own_side, max_gain_db, strength)
            if curve["side"] is not None and own_side is not None
            else mid_gains
        )

    mid_fir = design(mid_gains, acc.rate)
    side_fir = None if side_gains is None else design(side_gains, acc.rate)
    scratch = os.path.join(os.path.dirname(output), "filtered.wav")
    try:
        peak = _filter(path, scratch, mid_fir, side_fir)
        scale = PEAK_CEILING / peak if peak > PEAK_CEILING else 1.0
        _write(scratch, output, scale, bit_depth)
    finally:
        if os.path.exists(scratch):
            os.remove(scratch)

    return {
        "applied_curve": {
            "bands_hz": [round(float(f), 2) for f in SPECTRUM_BANDS_HZ],
            "mid_db": mid_gains.tolist(),
            "side_db": None if side_gains is None else side_gains.tolist(),
        },
        "output_gain_db": round(20 * math.log10(scale), 2) + 0.0,
    }
