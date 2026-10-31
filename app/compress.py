"""Stereo-linked bus compression, for glue.

Pedalboard's ``Compressor`` (JUCE's) is used for its gain computer and
attack/release ballistics, but never on the audio itself: run per channel, it
would turn the left and right channels down by different amounts and move the
stereo image. Instead it runs once over a detector signal — the louder of the
channels at each sample — and the gain it applied to that signal is applied
to every channel alike. One gain for all channels is what a mastering bus
compressor does.

Choices that are deliberate rather than policy:
  * No make-up gain. The caller sets the loudness afterwards, and a
    compressor that also raised the level would make every comparison a
    comparison of levels.
  * ``mix`` blends the compressed signal with the original (parallel
    compression): 1.0 is fully compressed, 0.0 returns the input unchanged.
  * The output is 32-bit float, the same length and rate, sample-aligned with
    the input (the compressor has no look-ahead), so a caller can null it.
  * How hard it worked is measured, not estimated: the gain reduction is
    reported from the gain actually applied.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from pedalboard import Compressor

LIMITS = {
    "threshold_db": (-40.0, 0.0),
    "ratio": (1.0, 10.0),
    "attack_ms": (0.1, 100.0),
    "release_ms": (10.0, 1000.0),
    "mix": (0.0, 1.0),
}
_SILENT = 1e-9


class ParameterError(ValueError):
    """A compressor setting is not a number or is out of range."""


def parse_parameters(raw: dict[str, str]) -> dict[str, float]:
    values = {}
    for name, (low, high) in LIMITS.items():
        try:
            value = float(raw[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ParameterError(f"{name} must be a number") from error
        if not low <= value <= high:
            raise ParameterError(f"{name} must be within {low:g}..{high:g}, got {raw[name]}")
        values[name] = value
    return values


def compress(source: str, output: str, settings: dict[str, float]) -> dict:
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    detector = np.max(np.abs(audio), axis=1).astype(np.float32)

    compressor = Compressor(
        threshold_db=settings["threshold_db"],
        ratio=settings["ratio"],
        attack_ms=settings["attack_ms"],
        release_ms=settings["release_ms"],
    )
    compressed = compressor(detector[np.newaxis, :], rate)[0]
    gain = np.ones_like(detector)
    audible = detector > _SILENT
    gain[audible] = np.clip(compressed[audible] / detector[audible], 0.0, 1.0)

    mix = settings["mix"]
    blended = (1.0 - mix) + mix * gain
    sf.write(output, audio * blended[:, np.newaxis], rate, subtype="FLOAT")

    reduction_db = (
        -20.0 * np.log10(np.maximum(blended[audible], _SILENT)) if audible.any() else np.zeros(1)
    )
    return {
        "max_reduction_db": round(float(reduction_db.max()), 2) + 0.0,
        "mean_reduction_db": round(float(reduction_db.mean()), 2) + 0.0,
        "reduced_share": round(float((reduction_db > 0.5).mean()), 3) + 0.0,
    }
