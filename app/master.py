"""Reference-matched mastering with Matchering.

Matchering measures a reference track's frequency response, RMS and peak
behaviour and applies them to the target. That is the whole job here: the
caller decides the final loudness and true-peak ceiling afterwards, so this
module returns the matched audio and nothing about whether it is "good".

Two choices that are policy-free but deliberate:
  * Matchering works at an internal sample rate. It is set to the TARGET's own
    rate, so a 48 kHz master is not silently returned at 44.1 kHz.
  * Output is 16- or 24-bit PCM only — Matchering offers nothing wider, which
    suits delivery (32-bit masters are not accepted).
"""

from __future__ import annotations

import matchering as mg
import soundfile as sf
from matchering.log.exceptions import ModuleError

BIT_DEPTHS = {16: mg.pcm16, 24: mg.pcm24}


class MatchError(Exception):
    """Matchering refused the input (too short, too long, silent, identical...)."""


def match(target: str, reference: str, output: str, bit_depth: int = 24) -> dict:
    if bit_depth not in BIT_DEPTHS:
        raise MatchError(f"bit_depth must be 16 or 24, got {bit_depth}")

    rate = sf.info(target).samplerate
    config = mg.Config(internal_sample_rate=rate)
    try:
        mg.process(
            target=target,
            reference=reference,
            results=[BIT_DEPTHS[bit_depth](output)],
            config=config,
        )
    except (ModuleError, ValueError, RuntimeError) as error:
        raise MatchError(str(error)) from error
    return {"sample_rate": rate, "bit_depth": bit_depth}
