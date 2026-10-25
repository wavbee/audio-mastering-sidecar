"""Synthesised test audio. Nothing binary is committed: every signal is built
here from its definition, so each fixture's properties are auditable."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app

RATE = 48_000


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def write(path, data, rate=RATE, subtype="PCM_24") -> str:
    sf.write(str(path), data, rate, subtype=subtype)
    return str(path)


def sine(seconds: float, dbfs: float, freq: float = 1000.0, rate: int = RATE) -> np.ndarray:
    """Stereo sine, in phase, at the given per-channel PEAK level — EBU Tech
    3341's definition, under which a 1 kHz tone's loudness equals its peak."""
    t = np.arange(int(seconds * rate)) / rate
    mono = 10 ** (dbfs / 20) * np.sin(2 * np.pi * freq * t)
    return np.column_stack([mono, mono])


def noise(
    seconds: float, gain: float, brightness: float, seed: int, rate: int = RATE
) -> np.ndarray:
    """Stereo noise with a controllable spectral tilt: `brightness` near 0 is
    dark (heavily low-passed), near 1 is close to white."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((int(seconds * rate), 2))
    out = np.empty_like(raw)
    alpha = 1 - brightness * 0.95
    out[0] = raw[0]
    for i in range(1, len(raw)):  # one-pole low-pass
        out[i] = alpha * out[i - 1] + (1 - alpha) * raw[i]
    return gain * out / np.max(np.abs(out))
