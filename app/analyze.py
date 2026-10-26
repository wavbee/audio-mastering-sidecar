"""Raw shape of a recording: waveform overview, long-term spectrum, a
spectrogram and stereo image.

Everything is a measurement a caller can draw or compare; nothing here says
what a "good" spectrum or a "too wide" mix is.

Choices that matter to a caller comparing two files:
  * The spectrum is a power spectral DENSITY (per hertz, scipy's ``density``
    scaling), so the same audio at 44.1 kHz and 96 kHz reads the same band
    levels. A plain FFT magnitude would shift with the sample rate.
  * Frames quieter than -60 dBFS RMS (mid channel) are left out of the
    long-term spectrum. Intros, gaps and tails of digital silence would
    otherwise drag every band down by an amount that depends only on how
    long the silence lasts. If the whole file is that quiet, every frame is
    used, so there is still an answer.
  * ``relative_*_db`` subtracts the file's integrated loudness, so two
    masters at different levels can be compared on tonal balance alone.
    Those relative curves are what ``/eq_match`` takes as a target.
  * Bands narrower than one FFT bin (low bands at high sample rates) are
    filled by interpolating their neighbours in log-frequency. JSON has no
    NaN, and a hole in the curve is not a measurement either.
  * The waveform carries a true peak per point beside the sample peak: the
    largest value of the x4-oversampled signal, as BS.1770 defines true peak.
    It is not clipped at 1.0; an inter-sample peak over full scale is exactly
    what a converter or a lossy encoder will turn into distortion.

The file is read in blocks, so a ten-minute 96 kHz stereo file never sits in
memory as one float64 array.
"""

from __future__ import annotations

import math

import numpy as np
import pyebur128 as ebur
import soundfile as sf
from scipy.signal import get_window, resample_poly

ANALYSIS_VERSION = 2

FRAME = 8192
HOP = FRAME // 2
SILENCE_DBFS = -60.0
MAX_POINTS = 4000
MAX_SPECTROGRAM_FRAMES = 300
SPECTROGRAM_FLOOR_DB = -120
# Level reported for a band with no energy at all (digital silence). Far
# below the 24-bit noise floor (about -190 dB/Hz), and finite, unlike log(0).
DENSITY_FLOOR_DB = -300.0

_BLOCK = 64 * HOP

# True peak: x4 polyphase oversampling (ITU-R BS.1770-4 Annex 2). scipy's
# default resample_poly filter reaches 10 input samples either side, so 64
# samples of context at each block edge leave every evaluated value exactly
# what a single whole-file pass would give.
OVERSAMPLE = 4
_TP_CONTEXT = 64


def _centres(step: int, lowest: float, highest: float) -> np.ndarray:
    """Band centres 1000 * 2^(k/step) Hz with lowest <= f <= highest."""
    k_lo = math.ceil(step * math.log2(lowest / 1000))
    k_hi = math.floor(step * math.log2(highest / 1000))
    return 1000.0 * 2.0 ** (np.arange(k_lo, k_hi + 1) / step)


# 1/6-octave, 20 Hz .. 20 kHz: 22.1 Hz .. 17.96 kHz (the next centres up and
# down, 19.7 Hz and 20.16 kHz, fall just outside).
SPECTRUM_BANDS_HZ = _centres(6, 20.0, 20_000.0)
# 1/3-octave on the ISO 266 nominal "25 Hz .. 20 kHz" series: exact centres
# 24.8 Hz .. 20.16 kHz, hence the small tolerance on both ends.
SPECTROGRAM_BANDS_HZ = _centres(3, 24.0, 20_200.0)


def _band_matrix(centres: np.ndarray, step: int, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """(bins x bands) averaging matrix and a mask of bands that hold any bin."""
    freqs = np.fft.rfftfreq(FRAME, 1 / rate)
    half = 2.0 ** (1 / (2 * step))
    lo, hi = centres / half, centres * half
    member = (freqs[:, None] >= lo) & (freqs[:, None] < hi)
    counts = member.sum(axis=0)
    matrix = member / np.maximum(counts, 1)
    return matrix, counts > 0


def _fill(levels_db: np.ndarray, present: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Interpolate bands without bins from their neighbours in log-frequency;
    beyond the outermost measured band the nearest value is held."""
    if present.all() or not present.any():
        return levels_db
    x = np.log2(centres)
    return np.interp(x, x[present], levels_db[present])


def _density_db(power: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.maximum(10 * np.log10(power), DENSITY_FLOOR_DB)


class _Scan:
    """Accumulators for one streaming pass over the file."""

    def __init__(self, rate: int, channels: int, frames: int):
        self.rate, self.channels, self.frames = rate, channels, frames
        self.stereo = channels >= 2
        self.window = get_window("hann", FRAME)  # periodic, as scipy.signal.welch uses
        # Density scaling: |X|^2 / (fs * sum(w^2)), doubled for the one-sided
        # bins that stand for both a positive and a negative frequency.
        self.scale = np.full(FRAME // 2 + 1, 2.0 / (rate * np.sum(self.window**2)))
        self.scale[0] /= 2
        self.scale[-1] /= 2

        self.spec_matrix, self.spec_present = _band_matrix(SPECTRUM_BANDS_HZ, 6, rate)
        self.gram_matrix, self.gram_present = _band_matrix(SPECTROGRAM_BANDS_HZ, 3, rate)

        nb = len(SPECTRUM_BANDS_HZ)
        # [loud frames, all frames] sums of band densities, and frame counts.
        self.mid_sum = np.zeros((2, nb))
        self.side_sum = np.zeros((2, nb))
        self.count = np.zeros(2, dtype=np.int64)

        self.stft_frames = 1 if frames < FRAME else 1 + (frames - FRAME) // HOP
        self.gram_frames = min(MAX_SPECTROGRAM_FRAMES, self.stft_frames)
        self.gram_sum = np.zeros((self.gram_frames, len(SPECTROGRAM_BANDS_HZ)))
        self.gram_count = np.zeros(self.gram_frames, dtype=np.int64)
        self.done = 0  # STFT frames processed

        self.points = min(MAX_POINTS, frames)
        self.peak = np.zeros(self.points)
        self.square_sum = np.zeros(self.points)
        self.point_samples = np.zeros(self.points, dtype=np.int64)
        self.true_peak = np.zeros(self.points)
        self._tp_buffer = np.zeros((0, channels))
        self._tp_start = 0  # file position of _tp_buffer[0]
        self._tp_done = 0  # file position up to which true peak is final

        # sum L*L, R*R, L*R, M*M, S*S
        self.stereo_sums = np.zeros(5)

    def add_samples(self, block: np.ndarray, offset: int) -> None:
        """Waveform and stereo statistics, which see each sample once."""
        point = (np.arange(offset, offset + len(block)) * self.points) // self.frames
        starts = np.flatnonzero(np.diff(point)) + 1
        starts = np.concatenate(([0], starts))
        ids = point[starts]
        self.peak[ids] = np.maximum(
            self.peak[ids], np.maximum.reduceat(np.abs(block).max(axis=1), starts)
        )
        self.square_sum[ids] += np.add.reduceat((block**2).sum(axis=1), starts)
        self.point_samples[ids] += np.diff(np.append(starts, len(block))) * self.channels

        if self.stereo:
            left, right = block[:, 0], block[:, 1]
            mid, side = (left + right) / 2, (left - right) / 2
            self.stereo_sums += [
                left @ left,
                right @ right,
                left @ right,
                mid @ mid,
                side @ side,
            ]

    def add_true_peak(self, block: np.ndarray | None) -> None:
        """Feed the next block (``None`` at end of file). Each block is
        oversampled with the previous block's last samples in front of it, and
        its own last samples wait for the next block, so no peak is created or
        hidden at a block edge. The file's own ends are treated as silence on
        the far side, as any true-peak meter does."""
        if block is not None:
            self._tp_buffer = np.concatenate((self._tp_buffer, block))
        end = self._tp_start + len(self._tp_buffer)
        stop = end if block is None else end - _TP_CONTEXT
        if stop <= self._tp_done:
            return
        upsampled = resample_poly(self._tp_buffer, OVERSAMPLE, 1, axis=0)
        first = (self._tp_done - self._tp_start) * OVERSAMPLE
        last = (stop - self._tp_start) * OVERSAMPLE
        magnitude = np.abs(upsampled[first:last]).max(axis=1)
        # Upsampled value j sits at file position (_tp_done + j / 4).
        position = np.arange(self._tp_done * OVERSAMPLE, stop * OVERSAMPLE)
        point = (position * self.points) // (self.frames * OVERSAMPLE)
        starts = np.concatenate(([0], np.flatnonzero(np.diff(point)) + 1))
        ids = point[starts]
        reduced = np.maximum.reduceat(magnitude, starts)
        self.true_peak[ids] = np.maximum(self.true_peak[ids], reduced)

        self._tp_done = stop
        keep = max(stop - _TP_CONTEXT - self._tp_start, 0)
        self._tp_buffer = self._tp_buffer[keep:]
        self._tp_start += keep

    def add_frames(self, mid: np.ndarray, side: np.ndarray | None) -> None:
        """Spectral statistics for a batch of full FRAME-length frames (rows)."""
        loud = np.sqrt(np.mean(mid**2, axis=1)) >= 10 ** (SILENCE_DBFS / 20)
        mid_bands = self._density(mid) @ self.spec_matrix
        self.mid_sum[0] += mid_bands[loud].sum(axis=0)
        self.mid_sum[1] += mid_bands.sum(axis=0)
        if side is not None:
            side_bands = self._density(side) @ self.spec_matrix
            self.side_sum[0] += side_bands[loud].sum(axis=0)
            self.side_sum[1] += side_bands.sum(axis=0)
        self.count += [int(loud.sum()), len(mid)]

        gram_bands = self._density(mid) @ self.gram_matrix
        index = np.arange(self.done, self.done + len(mid))
        group = (index * self.gram_frames) // self.stft_frames
        np.add.at(self.gram_sum, group, gram_bands)
        np.add.at(self.gram_count, group, 1)
        self.done += len(mid)

    def _density(self, frames: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(frames * self.window, axis=1)) ** 2 * self.scale

    # Results -----------------------------------------------------------------

    def spectrum_db(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Long-term mid (and side) band densities in dB, unrounded."""
        row = 0 if self.count[0] > 0 else 1
        n = max(int(self.count[row]), 1)
        mid = _fill(_density_db(self.mid_sum[row] / n), self.spec_present, SPECTRUM_BANDS_HZ)
        if not self.stereo:
            return mid, None
        side = _fill(_density_db(self.side_sum[row] / n), self.spec_present, SPECTRUM_BANDS_HZ)
        return mid, side

    def spectrogram_db(self) -> list[list[int]]:
        widths = SPECTROGRAM_BANDS_HZ * (2 ** (1 / 6) - 2 ** (-1 / 6))
        rows = []
        for sums, n in zip(self.gram_sum, self.gram_count, strict=True):
            density = _fill(
                _density_db(sums / max(int(n), 1)), self.gram_present, SPECTROGRAM_BANDS_HZ
            )
            # Band power = mean density x bandwidth: a full-scale sine reads -3 dB.
            power = np.maximum(density + 10 * np.log10(widths), SPECTROGRAM_FLOOR_DB)
            rows.append([int(round(v)) for v in power])
        return rows

    def stereo_image(self) -> dict:
        if not self.stereo:
            return {"correlation": None, "side_to_mid_db": None}
        ll, rr, lr, mm, ss = self.stereo_sums
        correlation = lr / math.sqrt(ll * rr) if ll > 0 and rr > 0 else None
        ratio = 10 * math.log10(ss / mm) if ss > 0 and mm > 0 else None
        return {
            "correlation": None if correlation is None else round(float(correlation), 4) + 0.0,
            "side_to_mid_db": None if ratio is None else round(float(ratio), 2) + 0.0,
        }


def _mid_side(data: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if data.shape[1] == 1:
        return data[:, 0], None
    return (data[:, 0] + data[:, 1]) / 2, (data[:, 0] - data[:, 1]) / 2


def scan(path: str, timeline: bool = True) -> tuple[_Scan, float | None]:
    """One streaming pass: every accumulator, plus integrated loudness.
    ``timeline=False`` skips the true-peak oversampling, the costliest
    accumulator, for callers that only want the spectrum."""
    info = sf.info(path)
    if info.frames <= 0:
        raise ValueError("the file holds no audio frames")
    acc = _Scan(info.samplerate, info.channels, info.frames)
    meter = ebur.R128State(info.channels, info.samplerate, ebur.MeasurementMode.MODE_I)

    pending = np.zeros((0, info.channels))
    offset = 0
    with sf.SoundFile(path) as f:
        for block in f.blocks(blocksize=_BLOCK, dtype="float64", always_2d=True):
            meter.add_frames(np.ascontiguousarray(block).ravel(), len(block))
            acc.add_samples(block, offset)
            if timeline:
                acc.add_true_peak(block)
            offset += len(block)

            pending = np.concatenate((pending, block))
            usable = 0 if len(pending) < FRAME else 1 + (len(pending) - FRAME) // HOP
            usable = min(usable, acc.stft_frames - acc.done)
            if usable:
                mid, side = _mid_side(pending)
                view = np.lib.stride_tricks.sliding_window_view
                acc.add_frames(
                    view(mid, FRAME)[: usable * HOP : HOP],
                    None if side is None else view(side, FRAME)[: usable * HOP : HOP],
                )
                pending = pending[usable * HOP :]

    if acc.done == 0:  # shorter than one frame: one zero-padded frame
        padded = np.zeros((FRAME, info.channels))
        padded[: len(pending)] = pending
        mid, side = _mid_side(padded)
        acc.add_frames(mid[None, :], None if side is None else side[None, :])

    if timeline:
        acc.add_true_peak(None)

    lufs = ebur.get_loudness_global(meter)
    return acc, lufs if math.isfinite(lufs) else None


def relative(levels_db: np.ndarray | None, lufs: float | None) -> np.ndarray | None:
    if levels_db is None or lufs is None:
        return None
    return levels_db - lufs


def _rounded(values: np.ndarray | None, digits: int) -> list[float] | None:
    # + 0.0 turns a rounded -0.0 into 0.0: same number, less surprising JSON.
    return None if values is None else [round(float(v), digits) + 0.0 for v in values]


def analyze(path: str) -> dict:
    acc, lufs = scan(path)
    mid, side = acc.spectrum_db()
    duration = acc.frames / acc.rate
    rms = np.sqrt(acc.square_sum / np.maximum(acc.point_samples, 1))
    # The oversampling filter is not exactly interpolating, so at a point
    # whose peak sits on a sample it can read a hair under that sample. A true
    # peak is never below the sample peak, by definition.
    true_peak = np.maximum(acc.true_peak, acc.peak)

    return {
        "analysis_version": ANALYSIS_VERSION,
        "format": {
            "sample_rate": acc.rate,
            "channels": acc.channels,
            "duration_s": round(duration, 6),
        },
        "integrated_lufs": None if lufs is None else round(lufs, 3),
        "waveform": {
            "points": acc.points,
            "seconds_per_point": round(duration / acc.points, 6),
            "peak": _rounded(acc.peak, 4),
            "rms": _rounded(rms, 4),
            "true_peak": _rounded(true_peak, 4),
        },
        "spectrum": {
            "bands_hz": _rounded(SPECTRUM_BANDS_HZ, 2),
            "mid_db": _rounded(mid, 2),
            "side_db": _rounded(side, 2),
            "relative_mid_db": _rounded(relative(mid, lufs), 2),
            "relative_side_db": _rounded(relative(side, lufs), 2),
        },
        "spectrogram": {
            "frames": acc.gram_frames,
            "seconds_per_frame": round(duration / acc.gram_frames, 6),
            "bands_hz": _rounded(SPECTROGRAM_BANDS_HZ, 2),
            "db": acc.spectrogram_db(),
        },
        "stereo": acc.stereo_image(),
    }
