"""Where a recording sits at full scale: runs of consecutive full-scale
samples, located in time, so a caller can point at each one.

This is a physical definition, not a judgement. Whether a run is audible
distortion, a deliberate square wave or a harmless overshoot is the caller's
call; this module only says where the samples are and how many there are.

Choices a caller should know about:
  * Full scale is ``|sample| >= 0.9999``. Integer PCM is read as float, so a
    saturated 16-bit sample (32767) reads 0.99997 and a 24-bit one reads
    0.9999999 — both count, and so does anything over 1.0 in a float file.
  * One or two full-scale samples in a row are what any loud, unclipped
    master can hit on a peak. A clip is ``MIN_RUN`` (3) or more in a row on
    one channel.
  * Runs on any channel less than 50 ms apart are one event: a clipped
    passage is a burst of runs a few samples apart, and listing each would
    bury the timeline. An event carries the union of its channels, the sum of
    its runs' samples and the largest |sample| in them.
  * At most ``MAX_EVENTS`` events are listed — the largest by sample count,
    then put back in time order. ``events_total`` is the uncapped count.

The scan is streamed block by block. A run may start in one block and end in
another, and an event is only closed once no later run can still join it.
"""

from __future__ import annotations

import math

import numpy as np

FULL_SCALE = 0.9999
MIN_RUN = 3
MERGE_GAP_S = 0.05
MAX_EVENTS = 500


def _groups(start, end, channels, samples, peak):
    """Parallel arrays, one entry per group of merged runs."""
    return {"start": start, "end": end, "channels": channels, "samples": samples, "peak": peak}


class ClipScan:
    """Accumulates full-scale runs across a streamed file."""

    def __init__(self, rate: int, channels: int):
        self.rate, self.channels = rate, channels
        self.gap = MERGE_GAP_S * rate  # in samples: runs closer than this merge
        self.position = 0  # file position of the next block's first sample
        # A run still open at the end of the previous block, per channel.
        self.open_start = np.full(channels, -1, dtype=np.int64)
        self.open_peak = np.zeros(channels)
        self.clipped_samples = 0
        self.pending = self._empty()  # merged groups a later run may still join
        self.final: list[dict] = []  # closed events, in time order

    def _empty(self) -> dict:
        return _groups(
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, self.channels), dtype=bool),
            np.zeros(0, dtype=np.int64),
            np.zeros(0),
        )

    def add(self, block: np.ndarray) -> None:
        magnitude = np.abs(block)
        hot = magnitude >= FULL_SCALE
        n = len(block)
        runs = []  # (starts, ends, channel, peaks) of runs closed in this block
        for c in range(self.channels):
            edges = np.diff(hot[:, c].astype(np.int8), prepend=0, append=0)
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            peaks = (
                np.maximum.reduceat(magnitude[:, c], starts) if len(starts) else np.zeros(0)
            )  # max over [start, next start) is the run's max: the rest is below full scale
            starts = starts + self.position
            ends = ends + self.position

            if self.open_start[c] >= 0:
                if len(starts) and starts[0] == self.position:  # the open run continues
                    starts[0] = self.open_start[c]
                    peaks[0] = max(peaks[0], self.open_peak[c])
                else:  # it ended on the last sample of the previous block
                    starts = np.concatenate(([self.open_start[c]], starts))
                    ends = np.concatenate(([self.position], ends))
                    peaks = np.concatenate(([self.open_peak[c]], peaks))
                self.open_start[c] = -1

            if len(ends) and ends[-1] == self.position + n:  # still running at the block end
                self.open_start[c], self.open_peak[c] = starts[-1], peaks[-1]
                starts, ends, peaks = starts[:-1], ends[:-1], peaks[:-1]
            runs.append((starts, ends, c, peaks))

        self.position += n
        open_starts = self.open_start[self.open_start >= 0]
        horizon = min(self.position, int(open_starts.min())) if len(open_starts) else self.position
        self._merge(runs, horizon)

    def finish(self) -> None:
        """Close runs that reach the end of the file, and every event."""
        runs = []
        for c in np.flatnonzero(self.open_start >= 0):
            runs.append(
                (
                    self.open_start[c : c + 1].copy(),
                    np.array([self.position]),
                    int(c),
                    self.open_peak[c : c + 1].copy(),
                )
            )
            self.open_start[c] = -1
        self._merge(runs, math.inf)

    def _merge(self, runs: list, horizon: float) -> None:
        """Fold newly closed runs into the pending groups, then move every
        group no future run (all start at or after ``horizon``) can reach
        into the closed events."""
        start = [self.pending["start"]]
        end = [self.pending["end"]]
        channels = [self.pending["channels"]]
        peak = [self.pending["peak"]]
        samples = [self.pending["samples"]]
        for run_start, run_end, c, run_peak in runs:
            keep = (run_end - run_start) >= MIN_RUN
            if not keep.any():
                continue
            start.append(run_start[keep])
            end.append(run_end[keep])
            peak.append(run_peak[keep])
            length = run_end[keep] - run_start[keep]
            samples.append(length)
            self.clipped_samples += int(length.sum())
            mask = np.zeros((int(keep.sum()), self.channels), dtype=bool)
            mask[:, c] = True
            channels.append(mask)

        start = np.concatenate(start)
        if not len(start):
            return
        order = np.argsort(start, kind="stable")
        start = start[order]
        end = np.concatenate(end)[order]
        channels = np.concatenate(channels)[order]
        samples = np.concatenate(samples)[order]
        peak = np.concatenate(peak)[order]

        reach = np.maximum.accumulate(end)
        first = np.concatenate(([0], np.flatnonzero(start[1:] - reach[:-1] >= self.gap) + 1))
        groups = _groups(
            start[first],
            np.maximum.reduceat(end, first),
            np.logical_or.reduceat(channels, first, axis=0),
            np.add.reduceat(samples, first),
            np.maximum.reduceat(peak, first),
        )
        # Groups are disjoint and in time order, so the closed ones are a prefix.
        closed = int(np.count_nonzero(horizon - groups["end"] >= self.gap))
        if closed:
            self.final.append({key: value[:closed] for key, value in groups.items()})
        self.pending = {key: value[closed:] for key, value in groups.items()}

    def result(self) -> dict:
        events = self.final or [self._empty()]
        merged = {key: np.concatenate([e[key] for e in events]) for key in events[0]}
        total = len(merged["start"])
        index = np.arange(total)
        if total > MAX_EVENTS:  # the largest, ties to the earlier; then back in time order
            index = np.sort(np.argsort(-merged["samples"], kind="stable")[:MAX_EVENTS])
        return {
            "full_scale": FULL_SCALE,
            "min_run": MIN_RUN,
            "events": [
                {
                    "start_s": round(int(merged["start"][i]) / self.rate, 4) + 0.0,
                    "end_s": round(int(merged["end"][i]) / self.rate, 4) + 0.0,
                    "channels": np.flatnonzero(merged["channels"][i]).tolist(),
                    "samples": int(merged["samples"][i]),
                    "peak_dbfs": round(20 * math.log10(float(merged["peak"][i])), 2) + 0.0,
                }
                for i in index
            ],
            "events_total": total,
            "clipped_samples": self.clipped_samples,
        }
