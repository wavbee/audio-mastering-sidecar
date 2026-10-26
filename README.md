# audio-mastering-sidecar

A small, stateless HTTP service that does four things with audio files:

1. **Reference-matched mastering** — makes a track sound like a reference track
   you choose (frequency balance, loudness and peak behaviour), using
   [Matchering](https://github.com/sergree/matchering).
2. **Independent loudness measurement** — ITU-R BS.1770 / EBU R128 readings from
   two implementations that share no code:
   [libebur128](https://github.com/jiixyj/libebur128) (via
   [pyebur128](https://github.com/jodhus/pyebur128)) and
   [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm).
3. **Analysis** — a waveform overview, a long-term 1/6-octave spectrum, a
   coarse spectrogram and the stereo image, as numbers ready to draw or compare.
4. **EQ matching** — a linear-phase EQ that moves a track toward a tonal-balance
   curve the caller supplies (for example one taken from a reference's analysis).

Built by **wavbee** as one component of a release quality-control and
mastering pipeline, and open source so anyone can run the same processing.
Contributions are welcome.

---

## Design principle: audio and raw numbers only

This service contains **no policy**. There are no loudness targets, no
true-peak ceilings, no thresholds and no pass/fail logic. `/match` and
`/eq_match` return audio; `/measure` and `/analyze` return readings. Whether
−9 LUFS is "too loud", whether two meters "agree", or which curve a track should
be EQ'd toward is the caller's decision.

It also **never fetches URLs**. A sidecar that downloads whatever it is told to,
from inside a private network, is a server-side request forgery proxy. Callers
upload the bytes.

## Why GPL-3.0

Matchering is licensed under the **GNU GPL v3**. This service imports it, so
it is distributed under **GPL-3.0-or-later** — see [`LICENSE`](LICENSE).
Programs that call this service **over HTTP** are separate works communicating
across a process boundary. (Engineering rationale, not legal advice.)

---

## API

### `GET /healthz`

```json
{ "status": "ok", "service_version": "0.1.0", "matchering_version": "2.0.6",
  "pyloudnorm_version": "0.2.0", "libebur128_version": "1.2.6",
  "scipy_version": "1.18.1", "analysis_version": 1 }
```

`analysis_version` changes whenever `/analyze` or `/eq_match` would compute a
different number for the same file, so a caller can tell stored analyses apart.

### `POST /measure`

Multipart upload with a `file` field (WAV, FLAC, anything libsndfile reads).
Real output for EBU Tech 3341 test 1 (1 kHz stereo sine, −23 dBFS peak, 20 s),
which a compliant meter must read as −23.0 ±0.1 LUFS:

```json
{
  "format": { "sample_rate": 48000, "channels": 2, "subtype": "PCM_24", "duration_s": 20.0 },
  "libebur128": {
    "version": "1.2.6",
    "integrated_lufs": -22.993,
    "lra_lu": 0.0,
    "true_peak_dbtp": [-23.0, -23.0],
    "sample_peak_dbfs": [-23.0, -23.0]
  },
  "pyloudnorm": { "integrated_lufs": -23.035 }
}
```

A reading that cannot exist (the loudness of digital silence) is `null`, never
`-Infinity` — JSON has no infinity, and `null` is unambiguous.

### `POST /match`

Multipart upload with `target` (the track to master) and `reference` (the track
it should sound like), plus an optional form field `bit_depth` = `16` or `24`
(default `24`). The response body is the mastered WAV, at the **target's own
sample rate**, with headers:

| Header | Meaning |
|---|---|
| `X-Matchering-Version` | Matchering build that produced the audio |
| `X-Output-Sample-Rate` | equals the target's sample rate |
| `X-Output-Bit-Depth` | `16` or `24` — nothing wider is ever written |

Matchering refuses some inputs (target identical to reference, too short, too
long — 15 minutes by default). Those come back as `422 match_refused` with
Matchering's own explanation.

### `POST /analyze`

Multipart upload with a `file` field. Real output for five seconds of
synthesised stereo noise (each channel low-passed, so it is darker than white),
with every array cut to its first three entries:

```json
{
  "analysis_version": 1,
  "format": { "sample_rate": 48000, "channels": 2, "duration_s": 5.0 },
  "integrated_lufs": -14.732,
  "waveform": { "points": 1000, "seconds_per_point": 0.005,
                "peak": [0.2504, 0.3725, 0.3713], "rms": [0.0866, 0.1073, 0.1084] },
  "spectrum": { "bands_hz": [22.1, 24.8, 27.84],
                "mid_db": [-59.44, -59.44, -58.53], "side_db": [-58.24, -58.24, -59.03],
                "relative_mid_db": [-44.7, -44.7, -43.8],
                "relative_side_db": [-43.51, -43.51, -44.3] },
  "spectrogram": { "frames": 57, "seconds_per_frame": 0.087719,
                   "bands_hz": [24.8, 31.25, 39.37],
                   "db": [[-74, -46, -46], [-50, -51, -49], [-61, -60, -50]] },
  "stereo": { "correlation": 0.0, "side_to_mid_db": 0.0 }
}
```

| Field | What it is |
|---|---|
| `integrated_lufs` | libebur128 integrated loudness; `null` for silence or under 400 ms |
| `waveform` | `min(1000, samples)` points; per point the largest \|sample\| and the RMS over all channels |
| `spectrum.bands_hz` | 1/6-octave centres `1000·2^(k/6)` within 20 Hz–20 kHz (22.1 Hz … 17.96 kHz, 59 bands), the same list for every file |
| `spectrum.mid_db` / `side_db` | `10·log10` of the mean power spectral density in each band, of mid `(L+R)/2` and side `(L−R)/2`; `side_db` is `null` for mono |
| `spectrum.relative_*_db` | the band levels minus `integrated_lufs` — tonal balance independent of level; `null` when the loudness is |
| `spectrogram` | `min(300, frames)` time slices × 1/3-octave bands (ISO 25 Hz … 20 kHz, 30 bands) of mid band power in dB, integers floored at −120 (a full-scale sine reads −3) |
| `stereo` | L/R correlation and side-to-mid energy ratio in dB, both `null` for mono; `correlation` is also `null` if a channel is silent, `side_to_mid_db` if there is no side at all (dual mono) |

How the spectrum is measured, because two files are only comparable if they
were measured the same way:

* Hann-windowed 8192-sample frames at 50 % overlap, averaged as **power
  spectral density** (scipy's `density` scaling). The same audio at 44.1 kHz
  and 96 kHz therefore reads the same band levels.
* Frames whose mid RMS is below −60 dBFS are **left out**, so leading, trailing
  or mid-song digital silence does not drag the curve down. A file that is that
  quiet throughout uses every frame.
* A band too narrow to hold an FFT bin at the file's rate (the lowest bands,
  especially at 96 kHz) is interpolated from its neighbours in log-frequency.
  A band with no energy at all reads −300 dB rather than `-Infinity`.

### `POST /eq_match`

Multipart upload with `file`, plus form fields:

| Field | Default | Meaning |
|---|---|---|
| `target_curve` | required | JSON `{"bands_hz": [...], "mid_db": [...], "side_db": [...] or null}` in the `relative_*_db` convention, on the `/analyze` 1/6-octave grid (within 1 %) |
| `strength` | `1.0` | 0–1, scales every gain |
| `max_gain_db` | `6.0` | 0–12, the largest boost or cut in any band |
| `bit_depth` | `24` | `16` or `24` |

The file's own relative curve is measured exactly as `/analyze` measures it.
Per band, the gain is `target − own`, smoothed over three neighbouring bands,
limited to ±`max_gain_db`, never positive below 30 Hz or above 16 kHz, then
multiplied by `strength`. A linear-phase FIR (`scipy.signal.firwin2`, gains
interpolated in log-frequency) is applied to mid and to side, each with its own
curve when both the target and the file have a side; otherwise side gets the mid
curve. The filter's delay is removed, so the output is the same length as the
input and sample-aligned with it.

If the result would exceed 0.99 sample peak, the whole file is scaled down to
0.99 — never limited. The response body is the WAV at the input's sample rate,
with headers:

| Header | Meaning |
|---|---|
| `X-Applied-Curve` | compact JSON `{"bands_hz", "mid_db", "side_db"}` of the gains actually applied, in dB (2 dp); `side_db` is `null` for mono |
| `X-Output-Gain-Db` | the safety scale applied afterwards, in dB (negative); `0.0` if none |
| `X-Analysis-Version` | the `analysis_version` the curve was measured with |

Real result for the noise above, matched at `strength=0.8` to the curve of a
brighter noise (first and last three bands of 59):

```text
X-Applied-Curve: {"bands_hz":[22.1,24.8,27.84,…,14254.38,16000.0,17959.39],
                  "mid_db":[-4.8,-4.8,-4.8,…,3.23,3.46,0.0],
                  "side_db":[-4.8,-4.8,-4.8,…,3.24,3.46,0.0]}
X-Output-Gain-Db: 0.0
```

The lows are cut by the full 6 dB × 0.8, the top is lifted, and the 17.96 kHz
band, which also wanted a lift, stays at 0.0 because nothing is boosted above
16 kHz.

### Errors

```json
{ "error": { "code": "match_refused", "message": "..." } }
```

| Status | Code |
|---|---|
| 413 | `payload_too_large` — over `MAX_UPLOAD_BYTES` |
| 422 | `unreadable_audio` (including a silent file sent to `/eq_match`, which has no relative curve), `match_refused`, `invalid_curve`, `invalid_parameter` (`strength`, `max_gain_db` or `bit_depth` out of range) |
| 504 | `timeout` — over `PROCESS_TIMEOUT_S` |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_BYTES` | 629145600 (600 MB) | per uploaded file |
| `PROCESS_TIMEOUT_S` | 300 | wall-clock budget per request |

## Run it

```bash
docker build -t audio-mastering-sidecar .
docker run -p 8000:8000 audio-mastering-sidecar
curl -F target=@mix.wav -F reference=@favourite-master.wav http://localhost:8000/match -o mastered.wav
curl -F file=@mix.wav http://localhost:8000/analyze
curl -F file=@mix.wav -F 'target_curve=<curve.json' -F strength=0.5 \
     http://localhost:8000/eq_match -o eq.wav
```

Development:

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest -q
```

The test suite synthesises its own audio. Loudness is checked against the
EBU Tech 3341 conformance definitions, under which a 1 kHz stereo tone at
−23 dBFS peak must read −23.0 ±0.1 LUFS.
