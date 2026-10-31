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
3. **Analysis** — a waveform overview with true peak, a long-term 1/6-octave
   spectrum, a spectrogram, the stereo image and where the audio sits at full
   scale, as numbers ready to draw or compare.
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
  "scipy_version": "1.18.1", "analysis_version": 2 }
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
synthesised stereo noise (each channel low-passed, so it is darker than white)
whose 2.0–2.2 s stretch was driven eight times over full scale and saturated
by the 24-bit format, with every array cut to its first three entries:

```json
{
  "analysis_version": 2,
  "format": { "sample_rate": 48000, "channels": 2, "duration_s": 5.0 },
  "integrated_lufs": -9.646,
  "waveform": { "points": 4000, "seconds_per_point": 0.00125,
                "peak": [0.1996, 0.2861, 0.2555], "rms": [0.0778, 0.0953, 0.0948],
                "true_peak": [0.2063, 0.2864, 0.2617] },
  "spectrum": { "bands_hz": [22.1, 24.8, 27.84],
                "mid_db": [-57.01, -57.01, -55.8], "side_db": [-57.66, -57.66, -56.92],
                "relative_mid_db": [-47.36, -47.36, -46.15],
                "relative_side_db": [-48.01, -48.01, -47.27] },
  "spectrogram": { "frames": 57, "seconds_per_frame": 0.087719,
                   "bands_hz": [22.1, 24.8, 27.84],
                   "db": [[-80, -80, -52], [-56, -55, -57], [-67, -66, -66]] },
  "stereo": { "correlation": 0.0067, "side_to_mid_db": -0.06 },
  "clipping": { "full_scale": 0.9999, "min_run": 3,
                "events": [ { "start_s": 2.0, "end_s": 2.1996, "channels": [0, 1],
                              "samples": 1419, "peak_dbfs": 0.0 } ],
                "events_total": 1, "clipped_samples": 1419 }
}
```

Inside the saturated stretch (waveform points 1600–1602, from 2.0 s) the
sample peak is pinned at `[1.0, 1.0, 1.0]` while `true_peak` reads
`[1.4365, 1.4048, 1.2972]`: the reconstructed waveform overshoots full scale
by up to 3.15 dB between the samples.

| Field | What it is |
|---|---|
| `integrated_lufs` | libebur128 integrated loudness; `null` for silence or under 400 ms |
| `waveform` | `min(4000, samples)` points; per point the largest \|sample\| (`peak`), the RMS over all channels (`rms`) and the true peak (`true_peak`): the largest \|value\| over all channels after ×4 polyphase oversampling (`scipy.signal.resample_poly`), as ITU-R BS.1770 defines true peak. `true_peak` is never below `peak` and is **not** clipped at 1.0 |
| `spectrum.bands_hz` | 1/6-octave centres `1000·2^(k/6)` within 20 Hz–20 kHz (22.1 Hz … 17.96 kHz, 59 bands), the same list for every file |
| `spectrum.mid_db` / `side_db` | `10·log10` of the mean power spectral density in each band, of mid `(L+R)/2` and side `(L−R)/2`; `side_db` is `null` for mono |
| `spectrum.relative_*_db` | the band levels minus `integrated_lufs` — tonal balance independent of level; `null` when the loudness is |
| `spectrogram` | `min(1200, frames)` time slices × the same 59 bands as `spectrum.bands_hz`, of mid band power in dB, integers floored at −120 (a full-scale sine reads −3); each slice averages the 8192-sample frames that fall in it |
| `stereo` | L/R correlation and side-to-mid energy ratio in dB, both `null` for mono; `correlation` is also `null` if a channel is silent, `side_to_mid_db` if there is no side at all (dual mono) |
| `clipping` | where the file sits at full scale — see below |

`clipping` follows one physical definition and no judgement:

* A sample is at full scale when `|sample| >= full_scale` (0.9999). Integer PCM
  is read as float, so a saturated 16-bit sample (32767 → 0.99997) and a 24-bit
  one count, as does anything over 1.0 in a float file.
* A clip is a run of at least `min_run` (3) consecutive full-scale samples on
  one channel. One or two are an ordinary peak.
* Runs on any channels less than 50 ms apart form one event: `start_s` is the
  first full-scale sample, `end_s` the moment after the last (both 4 dp),
  `channels` the channels involved (0 = left), `samples` the full-scale samples
  in its runs summed over channels, `peak_dbfs` the largest \|sample\| in them.
* `events` lists at most 500 events — the 500 with the most samples, in time
  order. `events_total` is the uncapped count and `clipped_samples` the samples
  in every counted run, all channels.

The file is streamed in blocks: a 10-minute 96 kHz stereo file takes about
16 s on a laptop (most of it the ×4 oversampling) and peaks near 250 MB.

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

### `POST /compress`

A stereo-linked bus compressor for glue. Multipart `file`, plus form fields
`threshold_db` (−40..0), `ratio` (1..10), `attack_ms` (0.1..100),
`release_ms` (10..1000) and `mix` (0..1, default 1: fully compressed; 0
returns the input). The response body is the compressed file as 32-bit float
WAV, the same length and rate, sample-aligned with the input.

Pedalboard's compressor supplies the gain computer and ballistics, run once
over the louder channel at each sample; the gain it applies is applied to
every channel alike, so the stereo image does not move. There is no make-up
gain: loudness is the caller's to set afterwards. How hard it worked comes
back in `X-Compression`, measured from the gain actually applied:

```json
{"max_reduction_db": 10.49, "mean_reduction_db": 5.21, "reduced_share": 0.5}
```

`reduced_share` is the share of the audible signal turned down by more than
0.5 dB.

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
