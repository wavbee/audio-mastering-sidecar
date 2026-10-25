# audio-mastering-sidecar

A small, stateless HTTP service that does two things to audio files:

1. **Reference-matched mastering** — makes a track sound like a reference track
   you choose (frequency balance, loudness and peak behaviour), using
   [Matchering](https://github.com/sergree/matchering).
2. **Independent loudness measurement** — ITU-R BS.1770 / EBU R128 readings from
   two implementations that share no code:
   [libebur128](https://github.com/jiixyj/libebur128) (via
   [pyebur128](https://github.com/jodhus/pyebur128)) and
   [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm).

Built by **wavbee** as one component of a release quality-control and
mastering pipeline, and open source so anyone can run the same processing.
Contributions are welcome.

---

## Design principle: audio and raw numbers only

This service contains **no policy**. There are no loudness targets, no
true-peak ceilings, no thresholds and no pass/fail logic. `/match` returns the
matched audio; `/measure` returns the meters' readings. Whether −9 LUFS is "too
loud" or whether two meters "agree" is the caller's decision.

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
  "pyloudnorm_version": "0.2.0", "libebur128_version": "1.2.6" }
```

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

### Errors

```json
{ "error": { "code": "match_refused", "message": "..." } }
```

| Status | Code |
|---|---|
| 413 | `payload_too_large` — over `MAX_UPLOAD_BYTES` |
| 422 | `unreadable_audio`, `match_refused` |
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
