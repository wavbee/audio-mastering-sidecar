"""FastAPI entrypoint.

Endpoints:
  * GET  /healthz  -> liveness + library versions.
  * POST /measure  -> independent BS.1770 readings for one uploaded file.
  * POST /match    -> the uploaded ``target`` mastered to the uploaded
                      ``reference`` (Matchering); the response body is WAV.
  * POST /analyze  -> waveform overview, long-term spectrum, spectrogram and
                      stereo image of one uploaded file.
  * POST /eq_match -> the uploaded ``file`` EQ'd toward a caller-supplied
                      tonal-balance curve; the response body is WAV.

Uploads only. There is deliberately no "fetch this URL" variant: a sidecar
that fetches arbitrary URLs from inside a private network is an SSRF proxy,
so callers send the bytes. Raw measurements and audio only — no targets,
thresholds or verdicts live here.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from importlib.metadata import version

import matchering
import pyebur128
import scipy
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from app import __version__ as SERVICE_VERSION
from app.analyze import ANALYSIS_VERSION, analyze
from app.config import MAX_UPLOAD_BYTES, PROCESS_TIMEOUT_S
from app.eq import CurveError, ParameterError, eq_match, parse_curve, parse_parameters
from app.master import MatchError, match
from app.measure import measure

app = FastAPI(title="audio-mastering-sidecar", version=SERVICE_VERSION)

_CHUNK = 1024 * 1024


class _TooLarge(Exception):
    pass


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


async def _save(upload: UploadFile, directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    size = 0
    with open(path, "wb") as out:
        while chunk := await upload.read(_CHUNK):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise _TooLarge(name)
            out.write(chunk)
    return path


def _unreadable(error: Exception) -> JSONResponse:
    return _error(422, "unreadable_audio", f"could not read audio: {error}")


def _temp_dir() -> tuple[str, Callable[[], None]]:
    """A directory for a response that is streamed back after the handler
    returns, and the cleanup to run once it has been sent (or on error)."""
    directory = tempfile.mkdtemp()

    def cleanup() -> None:
        with contextlib.suppress(OSError):
            for name in os.listdir(directory):
                os.remove(os.path.join(directory, name))
            os.rmdir(directory)

    return directory, cleanup


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "service_version": SERVICE_VERSION,
        "matchering_version": matchering.__version__,
        # pyloudnorm has no __version__; the installed distribution knows it.
        "pyloudnorm_version": version("pyloudnorm"),
        "libebur128_version": str(pyebur128.get_libebur128_version()),
        "scipy_version": scipy.__version__,
        "analysis_version": ANALYSIS_VERSION,
    }


@app.post("/measure")
async def measure_endpoint(file: UploadFile):
    with tempfile.TemporaryDirectory() as directory:
        try:
            path = await _save(file, directory, "input.wav")
            return await asyncio.wait_for(asyncio.to_thread(measure, path), PROCESS_TIMEOUT_S)
        except _TooLarge:
            return _error(413, "payload_too_large", f"file exceeds {MAX_UPLOAD_BYTES} bytes")
        except TimeoutError:
            return _error(504, "timeout", f"measurement exceeded {PROCESS_TIMEOUT_S}s")
        except (RuntimeError, ValueError) as error:
            return _unreadable(error)


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile):
    with tempfile.TemporaryDirectory() as directory:
        try:
            path = await _save(file, directory, "input.wav")
            return await asyncio.wait_for(asyncio.to_thread(analyze, path), PROCESS_TIMEOUT_S)
        except _TooLarge:
            return _error(413, "payload_too_large", f"file exceeds {MAX_UPLOAD_BYTES} bytes")
        except TimeoutError:
            return _error(504, "timeout", f"analysis exceeded {PROCESS_TIMEOUT_S}s")
        except (RuntimeError, ValueError) as error:
            return _unreadable(error)


@app.post("/match")
async def match_endpoint(target: UploadFile, reference: UploadFile, bit_depth: int = Form(24)):
    directory, cleanup = _temp_dir()
    try:
        target_path = await _save(target, directory, "target.wav")
        reference_path = await _save(reference, directory, "reference.wav")
        output = os.path.join(directory, "matched.wav")
        details = await asyncio.wait_for(
            asyncio.to_thread(match, target_path, reference_path, output, bit_depth),
            PROCESS_TIMEOUT_S,
        )
    except _TooLarge as error:
        cleanup()
        return _error(413, "payload_too_large", f"{error} exceeds {MAX_UPLOAD_BYTES} bytes")
    except TimeoutError:
        cleanup()
        return _error(504, "timeout", f"matching exceeded {PROCESS_TIMEOUT_S}s")
    except MatchError as error:
        cleanup()
        return _error(422, "match_refused", str(error))

    return FileResponse(
        output,
        media_type="audio/wav",
        headers={
            "X-Matchering-Version": matchering.__version__,
            "X-Output-Sample-Rate": str(details["sample_rate"]),
            "X-Output-Bit-Depth": str(details["bit_depth"]),
        },
        background=BackgroundTask(cleanup),
    )


@app.post("/eq_match")
async def eq_match_endpoint(
    file: UploadFile,
    target_curve: str | None = Form(None),
    strength: str = Form("1.0"),
    max_gain_db: str = Form("6.0"),
    bit_depth: str = Form("24"),
):
    try:
        curve = parse_curve(target_curve)
        strength_value, max_gain_value, depth = parse_parameters(strength, max_gain_db, bit_depth)
    except CurveError as error:
        return _error(422, "invalid_curve", str(error))
    except ParameterError as error:
        return _error(422, "invalid_parameter", str(error))

    directory, cleanup = _temp_dir()
    try:
        path = await _save(file, directory, "input.wav")
        output = os.path.join(directory, "eq.wav")
        details = await asyncio.wait_for(
            asyncio.to_thread(eq_match, path, output, curve, strength_value, max_gain_value, depth),
            PROCESS_TIMEOUT_S,
        )
    except _TooLarge:
        cleanup()
        return _error(413, "payload_too_large", f"file exceeds {MAX_UPLOAD_BYTES} bytes")
    except TimeoutError:
        cleanup()
        return _error(504, "timeout", f"eq matching exceeded {PROCESS_TIMEOUT_S}s")
    except (RuntimeError, ValueError) as error:  # SilentError is a ValueError
        cleanup()
        return _unreadable(error)

    return FileResponse(
        output,
        media_type="audio/wav",
        headers={
            "X-Applied-Curve": json.dumps(details["applied_curve"], separators=(",", ":")),
            "X-Output-Gain-Db": str(details["output_gain_db"]),
            "X-Analysis-Version": str(ANALYSIS_VERSION),
        },
        background=BackgroundTask(cleanup),
    )
