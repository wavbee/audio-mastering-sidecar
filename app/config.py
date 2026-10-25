"""Runtime limits, all overridable via environment variables.

Operational guardrails (sizes and timeouts) only. This service holds no
loudness targets, thresholds or pass/fail logic — the caller owns policy.
"""

from __future__ import annotations

import os


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# Per uploaded file (target and reference are each limited).
MAX_UPLOAD_BYTES: int = _int_env("MAX_UPLOAD_BYTES", 600 * 1024 * 1024)

# Wall-clock budget for one request's DSP.
PROCESS_TIMEOUT_S: int = _int_env("PROCESS_TIMEOUT_S", 300)
