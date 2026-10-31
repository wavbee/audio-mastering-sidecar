# Builder: pyebur128 ships no prebuilt Linux wheel for CPython 3.13, so it is
# compiled here. The compiler never reaches the runtime image.
FROM python:3.13-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY app/ ./app/
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir .

FROM python:3.13-slim

# libsndfile backs soundfile, which Matchering and both meters read through;
# Pedalboard's native module links libatomic, which the slim image lacks.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libatomic1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /venv /venv
ENV PATH="/venv/bin:$PATH"
# One numeric thread per process. A request is handled by one worker; letting
# BLAS/OpenMP size its pool to the host oversubscribes the CPU as soon as
# there is more than one worker (measured: 8 workers on 16 cores, load 60).
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY app/ ./app/

RUN useradd --create-home --uid 10001 appuser
USER appuser
# Fail the build, not the container, when a native library is missing: the
# app is imported here exactly as uvicorn will import it.
RUN python -c "import app.main"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz').read()" || exit 1

# One worker: Matchering is CPU- and memory-bound per request; scale with
# replicas, not workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
