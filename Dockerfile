FROM python:3.13-slim

# libsndfile backs soundfile, which Matchering and both meters read through.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pyproject is the single source of truth for dependencies.
COPY pyproject.toml README.md LICENSE ./
COPY app/ ./app/
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz').read()" || exit 1

# One worker: Matchering is CPU- and memory-bound per request; scale with
# replicas, not workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
