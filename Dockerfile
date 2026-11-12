# Container image for the eurlex-rag HTTP service.
#
# Build:   docker build -t eurlex-rag .
# Run:     docker run -p 8000:8000 -v "$(pwd)/data/processed:/data:ro" eurlex-rag
# Check:   curl http://localhost:8000/healthz
#
# The image holds code only: no legal texts, no results, no secrets. The corpus
# (units.jsonl, built on the host with `eurlex ingest`) is mounted at /data when
# the container starts. Without it the service still starts, and /healthz
# answers 503 explaining what is missing.
#
# Choose what to serve at run time with -e EURLEX_RUNG=... and
# -e EURLEX_CHUNKING=article|paragraph. `docker compose up --build` does all of
# this for you; see compose.yaml.

# Start from the official Python 3.12 image, "slim" variant: Debian with only
# what Python needs, a few hundred MB smaller than the full image.
FROM python:3.12-slim

# Settings for Python and pip inside the container:
#   PYTHONDONTWRITEBYTECODE  do not write .pyc files (they are useless in an image)
#   PYTHONUNBUFFERED         print logs immediately, so `docker logs` shows them live
#   PIP_NO_CACHE_DIR         do not keep pip's download cache in the image layers
#   PIP_DISABLE_PIP_VERSION_CHECK  skip pip's "new version available" network call
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create an ordinary user to run the service. A process running as root inside a
# container that gets broken into has more power than it needs; this one cannot
# install software or change system files.
RUN useradd --create-home --uid 1000 eurlex

# All following commands run in /app (created if it does not exist).
WORKDIR /app

# Copy only what installing the package needs. pyproject.toml lists the
# dependencies; README.md is read by pyproject.toml as the package description;
# src/ is the code. Everything else (tests, docs, data) stays out of the image.
# Trade-off: because src/ is copied before the install, any code change re-runs
# the dependency install on the next build. For a project this size that takes
# seconds, and it keeps the dependency list in one place (pyproject.toml).
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package with the "serve" extra (FastAPI, uvicorn, pydantic). It is
# a normal install, not an editable one: the code is copied into site-packages.
RUN pip install ".[serve]"

# The mount point for the corpus. Created here and given to the eurlex user so
# the directory exists with the right owner even when nothing is mounted. We do
# not declare it with VOLUME: that would make Docker create an anonymous volume
# on every run, which then lingers on disk.
RUN mkdir -p /data && chown eurlex:eurlex /data

# From here on, and when the container runs, act as the unprivileged user.
USER eurlex

# Defaults the service reads at start-up (see eurlex_rag/api/app.py). Each can be
# overridden with `docker run -e NAME=value` or in compose.yaml.
ENV EURLEX_UNITS=/data/units.jsonl \
    EURLEX_RUNG=01_bm25 \
    EURLEX_CHUNKING=article

# Documents that the service listens on port 8000. It does not publish the port
# by itself; `-p 8000:8000` (or `ports:` in compose.yaml) does that.
EXPOSE 8000

# How Docker decides the container is healthy: every 30 s, request /healthz.
# slim images have no curl, so the check uses Python's standard library. urlopen
# raises an error on a non-2xx status (the 503 when the corpus is missing), which
# makes the command exit with 1, which Docker counts as a failed check. After 3
# failures in a row the container is marked "unhealthy". start-period gives the
# first start time to load and index the corpus before failures count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)" || exit 1

# The command the container runs. uvicorn is the web server; --factory tells it
# that create_app is a function to call (which builds the app at start-up),
# not an app object. --host 0.0.0.0 listens on all interfaces, so the port is
# reachable from outside the container (127.0.0.1 would only be reachable from
# inside it). One process: the index is in memory and each worker would build
# its own copy.
CMD ["uvicorn", "eurlex_rag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
