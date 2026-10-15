# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# SMS Spam Detector - application image
#
# Multi-stage build:
#   * `builder` installs the Python dependencies into a virtual environment.
#   * the final stage copies only that environment plus the source code, keeps
#     the image free of compilers and runs as an unprivileged user.
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# `libgomp1` is required by scikit-learn at runtime (OpenMP); we install it in
# the builder stage too so that wheels that link against it can be checked.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    DJANGO_SETTINGS_MODULE=config.settings \
    MODEL_ARTIFACTS_DIR=/app/artifacts \
    DATA_DIR=/app/data/raw \
    DJANGO_DB_PATH=/app/data/db.sqlite3 \
    PORT=8000

# Runtime shared library for scikit-learn, plus curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Source code. The dataset and the compiled front-end assets are part of the
# repository; generated artefacts (artifacts/, staticfiles/) are mounted volumes
# or created at runtime - see .dockerignore and docker-compose.yml.
COPY . .

# Dedicated unprivileged user owning the writable directories.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/artifacts/models /app/staticfiles /app/data/raw \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/" > /dev/null || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
