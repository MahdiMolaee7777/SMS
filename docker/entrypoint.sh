#!/bin/sh
# ---------------------------------------------------------------------------
# Container entry point.
#
#  1. waits for the mounted directories to exist,
#  2. applies database migrations (idempotent, fast),
#  3. collects static files when running with DEBUG disabled,
#  4. warns (does not fail) when the models have not been trained yet,
#  5. executes the container command (gunicorn by default).
#
# The models are deliberately NOT trained here: training takes a few seconds and
# doing it on every start would hide an important, explicit project step.
# ---------------------------------------------------------------------------
set -eu

echo "[entrypoint] SMS Spam Detector starting (DJANGO_DEBUG=${DJANGO_DEBUG:-0})"

mkdir -p "${MODEL_ARTIFACTS_DIR:-/app/artifacts}/models" /app/staticfiles /app/data

echo "[entrypoint] Applying database migrations ..."
python manage.py migrate --noinput

if [ "${DJANGO_DEBUG:-0}" = "0" ] || [ "${DJANGO_DEBUG:-0}" = "false" ]; then
  echo "[entrypoint] Collecting static files ..."
  python manage.py collectstatic --noinput > /dev/null
fi

if [ -f "${MODEL_ARTIFACTS_DIR:-/app/artifacts}/metrics.json" ]; then
  echo "[entrypoint] Trained models found in ${MODEL_ARTIFACTS_DIR:-/app/artifacts}."
else
  echo "[entrypoint] WARNING: no trained models found."
  echo "[entrypoint]          Run: docker compose run --rm web python manage.py train_models"
fi

exec "$@"
