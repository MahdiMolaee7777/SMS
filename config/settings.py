"""Django settings for the SMS Spam Detector.

The application is configured through environment variables (see ``.env.example``)
so that the same source tree runs unchanged on a laptop, in Docker and in CI.

Security defaults
-----------------
* ``DEBUG`` is only true when ``DJANGO_DEBUG`` is explicitly enabled.
* ``SECRET_KEY`` must be supplied in production; a development-only key is used
  as a fallback so that ``python manage.py runserver`` works out of the box.
* Production security switches (HTTPS redirect, secure cookies, HSTS) are opt-in
  through environment variables and are documented in the README.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load a local .env file if present (never committed - see .gitignore).
load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable ("1", "true", "yes", "on" are true)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: list[str] | None = None) -> list[str]:
    """Read a comma-separated environment variable into a list."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------

DEBUG = env_bool("DJANGO_DEBUG", True)

#: Development fallback key. Deployment must set DJANGO_SECRET_KEY.
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-dev-only-key-change-me-before-any-deployment",
)

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", ["*"] if DEBUG else [])
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS", [])

# X_FRAME_OPTIONS: empty in development so the app can be embedded in local
# preview panes and iframes; "DENY" is the safe production value.
X_FRAME_OPTIONS = os.environ.get(
    "DJANGO_X_FRAME_OPTIONS", "" if DEBUG else "DENY"
)

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --------------------------------------------------------------------------
# Applications
# --------------------------------------------------------------------------

INSTALLED_APPS = [
    "whitenoise.runserver_nostatic",  # serve collected static files in runserver
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "detector",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "detector.context_processors.project_metadata",
            ],
        },
    },
]

# --------------------------------------------------------------------------
# Database (SQLite is sufficient for this project)
# --------------------------------------------------------------------------

DB_PATH = Path(os.environ.get("DJANGO_DB_PATH", BASE_DIR / "data" / "db.sqlite3"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(DB_PATH),
        "OPTIONS": {"timeout": 20},
    }
}

# --------------------------------------------------------------------------
# Password validation (kept for the bundled admin app)
# --------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# --------------------------------------------------------------------------
# Internationalisation
# --------------------------------------------------------------------------

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.environ.get("DJANGO_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

# --------------------------------------------------------------------------
# Static files
# --------------------------------------------------------------------------

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "assets"]
STATIC_ROOT = BASE_DIR / "staticfiles"

#: WhiteNoise storage: gzip/brotli compression without the hashed-manifest
#: machinery, which keeps local development and Docker identical and avoids
#: "missing file in manifest" errors when assets are edited between builds.
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}
WHITENOISE_MAX_AGE = 0 if DEBUG else 31536000

MESSAGE_STORAGE = "django.contrib.messages.storage.session.SessionStorage"

# --------------------------------------------------------------------------
# SMS Spam Detector specific settings
# --------------------------------------------------------------------------

#: Where trained pipelines and evaluation results live. Mirrors
#: ``detector.ml.config.ARTIFACTS_DIR`` so Django settings and the ML layer can
#: never disagree.
MODEL_ARTIFACTS_DIR = Path(
    os.environ.get("MODEL_ARTIFACTS_DIR") or (BASE_DIR / "artifacts")
)
if not MODEL_ARTIFACTS_DIR.is_absolute():
    MODEL_ARTIFACTS_DIR = BASE_DIR / MODEL_ARTIFACTS_DIR

#: Dataset location (the professor-provided SMS Spam Collection file).
DATA_DIR = Path(os.environ.get("DATA_DIR") or (BASE_DIR / "data" / "raw"))
if not DATA_DIR.is_absolute():
    DATA_DIR = BASE_DIR / DATA_DIR

DETECTOR_MAX_MESSAGE_LENGTH = int(os.environ.get("DETECTOR_MAX_MESSAGE_LENGTH", 1600))
DETECTOR_DEFAULT_MODEL = os.environ.get("DETECTOR_DEFAULT_MODEL", "linear_svm")

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL", "INFO").upper()

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{asctime}] {levelname} {name}: {message}",
            "style": "{",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        }
    },
    "root": {"handlers": ["console"], "level": LOG_LEVEL},
    "loggers": {
        "detector": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
        "django.request": {"handlers": ["console"], "level": "WARNING", "propagate": False},
    },
}

# --------------------------------------------------------------------------
# Production hardening (opt-in through environment variables)
# --------------------------------------------------------------------------

if not DEBUG:
    SECURE_SSL_REDIRECT = env_bool("DJANGO_SECURE_SSL_REDIRECT", False)
    SESSION_COOKIE_SECURE = env_bool("DJANGO_SESSION_COOKIE_SECURE", True)
    CSRF_COOKIE_SECURE = env_bool("DJANGO_CSRF_COOKIE_SECURE", True)
    SECURE_HSTS_SECONDS = int(os.environ.get("DJANGO_SECURE_HSTS_SECONDS", 0))
    SECURE_HSTS_INCLUDE_SUBDOMAINS = SECURE_HSTS_SECONDS > 0
    SECURE_HSTS_PRELOAD = SECURE_HSTS_SECONDS > 0
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_REFERRER_POLICY = "same-origin"
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    if not os.environ.get("DJANGO_SECRET_KEY"):
        raise RuntimeError(
            "DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is disabled."
        )
    if not ALLOWED_HOSTS:
        raise RuntimeError(
            "DJANGO_ALLOWED_HOSTS must list the served hostnames when DJANGO_DEBUG is disabled."
        )
