"""Django application configuration for the detector app."""

import logging
import os

from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)


class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detector"
    verbose_name = "SMS Spam Detector"

    def ready(self) -> None:
        """Report artefact availability once per process (never trains anything).

        Training on startup would be slow and could race between workers, so the
        application only *checks* whether artefacts exist and tells the operator
        how to create them. This check can be turned off with
        ``DJANGO_SKIP_STARTUP_CHECK=1`` (useful when running migrations before the
        first training run).
        """
        if os.environ.get("DJANGO_SKIP_STARTUP_CHECK", "").lower() in {"1", "true", "yes"}:
            return
        if not getattr(settings, "MODEL_ARTIFACTS_DIR", None):
            return
        try:
            from .ml.prediction import registry

            available = [status.key for status in registry.statuses() if status.available]
            if available:
                logger.info(
                    "Trained models detected: %s (artefacts in %s)",
                    ", ".join(available),
                    settings.MODEL_ARTIFACTS_DIR,
                )
            else:
                logger.warning(
                    "No trained model artefacts found in %s. "
                    "Run 'python manage.py train_models' before using the detector.",
                    settings.MODEL_ARTIFACTS_DIR,
                )
        except Exception as exc:  # pragma: no cover - never break startup
            logger.warning("Could not inspect model artefacts at startup: %s", exc)
