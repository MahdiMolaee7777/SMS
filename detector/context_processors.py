"""Template context processors.

Keeps project metadata available to every template (navigation, footer) without
duplicating it in each view.
"""

from __future__ import annotations

import datetime
from typing import Any

from django.conf import settings


def project_metadata(request: Any) -> dict[str, Any]:
    """Expose the project name, version and configured message limit."""
    return {
        "PROJECT_NAME": "SMS Spam Detector",
        "PROJECT_VERSION": "1.0.0",
        "PROJECT_TAGLINE": "Machine-learning classification of unwanted SMS messages",
        "MAX_MESSAGE_LENGTH": getattr(settings, "DETECTOR_MAX_MESSAGE_LENGTH", 1600),
        "current_year": datetime.date.today().year,
    }
