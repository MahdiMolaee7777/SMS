#!/usr/bin/env python
"""Django's command-line utility for administrative tasks.

Common commands::

    python manage.py migrate          # create Django's own database tables
    python manage.py train_models     # train the three classifiers (once)
    python manage.py runserver        # start the development server
    python manage.py test             # run the test suite
"""

import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import Django. Is it installed and is the virtual "
            "environment activated? (pip install -r requirements.txt)"
        ) from exc

    # Give a clearer error than a stack trace when the dataset is missing.
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
