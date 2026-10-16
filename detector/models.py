"""Database models for the detector app.

**This application has no database models, by design.**

Why:

* The project specification asks for an SMS classifier, not a bookkeeping tool.
  It does not require accounts, history or stored messages.
* Trained models and evaluation results are *files* (``artifacts/``) and are
  loaded with `joblib`, which makes them inspectable and shareable with the
  report.
* Storing the submitted SMS text would create a privacy liability (users type
  real messages into the box) with no benefit for the required functionality.

Django is still used in full - ORM, migrations, forms, CSRF protection, template
engine, admin - because those are framework features, not application data. The
bundled ``admin`` app is kept reachable for operators, and ``python manage.py
migrate`` only creates Django's own tables.

If a future deployment wants an audit trail, add a model here and create an
explicit migration; the recommended design is to store only a SHA-256 hash of
the message plus the prediction, never the raw text.
"""

from __future__ import annotations

__all__: list[str] = []
