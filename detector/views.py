"""Views for the SMS Spam Detector.

Pages (as required by the project specification):

* ``/``                  - Home
* ``/detector/``         - SMS Detector (form + result panel)
* ``/detector/analyze/`` - the same analysis as a JSON endpoint (used for the
  asynchronous UI; CSRF protected, POST only)
* ``/model-comparison/`` - evaluation results of the three trained algorithms
* ``/about/``            - About the project

Predictions always come from the persisted scikit-learn pipelines through
:mod:`detector.ml.prediction`. No keyword rules, no mocked results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.generic import TemplateView

from .forms import AnalyzeMessageForm
from .ml import config as ml_config
from .ml.prediction import (
    InvalidMessageError,
    ModelLoadError,
    ModelNotTrainedError,
    Prediction,
    PredictionService,
    build_chart_series,
    load_evaluation_results,
    registry,
)

logger = logging.getLogger(__name__)

#: Setup hint shown whenever artefacts are missing.
TRAINING_HINT = "python manage.py train_models"


def _model_cards() -> list[dict[str, Any]]:
    """Algorithm descriptions shared by the Home and About pages."""
    return [
        {
            "key": spec.key,
            "name": spec.display_name,
            "short": spec.short_description,
            "about": spec.about,
            "hyperparameters": spec.hyperparameters,
        }
        for spec in ml_config.MODEL_SPECS.values()
    ]


class HomeView(TemplateView):
    """Landing page: what the tool does, how the pipeline works, the algorithms."""

    template_name = "detector/home.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        results = load_evaluation_results()
        context.update(
            {
                "models": _model_cards(),
                "results_available": results is not None,
                "summary": (results or {}).get("dataset"),
                "split": (results or {}).get("split"),
                "best_model": (results or {}).get("best_model"),
                "trained_at": (results or {}).get("trained_at"),
                "any_model_available": registry.any_available(),
                "pipeline_steps": [
                    {
                        "number": "01",
                        "title": "Text cleaning",
                        "text": "Unicode normalisation, lower-casing and replacement of URLs, "
                                "e-mails, phone numbers and amounts by placeholder tokens.",
                    },
                    {
                        "number": "02",
                        "title": "TF-IDF features",
                        "text": "The cleaned message is converted into weighted unigram and "
                                "bigram features. The vectorizer is fitted on the training "
                                "split only.",
                    },
                    {
                        "number": "03",
                        "title": "Classifier",
                        "text": "One of three trained models (Naive Bayes, Logistic "
                                "Regression, Linear SVM) assigns the class Ham or Spam.",
                    },
                    {
                        "number": "04",
                        "title": "Reported result",
                        "text": "The predicted class is shown together with the model name, "
                                "inference time and an honest explanation of the model score.",
                    },
                ],
            }
        )
        return context


class DetectorView(TemplateView):
    """Main interactive page: validate a message, run the model, show the result."""

    template_name = "detector/detector.html"

    def get(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        context = self.get_context_data(**kwargs)
        context.update(self._base_context(request))
        return render(request, self.template_name, context)

    def post(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        base = self._base_context(request)
        form = AnalyzeMessageForm(request.POST, available_keys=base["available_keys"])
        prediction: Prediction | None = None
        error_message: str | None = None
        setup_required = False

        if form.is_valid():
            service = PredictionService(registry)
            try:
                prediction = service.predict(form.cleaned_data["message"], form.cleaned_data["model"])
            except ModelNotTrainedError as exc:
                setup_required = True
                error_message = str(exc)
                logger.warning("Prediction requested without trained model: %s", exc)
            except ModelLoadError as exc:
                error_message = str(exc)
                logger.error("Model failed to load: %s", exc)
            except InvalidMessageError as exc:  # pragma: no cover - guarded by the form
                form.add_error("message", str(exc))
            except Exception:  # pragma: no cover - defensive, never leak a traceback
                logger.exception("Unexpected error while predicting")
                error_message = (
                    "An unexpected error occurred while analysing the message. "
                    "Please try again; if it persists, re-run training with "
                    f"'{TRAINING_HINT}'."
                )

        context = {
            **base,
            "form": form,
            "prediction": prediction,
            "error_message": error_message,
            "setup_required": setup_required,
        }
        status = 200
        return render(request, self.template_name, context, status=status)

    def _base_context(self, request: HttpRequest) -> dict[str, Any]:
        statuses = registry.statuses()
        available_keys = [status.key for status in statuses if status.available]
        return {
            "form": AnalyzeMessageForm(available_keys=available_keys),
            "model_statuses": statuses,
            "available_keys": available_keys,
            "any_model_available": bool(available_keys),
            "training_hint": TRAINING_HINT,
            "max_message_length": ml_config.MAX_MESSAGE_LENGTH,
        }


class AnalyzeApiView(TemplateView):
    """JSON endpoint powering the asynchronous UI (POST only).

    ``POST {"message": "...", "model": "naive_bayes"}`` returns the same
    structured result that the server-rendered page shows. CSRF protection comes
    from Django's ``CsrfViewMiddleware`` (the JavaScript client sends the token
    in the ``X-CSRFToken`` header). Validation errors are returned as HTTP 400
    with user-facing messages, never a stack trace.
    """

    http_method_names = ["post", "options"]

    def post(self, request: HttpRequest, *args: Any, **kwargs: Any) -> JsonResponse:
        payload: dict[str, Any] = {}
        if request.content_type and "application/json" in request.content_type:
            try:
                payload = json.loads(request.body or b"{}")
            except json.JSONDecodeError:
                return JsonResponse(
                    {"ok": False, "errors": {"message": ["Malformed JSON payload."]}}, status=400
                )
        else:
            payload = request.POST.dict()

        if not isinstance(payload, dict):
            return JsonResponse(
                {"ok": False, "errors": {"message": ["Malformed request payload."]}}, status=400
            )

        statuses = registry.statuses()
        available_keys = [status.key for status in statuses if status.available]
        form = AnalyzeMessageForm(payload, available_keys=available_keys)
        if not form.is_valid():
            return JsonResponse(
                {
                    "ok": False,
                    "errors": {field: [str(message) for message in errors] for field, errors in form.errors.items()},
                    "any_model_available": bool(available_keys),
                },
                status=400,
            )

        service = PredictionService(registry)
        try:
            prediction = service.predict(form.cleaned_data["message"], form.cleaned_data["model"])
        except ModelNotTrainedError as exc:
            return JsonResponse(
                {"ok": False, "setup_required": True, "message": str(exc), "training_hint": TRAINING_HINT},
                status=503,
            )
        except ModelLoadError as exc:
            return JsonResponse({"ok": False, "message": str(exc)}, status=500)
        except InvalidMessageError as exc:
            return JsonResponse({"ok": False, "errors": {"message": [str(exc)]}}, status=400)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Unexpected error in the analyse endpoint")
            return JsonResponse(
                {"ok": False, "message": "Unexpected error while analysing the message."},
                status=500,
            )

        result = prediction.as_dict()
        result["ok"] = True
        return JsonResponse(result)


class ModelComparisonView(TemplateView):
    """Scientific evaluation results produced by ``manage.py train_models``."""

    template_name = "detector/model_comparison.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        results = load_evaluation_results()
        context.update(
            {
                "results": results,
                "results_available": results is not None,
                "training_hint": TRAINING_HINT,
                "metrics_file": str(settings.MODEL_ARTIFACTS_DIR / ml_config.METRICS_FILENAME),
            }
        )
        if results:
            models = list(results.get("models", {}).values())
            context.update(
                {
                    "model_rows": models,
                    "chart_data": build_chart_series(results),
                    "best_model": results.get("best_model"),
                    "dataset": results.get("dataset"),
                    "split": results.get("split"),
                    "features": results.get("features"),
                    "environment": results.get("environment"),
                    "trained_at": results.get("trained_at"),
                    "positive_class": results.get("positive_class", "spam"),
                    "cv_folds": next(
                        (
                            (model.get("cross_validation") or {}).get("folds")
                            for model in models
                            if model.get("cross_validation")
                        ),
                        None,
                    ),
                    "artifact_availability": {
                        status.key: status.available for status in registry.statuses()
                    },
                }
            )
        return context


class AboutView(TemplateView):
    """Project objective, dataset, TF-IDF, algorithms, technology stack."""

    template_name = "detector/about.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        results = load_evaluation_results()
        context.update(
            {
                "models": _model_cards(),
                "summary": (results or {}).get("dataset"),
                "split": (results or {}).get("split"),
                "trained_at": (results or {}).get("trained_at"),
                "results_available": results is not None,
                "technologies": [
                    {"name": "Python 3.12", "role": "Implementation language"},
                    {"name": "Django 5.2", "role": "Web framework (views, forms, templates, CSRF)"},
                    {"name": "scikit-learn 1.6", "role": "TF-IDF, Naive Bayes, Logistic Regression, LinearSVC"},
                    {"name": "pandas / NumPy", "role": "Dataset loading, auditing and numeric work"},
                    {"name": "Joblib", "role": "Persistence of the fitted pipelines"},
                    {"name": "Tailwind CSS (compiled)", "role": "Styling, compiled to a local CSS file"},
                    {"name": "Chart.js", "role": "Metric and confusion-matrix charts"},
                    {"name": "SQLite", "role": "Django's own tables (sessions, admin)"},
                    {"name": "Docker / Compose", "role": "Reproducible development and runtime environment"},
                    {"name": "Django test suite", "role": "Automated tests for ML and web layers"},
                ],
            }
        )
        return context


# ---------------------------------------------------------------------------
# Error handlers (wired up in config/urls.py)
# ---------------------------------------------------------------------------


def page_not_found(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """Branded 404 page."""
    return render(request, "404.html", status=404)


def server_error(request: HttpRequest) -> HttpResponse:
    """Branded 500 page (Django requires this signature)."""
    return render(request, "500.html", status=500)
