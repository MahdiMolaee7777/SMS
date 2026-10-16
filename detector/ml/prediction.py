"""Model loading and prediction services.

Two public objects are exposed:

``ModelRegistry``
    Loads the persisted scikit-learn pipelines lazily and caches them in memory.
    The cache is keyed by model name and invalidated automatically when the
    artefact file changes on disk (detected through ``(mtime, size)``), so
    re-training while the server runs does not require a restart.

``PredictionService``
    Validates a raw message, applies the *persisted* pipeline (cleaning + TF-IDF +
    classifier are one object, so inference cannot drift from training) and
    returns a structured :class:`Prediction`.

Caching strategy and its limits
-------------------------------
Models are cached per process. With multiple worker processes (gunicorn, uwsgi)
each worker keeps its own copy, so memory usage grows linearly with the worker
count, and a freshly trained artefact is picked up by a worker the first time it
serves a prediction after the file changed. For this project's small models
(``< 1 MB``) both effects are negligible; a production deployment would use a
shared cache or a model server instead.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn

from . import config
from .cleaning import clean_text
from .evaluation import explain_prediction

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """Base class for model-related failures."""


class ModelNotTrainedError(ModelError):
    """Raised when a model artefact is missing (training has not been run)."""


class ModelLoadError(ModelError):
    """Raised when an artefact exists but cannot be used."""


class InvalidMessageError(ValueError):
    """Raised when the submitted message fails validation."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_message(raw_text: Any, *, max_length: int | None = None) -> str:
    """Validate and normalise user input.

    Returns the stripped message. Raises :class:`InvalidMessageError` with a
    user-facing message when the input is empty, too long or not text.
    """
    max_length = config.MAX_MESSAGE_LENGTH if max_length is None else max_length

    if raw_text is None:
        raise InvalidMessageError("Please enter a message to analyse.")
    if not isinstance(raw_text, str):
        raise InvalidMessageError("The message must be plain text.")

    message = raw_text.replace("\r\n", "\n").strip()
    if len(message) < config.MIN_MESSAGE_LENGTH:
        raise InvalidMessageError("Please enter a message before analysing it.")
    if not message:
        raise InvalidMessageError("Please enter a message before analysing it.")
    if len(message) > max_length:
        raise InvalidMessageError(
            f"The message is too long ({len(message)} characters). "
            f"Please submit at most {max_length} characters "
            f"(the limit of a concatenated SMS)."
        )
    # A message made only of control characters is not analysable.
    if not any(character.isprintable() and not character.isspace() for character in message):
        raise InvalidMessageError("The message must contain at least one readable character.")
    return message


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class ModelStatus:
    """Cheap, load-free description of a model, used to render the model picker."""

    key: str
    display_name: str
    short_description: str
    score_type: str
    score_label: str
    score_hint: str
    artifact_path: Path
    available: bool
    artifact_size: int | None = None
    trained_at: str | None = None
    error: str | None = None


@dataclass
class Predictor:
    """A loaded model plus everything needed to present its output honestly."""

    key: str
    display_name: str
    short_description: str
    score_type: str
    score_label: str
    score_hint: str
    pipeline: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_path: Path | None = None
    loaded_at: str = ""


class ModelRegistry:
    """Lazy, self-refreshing cache of trained pipelines."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
        self._predictors: dict[str, Predictor] = {}
        self._fingerprints: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    # -- helpers ---------------------------------------------------------
    def artifact_path(self, key: str) -> Path:
        return self.models_dir / f"{key}.joblib"

    def _fingerprint(self, path: Path) -> tuple[float, int] | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime, stat.st_size)

    def _read_metadata(self, key: str) -> dict[str, Any]:
        meta_path = self.models_dir / f"{key}.meta.json"
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # -- public API ------------------------------------------------------
    def is_available(self, key: str) -> bool:
        return key in config.MODEL_SPECS and self.artifact_path(key).exists()

    def any_available(self) -> bool:
        return any(self.is_available(key) for key in config.model_keys())

    def statuses(self) -> list[ModelStatus]:
        """Describe every configured model without loading it from disk."""
        statuses: list[ModelStatus] = []
        for key in config.model_keys():
            spec = config.MODEL_SPECS[key]
            path = self.artifact_path(key)
            exists = path.exists()
            metadata = self._read_metadata(key) if exists else {}
            statuses.append(
                ModelStatus(
                    key=key,
                    display_name=spec.display_name,
                    short_description=spec.short_description,
                    score_type=spec.score_type,
                    score_label=spec.score_label,
                    score_hint=spec.score_hint,
                    artifact_path=path,
                    available=exists,
                    artifact_size=path.stat().st_size if exists else None,
                    trained_at=metadata.get("trained_at"),
                    error=None
                    if exists
                    else "Not trained yet. Run: python manage.py train_models",
                )
            )
        return statuses

    def load(self, key: str) -> Predictor:
        """Return the (cached) predictor for ``key``.

        Raises
        ------
        ModelNotTrainedError
            When the artefact file does not exist.
        ModelLoadError
            When the artefact exists but cannot be deserialised (for example it
            was produced by an incompatible scikit-learn version).
        """
        if key not in config.MODEL_SPECS:
            raise ModelNotTrainedError(
                f"Unknown model '{key}'. Available models: {', '.join(config.model_keys())}."
            )

        path = self.artifact_path(key)
        fingerprint = self._fingerprint(path)
        if fingerprint is None:
            raise ModelNotTrainedError(
                f"The model '{config.MODEL_SPECS[key].display_name}' has not been trained yet "
                f"(missing artefact: {path}). "
                f"Run 'python manage.py train_models' to create it."
            )

        with self._lock:
            cached = self._predictors.get(key)
            if cached is not None and self._fingerprints.get(key) == fingerprint:
                return cached

            metadata = self._read_metadata(key)
            artifact_version = str(metadata.get("sklearn_version", ""))
            if artifact_version and artifact_version.split(".")[:2] != sklearn.__version__.split(".")[:2]:
                logger.warning(
                    "Model '%s' was trained with scikit-learn %s but %s is installed. "
                    "Retrain with 'python manage.py train_models' if loading fails.",
                    key,
                    artifact_version,
                    sklearn.__version__,
                )

            try:
                pipeline = joblib.load(path)
            except Exception as exc:
                raise ModelLoadError(
                    f"Could not load the model '{key}' from {path}: {exc}. "
                    f"The artefact may be corrupted or built with an incompatible "
                    f"library version. Re-run 'python manage.py train_models'."
                ) from exc

            missing_steps = [
                step for step in ("clean", "tfidf", "clf") if step not in getattr(pipeline, "named_steps", {})
            ]
            if missing_steps:
                raise ModelLoadError(
                    f"The artefact '{path}' is not a valid SMS Spam Detector pipeline "
                    f"(missing step(s): {', '.join(missing_steps)}). "
                    f"Re-run 'python manage.py train_models'."
                )

            spec = config.MODEL_SPECS[key]
            predictor = Predictor(
                key=key,
                display_name=spec.display_name,
                short_description=spec.short_description,
                score_type=spec.score_type,
                score_label=spec.score_label,
                score_hint=spec.score_hint,
                pipeline=pipeline,
                metadata=metadata,
                artifact_path=path,
                loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            self._predictors[key] = predictor
            self._fingerprints[key] = fingerprint
            logger.info("Loaded model '%s' (%s bytes) from %s", key, fingerprint[1], path)
            return predictor

    def clear(self) -> None:
        """Drop every cached predictor (used after training and by the tests)."""
        with self._lock:
            self._predictors.clear()
            self._fingerprints.clear()

    # -- evaluation results ---------------------------------------------
    def evaluation_results(self, metrics_path: Path | None = None) -> dict[str, Any] | None:
        """Load ``metrics.json`` produced by the training step, or ``None``."""
        return load_evaluation_results(metrics_path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    """Structured prediction result handed to the view/template."""

    label: str
    display_label: str
    model_key: str
    model_display_name: str
    score: float | None
    score_type: str
    score_display: str | None
    score_label: str
    score_hint: str
    latency_ms: float
    cleaned_text: str
    explanation: list[dict[str, Any]] = field(default_factory=list)
    message_length: int = 0
    predicted_at: str = ""

    @property
    def is_spam(self) -> bool:
        return self.label == config.SPAM_LABEL

    @property
    def score_pct(self) -> float | None:
        """Probability of spam in percent, or ``None`` for decision-margin models."""
        if self.score is None or self.score_type != "probability":
            return None
        return round(max(0.0, min(1.0, float(self.score))) * 100, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "display_label": self.display_label,
            "is_spam": self.is_spam,
            "model_key": self.model_key,
            "model_display_name": self.model_display_name,
            "score": self.score,
            "score_type": self.score_type,
            "score_display": self.score_display,
            "score_pct": self.score_pct,
            "score_label": self.score_label,
            "score_hint": self.score_hint,
            "latency_ms": self.latency_ms,
            "cleaned_text": self.cleaned_text,
            "explanation": self.explanation,
            "message_length": self.message_length,
            "predicted_at": self.predicted_at,
        }


def format_score(score: float | None, score_type: str) -> str | None:
    """Render a model score for display, without ever faking a probability.

    * ``probability`` -> ``"96.4%"`` (a real ``predict_proba`` output).
    * ``decision``    -> ``"+1.28"`` signed margin (LinearSVC decision function).
    """
    if score is None:
        return None
    if score_type == "probability":
        return f"{max(0.0, min(1.0, float(score))) * 100:.1f}%"
    return f"{float(score):+.2f}"


class PredictionService:
    """Ties the registry, validation and presentation flags together."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def predict(self, message: str, model_key: str | None = None) -> Prediction:
        """Classify ``message`` with the selected (or default) trained model."""
        cleaned_input = validate_message(message)
        key = model_key or config.DEFAULT_MODEL_KEY
        predictor = self.registry.load(key)

        started = time.perf_counter()
        prepared = predictor.pipeline.named_steps["clean"].transform([cleaned_input])[0]
        label_int = int(predictor.pipeline.predict([cleaned_input])[0])

        if predictor.score_type == "probability":
            probabilities = predictor.pipeline.predict_proba([cleaned_input])[0]
            classes = [int(c) for c in predictor.pipeline.named_steps["clf"].classes_]
            score = float(probabilities[classes.index(config.POSITIVE_CLASS_INT)])
        else:
            score = float(predictor.pipeline.decision_function([cleaned_input])[0])
        latency_ms = (time.perf_counter() - started) * 1000

        label = config.INT_TO_LABEL.get(label_int, config.HAM_LABEL)
        return Prediction(
            label=label,
            display_label=config.LABEL_DISPLAY[label],
            model_key=key,
            model_display_name=predictor.display_name,
            score=round(score, 6),
            score_type=predictor.score_type,
            score_display=format_score(score, predictor.score_type),
            score_label=predictor.score_label,
            score_hint=predictor.score_hint,
            latency_ms=round(latency_ms, 2),
            cleaned_text=prepared,
            explanation=explain_prediction(predictor.pipeline, prepared),
            message_length=len(cleaned_input),
            predicted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------


def load_evaluation_results(metrics_path: Path | None = None) -> dict[str, Any] | None:
    """Return the parsed ``metrics.json`` or ``None`` when training has not run."""
    path = Path(metrics_path) if metrics_path else config.METRICS_PATH
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read evaluation results from %s: %s", path, exc)
        return None
    if not isinstance(payload, dict) or "models" not in payload:
        logger.warning("Evaluation results in %s have an unexpected structure.", path)
        return None
    return payload


def build_chart_series(results: dict[str, Any]) -> dict[str, Any]:
    """Transform ``metrics.json`` into the series consumed by Chart.js.

    Built on the server so the template never contains hardcoded numbers and the
    page still renders coherently if the charting library cannot be loaded.
    """
    labels: list[str] = []
    accuracy: list[float] = []
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for model in results.get("models", {}).values():
        metrics = model.get("metrics", {})
        labels.append(model.get("display_name", model.get("key", "")))
        accuracy.append(round(float(metrics.get("accuracy") or 0.0) * 100, 2))
        precision.append(round(float(metrics.get("precision") or 0.0) * 100, 2))
        recall.append(round(float(metrics.get("recall") or 0.0) * 100, 2))
        f1.append(round(float(metrics.get("f1") or 0.0) * 100, 2))
    return {
        "labels": labels,
        "datasets": [
            {"label": "Accuracy", "data": accuracy},
            {"label": "Precision (spam)", "data": precision},
            {"label": "Recall (spam)", "data": recall},
            {"label": "F1 (spam)", "data": f1},
        ],
        "unit": "%",
    }


#: Shared, process-wide registry used by the views.
registry = ModelRegistry()
