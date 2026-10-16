"""Training orchestration (steps 3-6 of the specification).

The module is framework-agnostic: it is called by the ``train_models`` Django
management command, by the tests and from a plain Python shell.

What it does
------------
1. Load and audit the dataset (:mod:`detector.ml.data`).
2. Build a reproducible, stratified train/test split.
3. Train three TF-IDF pipelines (Naive Bayes, Logistic Regression, Linear SVM).
4. Evaluate them on the untouched test split and cross-validate on the training
   split for a stability check.
5. Persist every fitted pipeline, a metrics JSON file and a Markdown report.

Artefacts are written into a temporary directory and moved into place at the end,
so the web application never observes a half-written model file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config
from .data import DatasetBundle, DatasetSplit, load_dataset, split_dataset
from .evaluation import evaluate_classifier, extract_top_features

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


class TrainingError(RuntimeError):
    """Raised when the training pipeline cannot be completed."""


def _noop(message: str) -> None:  # pragma: no cover - trivial
    logger.info(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


@dataclass
class TrainingResult:
    """Everything the caller (command, tests, UI) needs to know about a run."""

    metrics: dict[str, Any]
    artifacts: dict[str, Path]
    models: dict[str, Any]
    bundle: DatasetBundle
    split: DatasetSplit
    duration_seconds: float

    @property
    def best_model_key(self) -> str:
        return str(self.metrics["best_model"]["key"])


def _cross_validate(pipeline: Any, X: pd.Series, y: pd.Series, folds: int) -> dict[str, Any] | None:
    """Stratified k-fold F1 on the *training* split only (never touches the test set)."""
    smallest_class = int(y.value_counts().min()) if len(y) else 0
    usable_folds = min(folds, smallest_class)
    if usable_folds < 2:
        return None
    cv = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=config.RANDOM_STATE)
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=1)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Cross-validation failed: %s", exc)
        return None
    return {
        "folds": usable_folds,
        "scoring": "f1 (spam as positive class)",
        "mean_f1": round(float(scores.mean()), 4),
        "std_f1": round(float(scores.std()), 4),
        "scores": [round(float(s), 4) for s in scores],
    }


def _build_report(
    metrics: dict[str, Any],
    dataset_stats: dict[str, Any],
    split_stats: dict[str, Any],
    model_features: dict[str, Any],
) -> str:
    """Render the Markdown training report handed in with the project."""
    lines: list[str] = []
    add = lines.append
    env = metrics["environment"]

    add("# SMS Spam Detector - Training Report")
    add("")
    add(f"- Generated: {metrics['trained_at']}")
    add(f"- Dataset file: `{dataset_stats['source_file']}`")
    add(f"- Dataset SHA-256: `{metrics['dataset']['sha256']}`")
    add(
        f"- Python {env['python']} | scikit-learn {env['scikit_learn']} | "
        f"pandas {env['pandas']} | numpy {env['numpy']}"
    )
    add(f"- Training time: {metrics['duration_seconds']} s")
    add("")
    add("## 1. Dataset")
    add("")
    add(f"- Rows in file: **{dataset_stats['rows_in_file']}**")
    add(f"- Rows used after cleaning: **{dataset_stats['rows_after_cleaning']}**")
    add(
        f"- Missing messages: {dataset_stats['missing_messages']} | "
        f"missing labels: {dataset_stats['missing_labels']}"
    )
    add(
        f"- Duplicate rows found: {dataset_stats['duplicate_rows']} "
        f"(removed: {'yes' if dataset_stats['duplicates_removed'] else 'no'})"
    )
    add(f"- Identical text with conflicting labels: {dataset_stats['conflicting_labels_for_same_text']}")
    add(f"- Unrecognised labels dropped: {dataset_stats['unknown_label_rows_dropped']}")
    add("")
    add("| Class | Rows | Share |")
    add("| --- | ---: | ---: |")
    for label, count in dataset_stats["class_counts"].items():
        share = dataset_stats["class_distribution"][label] * 100
        add(f"| {config.LABEL_DISPLAY[label]} | {count} | {share:.2f}% |")
    add("")
    add(
        f"Message length (characters): min {dataset_stats['message_length']['min']}, "
        f"median {dataset_stats['message_length']['median']}, "
        f"mean {dataset_stats['message_length']['mean']}, "
        f"max {dataset_stats['message_length']['max']}."
    )
    add("")
    add("## 2. Split")
    add("")
    add(
        f"- Test size: {split_stats['test_size']} | random_state: {split_stats['random_state']} | "
        f"stratified: {split_stats['stratified']}"
    )
    add(f"- Train rows: {split_stats['train_rows']} | Test rows: {split_stats['test_rows']}")
    add(f"- Positive class for Precision/Recall/F1: **{split_stats['positive_class']}**")
    add("")
    add("## 3. Text cleaning and features")
    add("")
    add("- Unicode NFKC normalisation, lowercasing, non-ASCII removal.")
    add(
        "- URLs, e-mails, phone numbers, money amounts and numbers replaced by "
        "placeholder tokens (`__url__`, `__email__`, `__phone__`, `__money__`, `__num__`)."
    )
    add(
        f"- TF-IDF: ngram_range={config.VECTORIZER_PARAMS['ngram_range']}, "
        f"min_df={config.VECTORIZER_PARAMS['min_df']}, max_df={config.VECTORIZER_PARAMS['max_df']}, "
        f"sublinear_tf={config.VECTORIZER_PARAMS['sublinear_tf']}."
    )
    add(
        "- The vectorizer is fitted **inside** each pipeline on the training split only "
        "(no data leakage)."
    )
    add("")
    add("## 4. Results on the held-out test split")
    add("")
    add("| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | ROC-AUC | CV mean F1 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, model in metrics["models"].items():
        cv = model.get("cross_validation") or {}
        cv_text = f"{cv['mean_f1']:.4f}" if cv else "n/a"
        row_metrics = model["metrics"]
        add(
            f"| {model['display_name']} | {row_metrics['accuracy']:.4f} | "
            f"{row_metrics['precision']:.4f} | {row_metrics['recall']:.4f} | "
            f"{row_metrics['f1']:.4f} | {row_metrics['roc_auc']:.4f} | {cv_text} |"
        )
    add("")
    add(
        f"Best model by F1 (spam): **{metrics['best_model']['display_name']}** "
        f"(F1 = {metrics['best_model']['f1']:.4f})."
    )
    add("")
    add("## 5. Confusion matrices (test split)")
    add("")
    for key, model in metrics["models"].items():
        cm = model["metrics"]["confusion_matrix"]
        add(f"**{model['display_name']}**")
        add("")
        add("|  | predicted Ham | predicted Spam |")
        add("| --- | ---: | ---: |")
        add(f"| actual Ham | {cm['true_negative']} | {cm['false_positive']} |")
        add(f"| actual Spam | {cm['false_negative']} | {cm['true_positive']} |")
        add("")
    add("## 6. Strongest learned indicators")
    add("")
    for key, top in model_features.items():
        if not top.get("spam"):
            continue
        name = metrics["models"][key]["display_name"]
        add(f"**{name}** - spam side:")
        add("")
        add(", ".join(f"`{item['term']}` ({item['weight']:+.3f})" for item in top["spam"][:10]))
        add("")
        add(f"**{name}** - ham side:")
        add("")
        add(", ".join(f"`{item['term']}` ({item['weight']:+.3f})" for item in top["ham"][:10]))
        add("")
    add("## 7. Limitations")
    add("")
    add(
        "- The corpus is a 2011-vintage UK collection: vocabulary, currency and phone "
        "number formats differ from today's messages, so accuracy will drop on modern traffic."
    )
    add(
        "- Roughly 5% of the corpus is spam: high accuracy alone is misleading, which is "
        "why precision, recall and F1 for the spam class are reported as well."
    )
    add("- Model scores are estimates/decision margins, not calibrated proof.")
    add(
        "- The classifier only sees the message text; it has no metadata "
        "(sender reputation, sending frequency, link content)."
    )
    add("")
    return "\n".join(lines)


def train_all_models(
    dataset_path: Path | str | None = None,
    *,
    models_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    model_keys: list[str] | None = None,
    progress: ProgressFn | None = None,
    write_report: bool = True,
    drop_duplicates: bool | None = None,
) -> TrainingResult:
    """Train, evaluate and persist the requested models.

    Parameters
    ----------
    dataset_path:
        Overrides :data:`detector.ml.config.DATA_DIR` / ``SMSSpamCollection.txt``.
    models_dir / artifacts_dir:
        Overwrite the artefact locations (used by the test suite to train into a
        temporary directory).
    model_keys:
        Subset of :data:`detector.ml.config.MODEL_SPECS` to train. Defaults to
        all three algorithms required by the specification.
    progress:
        Callback receiving human-readable progress lines.
    write_report:
        Also write ``training_report.md`` next to ``metrics.json``.
    """
    emit = progress or _noop
    started = time.perf_counter()

    target_artifacts = Path(artifacts_dir) if artifacts_dir else config.ARTIFACTS_DIR
    target_models = Path(models_dir) if models_dir else (target_artifacts / "models")
    keys = list(model_keys or config.model_keys())
    unknown = [key for key in keys if key not in config.MODEL_SPECS]
    if unknown:
        raise TrainingError(f"Unknown model key(s): {', '.join(unknown)}")
    if not keys:
        raise TrainingError("No models selected for training.")

    emit("Step 1/6  Loading and auditing the dataset ...")
    bundle = load_dataset(path=dataset_path, drop_duplicates=drop_duplicates)
    stats = bundle.stats
    emit(
        f"          {stats['rows_after_cleaning']} messages "
        f"({stats['class_counts'][config.HAM_LABEL]} ham / "
        f"{stats['class_counts'][config.SPAM_LABEL]} spam), "
        f"{stats['rows_removed_as_duplicates']} duplicates removed"
    )

    emit("Step 2/6  Cleaning text and building TF-IDF features (inside each pipeline) ...")
    split = split_dataset(bundle)
    emit(
        f"          train={split.stats['train_rows']} rows, test={split.stats['test_rows']} rows, "
        f"stratified={split.stats['stratified']}, random_state={split.stats['random_state']}"
    )

    # Train into a temporary directory first so a failed run cannot leave the
    # application with a partially written artefact set.
    scratch_parent = target_artifacts.parent
    if not scratch_parent.exists():
        scratch_parent = Path(tempfile.gettempdir())
    tmp_root = Path(tempfile.mkdtemp(prefix=".training-", dir=str(scratch_parent)))
    tmp_models = tmp_root / "models"
    tmp_models.mkdir(parents=True, exist_ok=True)

    metrics_models: dict[str, Any] = {}
    features: dict[str, Any] = {}
    vocab_sizes: dict[str, int] = {}
    try:
        for index, key in enumerate(keys, start=3):
            spec = config.MODEL_SPECS[key]
            emit(f"Step {index}/6  Training {spec.display_name} ...")
            pipeline = spec.build()

            fit_started = time.perf_counter()
            pipeline.fit(split.X_train, split.y_train)
            fit_seconds = time.perf_counter() - fit_started

            model_metrics = evaluate_classifier(pipeline, split.X_test, split.y_test)
            cv = _cross_validate(pipeline, split.X_train, split.y_train, config.CV_FOLDS)
            top_features = extract_top_features(pipeline)
            vectorizer = pipeline.named_steps["tfidf"]

            artifact_file = tmp_models / f"{key}.joblib"
            joblib.dump(pipeline, artifact_file, compress=3)

            metrics_models[key] = {
                "key": key,
                "display_name": spec.display_name,
                "short_description": spec.short_description,
                "family": spec.family,
                "score_type": spec.score_type,
                "score_label": spec.score_label,
                "score_hint": spec.score_hint,
                "about": spec.about,
                "hyperparameters": spec.hyperparameters,
                "metrics": model_metrics,
                "cross_validation": cv,
                "training_seconds": round(fit_seconds, 3),
                "artifact": f"models/{key}.joblib",
                "artifact_bytes": artifact_file.stat().st_size,
                "vocabulary_size": int(len(vectorizer.get_feature_names_out())),
                "top_features": top_features,
            }
            features[key] = top_features
            vocab_sizes[key] = int(len(vectorizer.get_feature_names_out()))
            emit(
                f"          accuracy={model_metrics['accuracy']:.4f}  "
                f"precision={model_metrics['precision']:.4f}  "
                f"recall={model_metrics['recall']:.4f}  "
                f"f1={model_metrics['f1']:.4f}  "
                f"({model_metrics['latency']['per_message_ms']:.3f} ms/message)"
            )

        best_key = max(
            keys,
            key=lambda k: (
                metrics_models[k]["metrics"]["f1"] or 0.0,
                metrics_models[k]["metrics"]["accuracy"] or 0.0,
            ),
        )
        emit(f"          Best model by F1 (spam): {config.MODEL_SPECS[best_key].display_name}")

        metrics: dict[str, Any] = {
            "schema_version": 1,
            "project": "SMS Spam Detector",
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_seconds": round(time.perf_counter() - started, 2),
            "positive_class": config.POSITIVE_LABEL,
            "negative_class": config.NEGATIVE_LABEL,
            "environment": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "dataset": {
                **stats,
                "sha256": _sha256(bundle.source_path) if bundle.source_path else None,
                "usage": "loaded exactly as provided; only labels were normalised to ham/spam",
            },
            "split": split.stats,
            "features": {
                "vectorizer": "TfidfVectorizer",
                "params": {
                    **config.VECTORIZER_PARAMS,
                    "ngram_range": list(config.VECTORIZER_PARAMS["ngram_range"]),
                },
                "cleaning": "detector.ml.cleaning.clean_text (inside each pipeline)",
                "vocabulary_size": vocab_sizes,
                "fitted_on": "training split only (inside each Pipeline)",
            },
            "models": metrics_models,
            "best_model": {
                "key": best_key,
                "display_name": config.MODEL_SPECS[best_key].display_name,
                "f1": metrics_models[best_key]["metrics"]["f1"],
                "accuracy": metrics_models[best_key]["metrics"]["accuracy"],
                "selected_by": "highest F1 on the spam class, then accuracy",
            },
            "artifacts": {
                "directory": str(target_artifacts),
                "models": {key: f"models/{key}.joblib" for key in keys},
                "metrics_file": config.METRICS_FILENAME,
            },
        }

        # --- Publish artefacts (copy + atomic replace) ----------------------
        target_models.mkdir(parents=True, exist_ok=True)
        publish_started = time.perf_counter()
        for key in keys:
            source = tmp_models / f"{key}.joblib"
            destination = target_models / f"{key}.joblib"
            temporary = destination.with_suffix(".joblib.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)  # atomic on POSIX and Windows

            metadata = {
                "key": key,
                "display_name": config.MODEL_SPECS[key].display_name,
                "trained_at": metrics["trained_at"],
                "pipeline_steps": [step[0] for step in config.MODEL_SPECS[key].build().steps],
                "hyperparameters": config.MODEL_SPECS[key].hyperparameters,
                "score_type": config.MODEL_SPECS[key].score_type,
                "sklearn_version": sklearn.__version__,
                "feature_count": vocab_sizes[key],
                "metrics": metrics_models[key]["metrics"],
                "dataset_sha256": metrics["dataset"]["sha256"],
                "positive_class": config.POSITIVE_LABEL,
            }
            (target_models / f"{key}.meta.json").write_text(
                json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
            )

        target_artifacts.mkdir(parents=True, exist_ok=True)
        (target_artifacts / config.METRICS_FILENAME).write_text(
            json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
        )
        if write_report:
            (target_artifacts / config.REPORT_FILENAME).write_text(
                _build_report(metrics, stats, split.stats, features), encoding="utf-8"
            )
        emit(
            f"Step 6/6  Saved {len(keys)} model(s), metrics and report in {target_artifacts} "
            f"({time.perf_counter() - publish_started:.2f} s)"
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # Hand the caller the persisted pipelines (tests and the training command use
    # them directly; the web application loads them through the registry instead).
    trained_models = {key: joblib.load(target_models / f"{key}.joblib") for key in keys}

    return TrainingResult(
        metrics=metrics,
        artifacts={
            "metrics": target_artifacts / config.METRICS_FILENAME,
            "report": target_artifacts / config.REPORT_FILENAME,
            "models_dir": target_models,
        },
        models=trained_models,
        bundle=bundle,
        split=split,
        duration_seconds=round(time.perf_counter() - started, 2),
    )
