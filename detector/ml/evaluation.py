"""Step 6 of the specification: evaluation metrics and model introspection.

All numbers produced here come from running the fitted pipelines on the held-out
test split - nothing is estimated, copied from a paper or hardcoded.

Positive class convention
-------------------------
The specification asks for Precision / Recall / F1 with the **spam** class as the
positive class.  Labels are encoded as ``ham = 0`` and ``spam = 1``
(:data:`detector.ml.config.LABEL_TO_INT`), so ``pos_label=1`` everywhere.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def _clean_float(value: Any, digits: int = 4) -> float | None:
    """Convert numpy scalars to JSON-safe rounded floats."""
    if value is None:
        return None
    number = float(value)
    if np.isnan(number):
        return None
    return round(number, digits)


def evaluate_classifier(
    pipeline: Any,
    X_test: Any,
    y_test: Any,
    latency_repeats: int = 3,
) -> dict[str, Any]:
    """Compute the full metric set for one fitted pipeline.

    Returns a JSON-serialisable dictionary containing accuracy, precision,
    recall, F1 (spam as positive class), ROC-AUC, the 2x2 confusion matrix and
    a fair per-message inference latency measured on the test split.
    """
    started = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_seconds = time.perf_counter() - started
    n_samples = max(len(y_test), 1)

    # Average per-message inference latency (repeat a few times and keep the best
    # run so a busy machine does not distort the number).
    best_seconds = predict_seconds
    for _ in range(max(latency_repeats - 1, 0)):
        tick = time.perf_counter()
        pipeline.predict(X_test)
        best_seconds = min(best_seconds, time.perf_counter() - tick)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    # ROC-AUC uses the model score. For LinearSVC that is the decision function
    # (not a probability) - AUC only needs a ranking, so it is still valid, and
    # the score type is recorded so nothing is presented as a probability.
    try:
        if hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(X_test)[:, 1]
        else:
            scores = pipeline.decision_function(X_test)
        roc_auc = _clean_float(roc_auc_score(y_test, scores))
    except Exception:  # pragma: no cover - degenerate splits
        roc_auc = None

    return {
        "accuracy": _clean_float(accuracy_score(y_test, y_pred)),
        "precision": _clean_float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": _clean_float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": _clean_float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": roc_auc,
        "positive_class": config.POSITIVE_LABEL,
        "confusion_matrix": {
            "labels": [config.HAM_LABEL, config.SPAM_LABEL],
            "matrix": [[tn, fp], [fn, tp]],
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "interpretation": {
                "true_negative": "Ham correctly classified as Ham",
                "false_positive": "Ham wrongly flagged as Spam",
                "false_negative": "Spam that slipped through",
                "true_positive": "Spam correctly detected",
            },
        },
        "support": {
            config.HAM_LABEL: int(np.sum(np.asarray(y_test) == 0)),
            config.SPAM_LABEL: int(np.sum(np.asarray(y_test) == 1)),
        },
        "test_rows": int(len(y_test)),
        "latency": {
            "total_predict_seconds": round(predict_seconds, 4),
            "per_message_ms": round(best_seconds / n_samples * 1000, 3),
        },
    }


def extract_top_features(pipeline: Any, top_k: int = 12) -> dict[str, list[dict[str, Any]]]:
    """Return the strongest spam / ham indicators learned by the model.

    This is real introspection of the fitted model, not a keyword list:

    * ``MultinomialNB``  -> ``feature_log_prob_`` (spam) minus (ham)
    * ``LogisticRegression`` / ``LinearSVC`` -> ``coef_``
    """
    try:
        vectorizer = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["clf"]
        names = np.asarray(vectorizer.get_feature_names_out())

        if hasattr(classifier, "feature_log_prob_"):
            weights = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
        elif hasattr(classifier, "coef_"):
            weights = np.ravel(classifier.coef_)
        else:  # pragma: no cover - unsupported estimator
            return {"spam": [], "ham": []}

        order = np.argsort(weights)
        spam_idx = order[::-1][:top_k]
        ham_idx = order[:top_k]

        def pack(indices: np.ndarray) -> list[dict[str, Any]]:
            return [
                {"term": str(names[i]), "weight": round(float(weights[i]), 4)}
                for i in indices
                if abs(float(weights[i])) > 0
            ]

        return {"spam": pack(spam_idx), "ham": pack(ham_idx)}
    except Exception:  # pragma: no cover - never let introspection break training
        return {"spam": [], "ham": []}


def explain_prediction(pipeline: Any, cleaned_text: str, top_k: int = 6) -> list[dict[str, Any]]:
    """Score the n-grams that pushed a *single* message towards spam or ham.

    Contribution = TF-IDF value of the n-gram x its learned weight.  The result
    is a heuristic explanation of the linear model, not a causal analysis.
    """
    try:
        vectorizer = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["clf"]
        matrix = vectorizer.transform([cleaned_text])

        if hasattr(classifier, "feature_log_prob_"):
            weights = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
        elif hasattr(classifier, "coef_"):
            weights = np.ravel(classifier.coef_)
        else:  # pragma: no cover
            return []

        # A single message has only a handful of non-zero features, so iterate the
        # sparse row directly (contribution = tf-idf value x learned weight).
        row = matrix.tocoo()
        names = vectorizer.get_feature_names_out()
        items = [
            {
                "term": str(names[column]),
                "contribution": round(float(value * weights[column]), 4),
                "direction": "spam" if value * weights[column] > 0 else "ham",
            }
            for column, value in zip(row.col, row.data)
        ]
        items.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return items[:top_k]
    except Exception:  # pragma: no cover - explanations are best-effort
        return []
