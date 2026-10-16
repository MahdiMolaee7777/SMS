"""Central configuration for the machine-learning layer.

This module is intentionally **Django-independent**: the whole ``detector.ml``
package can be imported and executed from a plain Python interpreter or a
management command.  Django settings only *mirror* the values defined here
(see ``config/settings.py``).

Every path can be overridden with an environment variable so that the same code
works on a developer laptop, inside Docker and in CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment, resolving relative paths to project root."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


#: Directory that holds the original, untouched SMS Spam Collection file.
DATA_DIR: Path = _env_path("DATA_DIR", PROJECT_ROOT / "data" / "raw")

#: Default location of the professor-provided dataset file.
DEFAULT_DATASET_FILENAME: str = "SMSSpamCollection.txt"

#: Every generated artefact (models, metrics, reports) lives here.
ARTIFACTS_DIR: Path = _env_path("MODEL_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")

#: Trained scikit-learn pipelines.
MODELS_DIR: Path = ARTIFACTS_DIR / "models"

#: Machine-readable evaluation results (consumed by the Model Comparison page).
METRICS_FILENAME: str = "metrics.json"

#: Human-readable training report (nice to hand in to the professor).
REPORT_FILENAME: str = "training_report.md"

METRICS_PATH: Path = ARTIFACTS_DIR / METRICS_FILENAME
REPORT_PATH: Path = ARTIFACTS_DIR / REPORT_FILENAME

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

HAM_LABEL: str = "ham"
SPAM_LABEL: str = "spam"

#: The professor's specification evaluates the *spam* class, therefore spam is
#: encoded as the positive class (1) and used as ``pos_label`` everywhere.
POSITIVE_LABEL: str = SPAM_LABEL
NEGATIVE_LABEL: str = HAM_LABEL
POSITIVE_CLASS_INT: int = 1
NEGATIVE_CLASS_INT: int = 0

LABEL_TO_INT: dict[str, int] = {HAM_LABEL: 0, SPAM_LABEL: 1}
INT_TO_LABEL: dict[int, str] = {0: HAM_LABEL, 1: SPAM_LABEL}

#: Accepted spellings of the two classes.  Anything else is discarded (and
#: counted) instead of being silently coerced.
LABEL_ALIASES: dict[str, str] = {
    "ham": HAM_LABEL,
    "0": HAM_LABEL,
    "normal": HAM_LABEL,
    "legit": HAM_LABEL,
    "legitimate": HAM_LABEL,
    "not spam": HAM_LABEL,
    "nonspam": HAM_LABEL,
    "spam": SPAM_LABEL,
    "1": SPAM_LABEL,
    "junk": SPAM_LABEL,
    "smishing": SPAM_LABEL,
    "phishing": SPAM_LABEL,
    "spam message": SPAM_LABEL,
}

#: Human-friendly names used in the UI.
LABEL_DISPLAY: dict[str, str] = {HAM_LABEL: "Ham", SPAM_LABEL: "Spam"}

# --------------------------------------------------------------------------
# Cleaning / feature extraction
# --------------------------------------------------------------------------

#: Tokens that replace noisy entities. Underscores keep them inside scikit-learn's
#: default ``\w\w+`` token pattern, so "this message contains a URL" stays a
#: usable feature instead of being deleted.
URL_TOKEN: str = "__url__"
EMAIL_TOKEN: str = "__email__"
PHONE_TOKEN: str = "__phone__"
MONEY_TOKEN: str = "__money__"
NUMBER_TOKEN: str = "__num__"

#: ``TfidfVectorizer`` configuration, shared by all three models so the
#: comparison is fair.
VECTORIZER_PARAMS: dict[str, Any] = {
    "ngram_range": (1, 2),      # unigrams + bigrams ("free entry", "call now")
    "min_df": 2,                # drop n-grams seen once (they are pure noise)
    "max_df": 0.98,             # drop n-grams present in almost every message
    "sublinear_tf": True,       # 1 + log(tf), dampens very long messages
    "lowercase": True,
    "strip_accents": "unicode",
    "dtype": "float64",  # JSON-friendly; converted in _vectorizer()
}

# Engineering decision (documented): we do **not** remove English stop words.
# Spam/Ham differences live in exactly the short function words and call-to-action
# bigrams ("call now", "text to", "claim your") that stop-word removal destroys,
# and TF-IDF already down-weights uninformative terms.

# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------

TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42
CV_FOLDS: int = 5

#: Row duplicated across the whole dataset (label + identical text). The SMS Spam
#: Collection contains ~400 of them; keeping them would put identical messages in
#: both the train and the test split and inflate the reported scores.
DROP_EXACT_DUPLICATES: bool = True


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of one candidate model."""

    key: str
    display_name: str
    short_description: str
    family: str
    #: ``"probability"`` for calibrated-ish probabilities, ``"decision"`` for a
    #: raw SVM decision margin (never presented as a probability).
    score_type: str
    score_label: str
    score_hint: str
    build: Callable[[], Any]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    about: str = ""


def _build_naive_bayes() -> Any:
    """Multinomial Naive Bayes on TF-IDF counts."""
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    from .cleaning import TextCleaner

    return Pipeline(
        steps=[
            ("clean", TextCleaner()),
            ("tfidf", _vectorizer()),
            ("clf", MultinomialNB(alpha=0.1, fit_prior=True)),
        ]
    )


def _build_logistic_regression() -> Any:
    """Logistic Regression baseline with balanced class weights."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from .cleaning import TextCleaner

    return Pipeline(
        steps=[
            ("clean", TextCleaner()),
            ("tfidf", _vectorizer()),
            (
                "clf",
                LogisticRegression(
                    C=10.0,
                    solver="liblinear",
                    max_iter=2000,
                    class_weight="balanced",  # spam is only ~13% of the data
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _build_linear_svm() -> Any:
    """Linear Support Vector Machine (LinearSVC), the standard text-classification SVM."""
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    from .cleaning import TextCleaner

    return Pipeline(
        steps=[
            ("clean", TextCleaner()),
            ("tfidf", _vectorizer()),
            (
                "clf",
                LinearSVC(
                    C=1.0,
                    loss="squared_hinge",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _vectorizer() -> Any:
    """Build the shared TF-IDF vectorizer.

    ``VECTORIZER_PARAMS`` stores the dtype as a string so the configuration stays
    JSON-serialisable (it is copied into ``metrics.json``); here it is converted
    to the NumPy type scikit-learn expects.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    params = dict(VECTORIZER_PARAMS)
    dtype = params.get("dtype")
    if isinstance(dtype, str):
        params["dtype"] = getattr(np, dtype)
    return TfidfVectorizer(**params)


#: The three algorithms required by the project specification.
MODEL_SPECS: dict[str, ModelSpec] = {
    "naive_bayes": ModelSpec(
        key="naive_bayes",
        display_name="Multinomial Naive Bayes",
        short_description="Fast probabilistic baseline based on word frequencies.",
        family="naive_bayes",
        score_type="probability",
        score_label="Model score (estimated probability of spam)",
        score_hint=(
            "Multinomial Naive Bayes estimates class probabilities from word "
            "frequencies. The value is a model estimate, not proof that a "
            "message is safe or fraudulent."
        ),
        build=_build_naive_bayes,
        hyperparameters={"alpha": 0.1, "fit_prior": True},
        about=(
            "Naive Bayes applies Bayes' theorem and assumes every word is "
            "independent of the others. That assumption is wrong for language, "
            "yet the algorithm is extremely fast and a famously strong baseline "
            "for spam filtering, which is why it was used in the first "
            "generation of production spam filters."
        ),
    ),
    "logistic_regression": ModelSpec(
        key="logistic_regression",
        display_name="Logistic Regression",
        short_description="Linear classifier with interpretable per-word weights.",
        family="logistic_regression",
        score_type="probability",
        score_label="Model score (estimated probability of spam)",
        score_hint=(
            "Logistic Regression maps the TF-IDF features through a sigmoid "
            "function. The value is an estimated probability produced by the "
            "model, not a guarantee."
        ),
        build=_build_logistic_regression,
        hyperparameters={
            "C": 10.0,
            "solver": "liblinear",
            "max_iter": 2000,
            "class_weight": "balanced",
        },
        about=(
            "Logistic Regression learns one weight per TF-IDF feature and "
            "squashes the weighted sum through a sigmoid. It is the standard "
            "baseline for text classification: cheap to train, well calibrated "
            "in practice and easy to explain, because each weight can be read as "
            "'how spammy is this word'."
        ),
    ),
    "linear_svm": ModelSpec(
        key="linear_svm",
        display_name="Linear Support Vector Machine",
        short_description="Maximum-margin linear model, strong on high-dimensional text.",
        family="svm",
        score_type="decision",
        score_label="Model score (SVM decision margin, not a probability)",
        score_hint=(
            "LinearSVC outputs a distance from the separating hyperplane "
            "(decision function), not a probability. The sign gives the class; "
            "the magnitude only expresses how far from the boundary the message "
            "sits. It is shown as a margin, never as a percentage."
        ),
        build=_build_linear_svm,
        hyperparameters={
            "C": 1.0,
            "loss": "squared_hinge",
            "class_weight": "balanced",
            "max_iter": 5000,
        },
        about=(
            "A Support Vector Machine finds the hyperplane that separates spam "
            "from ham with the widest possible margin. With TF-IDF features the "
            "data is high-dimensional and almost linearly separable, so the "
            "linear variant (LinearSVC) is both accurate and fast."
        ),
    ),
}

#: Key used when the user does not pick a model explicitly.
DEFAULT_MODEL_KEY: str = "linear_svm"

# --------------------------------------------------------------------------
# Application-level limits (also mirrored into Django settings)
# --------------------------------------------------------------------------

#: Longest accepted SMS body. 1600 characters is the upper bound of a
#: concatenated (multi-part) SMS, so anything longer is not an SMS.
MAX_MESSAGE_LENGTH: int = 1600

#: Shortest accepted message after cleaning.
MIN_MESSAGE_LENGTH: int = 1


def artifact_path(model_key: str) -> Path:
    """Return the ``.joblib`` path of a trained model."""
    return MODELS_DIR / f"{model_key}.joblib"


def model_keys() -> list[str]:
    """Ordered list of the supported model keys."""
    return list(MODEL_SPECS.keys())
