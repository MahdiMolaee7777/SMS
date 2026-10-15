"""Step 1 & 4 of the specification: load the SMS corpus, inspect it and split it.

Responsibilities
----------------
* Load the professor-provided ``SMSSpamCollection`` file **without changing its
  content** (delimiter, header and encoding are detected, not assumed).
* Normalise the label column into the two canonical classes ``ham`` / ``spam``.
* Report missing values, duplicates and the class distribution.
* Produce a reproducible, stratified train/test split that never leaks test
  messages into the training data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

logger = logging.getLogger(__name__)

MESSAGE_COLUMN = "message"
LABEL_COLUMN = "label"
LABEL_INT_COLUMN = "label_int"


class DatasetError(RuntimeError):
    """Raised when the dataset file is missing or cannot be parsed."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_CANDIDATE_DELIMITERS = ["\t", ";", ","]
_HEADER_NAMES = {"label", "v1", "class", "category", "target", "type"}


def _read_raw(path: Path) -> pd.DataFrame:
    """Read the file with a delimiter/header/encoding detected from its content."""
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            head = path.read_text(encoding=encoding, errors="strict")[: 64 * 1024]
        except (UnicodeDecodeError, OSError) as exc:  # pragma: no cover - fallback path
            last_error = exc
            continue

        first_line = head.splitlines()[0] if head.strip() else ""
        # Detect the delimiter by counting occurrences in the first line.
        delimiter = max(_CANDIDATE_DELIMITERS, key=first_line.count)
        if first_line.count(delimiter) == 0:
            delimiter = "\t"

        # A header is present when the first field is a known column name and is
        # *not* a label value such as "ham"/"spam".
        first_field = first_line.split(delimiter)[0].strip().lower()
        has_header = first_field in _HEADER_NAMES and first_field not in config.LABEL_ALIASES

        try:
            frame = pd.read_csv(
                path,
                sep=delimiter,
                header=0 if has_header else None,
                names=None if has_header else [LABEL_COLUMN, MESSAGE_COLUMN],
                engine="python",
                quoting=3,  # csv.QUOTE_NONE: spam rows contain quotes and commas
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
            )
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            continue

        frame.columns = [str(c).strip().lower() for c in frame.columns]
        return frame

    raise DatasetError(
        f"Could not parse the dataset at '{path}' "
        f"({last_error}). Expected a two-column file such as "
        f"'SMSSpamCollection' (label<TAB>message)."
    )


def normalize_label(raw_label: Any) -> str | None:
    """Map a raw label to ``ham`` / ``spam``.

    Returns ``None`` for anything that is not a recognised spelling, so the
    caller can decide to drop and *count* it instead of guessing.
    """
    if raw_label is None:
        return None
    text = str(raw_label).strip().lower()
    text = " ".join(text.split())  # collapse inner whitespace / newlines
    if not text:
        return None
    if text in config.LABEL_ALIASES:
        return config.LABEL_ALIASES[text]
    # Accept "spam." / "ham!" style noise.
    stripped = text.strip(".,;:!?()[]{}\"'")
    return config.LABEL_ALIASES.get(stripped)


@dataclass
class DatasetBundle:
    """A loaded, cleaned dataset plus the statistics needed for the report."""

    frame: pd.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.frame)

    @property
    def messages(self) -> pd.Series:
        return self.frame[MESSAGE_COLUMN]

    @property
    def labels(self) -> pd.Series:
        return self.frame[LABEL_COLUMN]


def load_dataset(
    path: Path | str | None = None,
    drop_duplicates: bool | None = None,
) -> DatasetBundle:
    """Load, normalise and audit the SMS Spam Collection.

    Parameters
    ----------
    path:
        Location of the dataset file. Defaults to
        ``data/raw/<DEFAULT_DATASET_FILENAME>`` (overridable with ``DATA_DIR``).
    drop_duplicates:
        Remove rows whose ``(label, message)`` pair already appeared. Defaults
        to :data:`detector.ml.config.DROP_EXACT_DUPLICATES`. Dropping them keeps
        identical messages out of both the training and the test split; the
        count of removed rows is always reported.
    """
    dataset_path = Path(path) if path is not None else config.DATA_DIR / config.DEFAULT_DATASET_FILENAME
    if not dataset_path.exists():
        raise DatasetError(
            f"Dataset file not found: '{dataset_path}'. "
            f"Place the professor-provided SMSSpamCollection file in '{config.DATA_DIR}' "
            f"or set the DATA_DIR environment variable."
        )
    if drop_duplicates is None:
        drop_duplicates = config.DROP_EXACT_DUPLICATES

    raw = _read_raw(dataset_path)

    if LABEL_COLUMN not in raw.columns or MESSAGE_COLUMN not in raw.columns:
        # Fallback: assume the first two columns are label and message.
        if raw.shape[1] >= 2:
            raw = raw.iloc[:, :2]
            raw.columns = [LABEL_COLUMN, MESSAGE_COLUMN]
        else:
            raise DatasetError(
                f"'{dataset_path}' must contain two columns (label, message); "
                f"found {list(raw.columns)}."
            )

    # --- Missing values -----------------------------------------------------
    missing_messages = int(raw[MESSAGE_COLUMN].isna().sum()) + int(
        (raw[MESSAGE_COLUMN].astype(str).str.strip() == "").sum()
    )
    missing_labels = int(raw[LABEL_COLUMN].isna().sum())

    frame = raw.copy()
    frame[MESSAGE_COLUMN] = frame[MESSAGE_COLUMN].astype(str).str.strip()
    frame = frame[frame[MESSAGE_COLUMN] != ""]  # drop empty / whitespace-only rows

    # --- Label normalisation ------------------------------------------------
    frame["raw_label"] = frame[LABEL_COLUMN].astype(str)
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].map(normalize_label)
    unknown_labels = frame[frame[LABEL_COLUMN].isna()]["raw_label"].value_counts().to_dict()
    frame = frame[frame[LABEL_COLUMN].notna()].copy()

    # --- Duplicates ---------------------------------------------------------
    duplicate_rows = int(frame.duplicated(subset=[LABEL_COLUMN, MESSAGE_COLUMN]).sum())
    conflicting = int(
        frame.groupby(MESSAGE_COLUMN)[LABEL_COLUMN].nunique().gt(1).sum()
    )  # identical text labelled both ham and spam -> ambiguous, must be reported
    if drop_duplicates:
        frame = frame.drop_duplicates(subset=[LABEL_COLUMN, MESSAGE_COLUMN], keep="first")

    frame = frame.reset_index(drop=True)
    frame[LABEL_INT_COLUMN] = frame[LABEL_COLUMN].map(config.LABEL_TO_INT).astype(int)

    class_counts = {
        label: int(count)
        for label, count in frame[LABEL_COLUMN].value_counts().reindex(
            [config.HAM_LABEL, config.SPAM_LABEL], fill_value=0
        ).items()
    }
    total = int(len(frame))
    lengths = frame[MESSAGE_COLUMN].str.len()

    stats: dict[str, Any] = {
        "source_file": str(dataset_path),
        "rows_in_file": int(len(raw)),
        "rows_after_cleaning": total,
        "missing_messages": missing_messages,
        "missing_labels": missing_labels,
        "duplicate_rows": duplicate_rows,
        "duplicates_removed": bool(drop_duplicates),
        "rows_removed_as_duplicates": duplicate_rows if drop_duplicates else 0,
        "conflicting_labels_for_same_text": conflicting,
        "unknown_labels": unknown_labels,
        "unknown_label_rows_dropped": int(sum(unknown_labels.values())),
        "class_counts": class_counts,
        "class_distribution": {
            label: round(count / total, 4) if total else 0.0 for label, count in class_counts.items()
        },
        "imbalance_ratio_ham_to_spam": round(
            class_counts[config.HAM_LABEL] / class_counts[config.SPAM_LABEL], 3
        )
        if class_counts[config.SPAM_LABEL]
        else None,
        "message_length": {
            "min": int(lengths.min()) if total else 0,
            "max": int(lengths.max()) if total else 0,
            "mean": round(float(lengths.mean()), 2) if total else 0.0,
            "median": float(lengths.median()) if total else 0.0,
        },
    }

    logger.info(
        "Loaded %s rows from %s (%s ham / %s spam, %s duplicates removed)",
        total,
        dataset_path.name,
        class_counts[config.HAM_LABEL],
        class_counts[config.SPAM_LABEL],
        stats["rows_removed_as_duplicates"],
    )
    return DatasetBundle(frame=frame[[MESSAGE_COLUMN, LABEL_COLUMN, LABEL_INT_COLUMN]], stats=stats, source_path=dataset_path)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


@dataclass
class DatasetSplit:
    """Train/test split returned by :func:`split_dataset`."""

    X_train: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series
    stats: dict[str, Any] = field(default_factory=dict)


def split_dataset(
    bundle: DatasetBundle,
    test_size: float | None = None,
    random_state: int | None = None,
    stratify: bool | None = None,
) -> DatasetSplit:
    """Split the dataset reproducibly, stratifying on the class when possible.

    ``stratify`` auto-enables only when both classes are present and each has at
    least two members, because scikit-learn raises otherwise.
    """
    test_size = config.TEST_SIZE if test_size is None else test_size
    random_state = config.RANDOM_STATE if random_state is None else random_state
    frame = bundle.frame

    if stratify is None:
        counts = frame[LABEL_COLUMN].value_counts()
        stratify = bool(len(counts) == 2 and counts.min() >= 2)

    X = frame[MESSAGE_COLUMN]
    y = frame[LABEL_INT_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
        shuffle=True,
    )

    def distribution(series: pd.Series) -> dict[str, float]:
        total = len(series) or 1
        return {
            config.INT_TO_LABEL[int(value)]: round(float(count) / total, 4)
            for value, count in series.value_counts().items()
        }

    split_stats: dict[str, Any] = {
        "test_size": test_size,
        "random_state": random_state,
        "stratified": bool(stratify),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_class_distribution": distribution(y_train),
        "test_class_distribution": distribution(y_test),
        "train_class_counts": {
            config.INT_TO_LABEL[int(k)]: int(v) for k, v in y_train.value_counts().items()
        },
        "test_class_counts": {
            config.INT_TO_LABEL[int(k)]: int(v) for k, v in y_test.value_counts().items()
        },
        "positive_class": config.POSITIVE_LABEL,
    }
    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, stats=split_stats)
