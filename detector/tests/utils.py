"""Shared test helpers.

The test suite never touches the committed ``artifacts/`` directory: it trains
into a temporary folder and re-points the process-wide registry at it, so running
the tests can never overwrite the artefacts the application uses - and the tests
pass on a fresh checkout where training has not been run yet.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

from detector.ml import config as ml_config
from detector.ml.prediction import ModelRegistry, registry


@contextlib.contextmanager
def temporary_model_directory() -> Iterator[dict[str, Path]]:
    """Context manager providing an isolated artefacts directory.

    Yields a mapping with ``root`` (artefacts root), ``models`` (model directory)
    and ``metrics`` (path of ``metrics.json``).
    """
    root = Path(tempfile.mkdtemp(prefix="sms-detector-tests-"))
    paths = {"root": root, "models": root / "models", "metrics": root / "metrics.json"}
    paths["models"].mkdir(parents=True, exist_ok=True)
    try:
        yield paths
    finally:
        shutil.rmtree(root, ignore_errors=True)


@contextlib.contextmanager
def use_model_directory(models_dir: Path, metrics_path: Path | None = None) -> Iterator[ModelRegistry]:
    """Point the global registry (and metrics lookup) at a temporary directory.

    Restores the previous location and cache afterwards, which keeps the test
    process's registry consistent between test classes.
    """
    previous_dir = registry.models_dir
    previous_metrics = ml_config.METRICS_PATH
    registry.models_dir = models_dir
    registry.clear()
    if metrics_path is not None:
        ml_config.METRICS_PATH = metrics_path
    try:
        yield registry
    finally:
        registry.models_dir = previous_dir
        registry.clear()
        ml_config.METRICS_PATH = previous_metrics


#: A spam-like and a ham-like message used across the tests. They are plain text
#: (no assertions about the expected label live here; each test states its own
#: expectation based on the measured behaviour of the trained model).
SPAM_EXAMPLE = (
    "WINNER!! As a valued network customer you have been selected to receive a "
    "£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
)
HAM_EXAMPLE = "Hi mum, I'll be home around 7pm. Can you put the pasta on? Thanks!"
