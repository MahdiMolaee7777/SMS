"""``python manage.py train_models`` - the explicit training command.

Runs the full pipeline described in the project specification: load the dataset,
clean the text, build TF-IDF features, split the data, train Naive Bayes,
Logistic Regression and the Linear SVM, evaluate them on the held-out test split,
and persist the pipelines plus ``metrics.json`` and a Markdown report.

Examples
--------
Train everything (default)::

    python manage.py train_models

Train only one algorithm (useful while iterating)::

    python manage.py train_models --model naive_bayes

Use a different dataset copy::

    python manage.py train_models --dataset data/raw/SMSSpamCollection.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from detector.ml import config as ml_config
from detector.ml.data import DatasetError
from detector.ml.prediction import registry
from detector.ml.training import TrainingError, train_all_models


class Command(BaseCommand):
    help = "Train and evaluate the SMS spam classifiers and store the artefacts."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--dataset",
            type=Path,
            default=None,
            help=(
                "Path to the SMS Spam Collection file "
                f"(default: {ml_config.DATA_DIR / ml_config.DEFAULT_DATASET_FILENAME})."
            ),
        )
        parser.add_argument(
            "--model",
            action="append",
            choices=ml_config.model_keys(),
            dest="models",
            help="Train only this model (repeatable). Default: all three algorithms.",
        )
        parser.add_argument(
            "--keep-duplicates",
            action="store_true",
            help=(
                "Keep exact duplicate messages instead of dropping them. "
                "Default: duplicates are removed to prevent identical rows from "
                "appearing in both the train and the test split."
            ),
        )
        parser.add_argument(
            "--no-report",
            action="store_true",
            help="Do not write artifacts/training_report.md.",
        )
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Only print the final summary table.",
        )

    def handle(self, *args: Any, **options: Any) -> str:
        dataset_path: Path | None = options.get("dataset")
        models: list[str] | None = options.get("models")
        quiet: bool = bool(options.get("quiet"))

        self.stdout.write(self.style.MIGRATE_HEADING("SMS Spam Detector - training"))
        self.stdout.write(
            f"Dataset : {dataset_path or (ml_config.DATA_DIR / ml_config.DEFAULT_DATASET_FILENAME)}"
        )
        self.stdout.write(f"Models  : {', '.join(models) if models else ', '.join(ml_config.model_keys())}")
        self.stdout.write(f"Artefacts: {ml_config.ARTIFACTS_DIR}")
        self.stdout.write("")

        def progress(message: str) -> None:
            if not quiet:
                self.stdout.write(message)

        try:
            result = train_all_models(
                dataset_path=dataset_path,
                model_keys=models,
                progress=progress,
                write_report=not options.get("no_report"),
                drop_duplicates=False if options.get("keep_duplicates") else None,
            )
        except DatasetError as exc:
            raise CommandError(str(exc)) from exc
        except TrainingError as exc:
            raise CommandError(str(exc)) from exc

        # Make this process aware of the new artefacts immediately.
        registry.clear()

        self.stdout.write("")
        header = f"{'model':<34}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>9}{'ms/msg':>10}"
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        for key, model in result.metrics["models"].items():
            metrics = model["metrics"]
            self.stdout.write(
                f"{model['display_name']:<34}"
                f"{metrics['accuracy']:>10.4f}"
                f"{metrics['precision']:>11.4f}"
                f"{metrics['recall']:>9.4f}"
                f"{metrics['f1']:>9.4f}"
                f"{metrics['latency']['per_message_ms']:>10.3f}"
            )
        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Best model by F1 (spam): {result.metrics['best_model']['display_name']} "
                f"(F1={result.metrics['best_model']['f1']:.4f})"
            )
        )
        self.stdout.write(f"Evaluation results: {result.artifacts['metrics']}")
        if not options.get("no_report"):
            self.stdout.write(f"Training report   : {result.artifacts['report']}")
        self.stdout.write(
            self.style.SUCCESS(f"Finished in {result.duration_seconds} s. Open /model-comparison/ to inspect it.")
        )
        return ""
