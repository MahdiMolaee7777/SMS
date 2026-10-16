"""``python manage.py model_status`` - health check for the ML artefacts.

Prints what is available, whether the artefacts can be loaded and which
evaluation results are on disk. Useful as a first diagnostic when the detector
page shows a setup message.
"""

from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand

from detector.ml import config as ml_config
from detector.ml.prediction import load_evaluation_results, registry


class Command(BaseCommand):
    help = "Show which trained models and evaluation results are available."

    def handle(self, *args: Any, **options: Any) -> str:
        self.stdout.write(self.style.MIGRATE_HEADING("SMS Spam Detector - artefact status"))
        self.stdout.write(f"Dataset directory : {ml_config.DATA_DIR}")
        dataset_file = ml_config.DATA_DIR / ml_config.DEFAULT_DATASET_FILENAME
        self.stdout.write(
            f"Dataset file      : {dataset_file} "
            f"[{'found' if dataset_file.exists() else 'MISSING'}]"
        )
        self.stdout.write(f"Artefacts         : {ml_config.ARTIFACTS_DIR}")
        self.stdout.write("")
        self.stdout.write(f"{'model':<24}{'file':<28}{'status':<12}{'trained at':<22}")
        self.stdout.write("-" * 86)

        any_available = False
        for status in registry.statuses():
            if status.available:
                any_available = True
                self.stdout.write(
                    f"{status.key:<24}{status.artifact_path.name:<28}"
                    f"{'ok':<12}{(status.trained_at or 'unknown'):<22}"
                )
            else:
                self.stdout.write(
                    f"{status.key:<24}{status.artifact_path.name:<28}"
                    f"{'not trained':<12}{'-':<22}"
                )

        self.stdout.write("")
        for status in registry.statuses():
            if not status.available:
                continue
            try:
                predictor = registry.load(status.key)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Loaded {predictor.display_name}: "
                        f"{len(predictor.pipeline.named_steps['tfidf'].get_feature_names_out())} features"
                    )
                )
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f"Failed to load {status.key}: {exc}"))

        results = load_evaluation_results()
        if results:
            self.stdout.write("")
            self.stdout.write(
                f"Evaluation results: {ml_config.METRICS_PATH} "
                f"(trained {results.get('trained_at')})"
            )
        else:
            self.stdout.write("")
            self.stdout.write(
                self.style.WARNING(
                    "No evaluation results yet. Run 'python manage.py train_models' "
                    "to train the models and generate artifacts/metrics.json."
                )
            )
        if not any_available:
            self.stdout.write(
                self.style.WARNING("No trained models found -> the detector page will show a setup notice.")
            )
        return ""
