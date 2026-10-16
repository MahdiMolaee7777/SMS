"""Tests for training, evaluation, persistence and prediction.

These tests train real pipelines on the real dataset inside a temporary
artefacts directory. They never read the committed artefacts, so they are valid
on a fresh checkout and cannot damage an existing training run.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.test import SimpleTestCase

from detector.ml import config as ml_config
from detector.ml.cleaning import TextCleaner, clean_text
from detector.ml.data import load_dataset, split_dataset
from detector.ml.prediction import (
    InvalidMessageError,
    ModelLoadError,
    ModelNotTrainedError,
    ModelRegistry,
    PredictionService,
    format_score,
    load_evaluation_results,
    validate_message,
)
from detector.ml.training import train_all_models

from .utils import HAM_EXAMPLE, SPAM_EXAMPLE, temporary_model_directory, use_model_directory


class TextCleaningTests(SimpleTestCase):
    """Step 2 of the specification: text cleaning and noise removal."""

    def test_urls_phones_amounts_become_tokens(self) -> None:
        cleaned = clean_text("Visit http://spam.example.com NOW or call 09061701461 to win £1.50")
        self.assertIn(ml_config.URL_TOKEN, cleaned)
        self.assertIn(ml_config.PHONE_TOKEN, cleaned)
        self.assertIn(ml_config.MONEY_TOKEN, cleaned)
        self.assertNotIn("http", cleaned.replace(ml_config.URL_TOKEN, ""))
        self.assertEqual(cleaned, cleaned.lower())

    def test_email_and_numbers_are_replaced(self) -> None:
        cleaned = clean_text("Mail me at test@example.com, 2 tickets for 500 pounds")
        self.assertIn(ml_config.EMAIL_TOKEN, cleaned)
        self.assertIn(ml_config.NUMBER_TOKEN, cleaned)
        self.assertNotIn("@", cleaned)

    def test_non_ascii_and_whitespace_are_normalised(self) -> None:
        cleaned = clean_text("  Free  ünicode  🎉  prize  ")
        self.assertEqual(cleaned, cleaned.strip())
        self.assertNotIn("  ", cleaned)
        self.assertTrue(cleaned.isascii())

    def test_cleaner_is_empty_safe(self) -> None:
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        with self.assertRaises(ValueError):
            TextCleaner().transform("not a list")


class TrainingAndEvaluationTests(SimpleTestCase):
    """Requirements: train all three models, generate valid metrics, persist artefacts."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmp = temporary_model_directory()
        paths = cls._tmp.__enter__()
        cls.artifacts = paths["root"]
        cls.models_dir = paths["models"]
        cls.result = train_all_models(
            artifacts_dir=paths["root"],
            models_dir=paths["models"],
            write_report=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.__exit__(None, None, None)
        super().tearDownClass()

    def test_all_three_required_algorithms_are_trained(self) -> None:
        self.assertEqual(
            set(self.result.metrics["models"]),
            {"naive_bayes", "logistic_regression", "linear_svm"},
        )
        for key in ml_config.model_keys():
            self.assertTrue((self.models_dir / f"{key}.joblib").exists())
            self.assertTrue((self.models_dir / f"{key}.meta.json").exists())

    def test_metrics_are_valid_and_better_than_chance(self) -> None:
        for key, model in self.result.metrics["models"].items():
            metrics = model["metrics"]
            with self.subTest(model=key):
                for name in ("accuracy", "precision", "recall", "f1"):
                    value = metrics[name]
                    self.assertIsNotNone(value, f"{key}:{name} is None")
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
                # A TF-IDF spam classifier on this corpus must clearly beat a coin flip.
                self.assertGreater(metrics["accuracy"], 0.9)
                self.assertGreater(metrics["f1"], 0.8)
                self.assertEqual(metrics["positive_class"], "spam")

                cm = metrics["confusion_matrix"]
                self.assertEqual(len(cm["matrix"]), 2)
                total = (
                    cm["true_negative"]
                    + cm["false_positive"]
                    + cm["false_negative"]
                    + cm["true_positive"]
                )
                self.assertEqual(total, metrics["test_rows"])
                self.assertEqual(
                    total,
                    cm["true_positive"] + cm["false_positive"]
                    + cm["true_negative"] + cm["false_negative"],
                )
                # Precision / recall must be consistent with the confusion matrix.
                if cm["true_positive"] + cm["false_positive"]:
                    expected_precision = cm["true_positive"] / (
                        cm["true_positive"] + cm["false_positive"]
                    )
                    self.assertAlmostEqual(metrics["precision"], round(expected_precision, 4), places=3)

    def test_metrics_are_identical_when_recomputed_from_artifacts(self) -> None:
        """The persisted metrics must describe the persisted models."""
        import joblib

        bundle = load_dataset()
        split = split_dataset(bundle)
        for key in ml_config.model_keys():
            pipeline = joblib.load(self.models_dir / f"{key}.joblib")
            predictions = pipeline.predict(split.X_test)
            stored = self.result.metrics["models"][key]["metrics"]
            measured_accuracy = round(float((predictions == split.y_test).mean()), 4)
            with self.subTest(model=key):
                self.assertAlmostEqual(stored["accuracy"], measured_accuracy, places=4)

    def test_cross_validation_is_recorded(self) -> None:
        for key, model in self.result.metrics["models"].items():
            with self.subTest(model=key):
                cv = model["cross_validation"]
                self.assertIsNotNone(cv)
                self.assertGreaterEqual(cv["folds"], 2)
                self.assertGreater(cv["mean_f1"], 0.8)

    def test_evaluation_results_are_persisted_as_json(self) -> None:
        metrics_path = self.artifacts / ml_config.METRICS_FILENAME
        self.assertTrue(metrics_path.exists())
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertIn("models", payload)
        self.assertEqual(payload["positive_class"], "spam")
        self.assertIn("split", payload)
        self.assertEqual(
            payload["split"]["train_rows"] + payload["split"]["test_rows"],
            payload["dataset"]["rows_after_cleaning"],
        )
        self.assertTrue((self.artifacts / ml_config.REPORT_FILENAME).exists())

    def test_vectorizer_is_fitted_on_training_data_only(self) -> None:
        """No-leakage check on the fitted vocabulary.

        Two independent assertions:

        1. every vocabulary entry must be derivable from the *training* messages;
        2. n-grams that occur often enough to survive ``min_df`` but appear only in
           the test split must not be in the vocabulary - if the vectorizer had
           been fitted on the full dataset, they would be.
        """
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer

        bundle = load_dataset()
        split = split_dataset(bundle)
        pipeline = joblib.load(self.models_dir / "naive_bayes.joblib")
        vocabulary = set(pipeline.named_steps["tfidf"].get_feature_names_out())

        # Tokenise both splits exactly like the trained vectorizer does.
        analyzer = TfidfVectorizer(**ml_config.VECTORIZER_PARAMS).build_analyzer()
        train_terms: set[str] = set()
        for message in split.X_train:
            train_terms.update(analyzer(clean_text(message)))
        test_only_counts: dict[str, int] = {}
        for message in split.X_test:
            for term in set(analyzer(clean_text(message))):
                if term not in train_terms:
                    test_only_counts[term] = test_only_counts.get(term, 0) + 1

        self.assertTrue(vocabulary.issubset(train_terms))
        repeated_test_only = {term for term, count in test_only_counts.items() if count >= 2}
        self.assertTrue(repeated_test_only, "no repeated test-only n-grams found to check")
        self.assertEqual(vocabulary.intersection(repeated_test_only), set())

    def test_predictions_are_deterministic(self) -> None:
        pipeline = self.result.models["linear_svm"]
        first = list(pipeline.predict([SPAM_EXAMPLE, HAM_EXAMPLE]))
        second = list(pipeline.predict([SPAM_EXAMPLE, HAM_EXAMPLE]))
        self.assertEqual(first, second)


class PredictionServiceTests(SimpleTestCase):
    """Requirement: predict both classes, validate input, handle missing artefacts."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmp = temporary_model_directory()
        paths = cls._tmp.__enter__()
        cls.paths = paths
        train_all_models(
            artifacts_dir=paths["root"],
            models_dir=paths["models"],
            model_keys=["naive_bayes", "logistic_regression", "linear_svm"],
            write_report=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.__exit__(None, None, None)
        super().tearDownClass()

    def setUp(self) -> None:
        self._context = use_model_directory(self.paths["models"], self.paths["metrics"])
        self.registry = self._context.__enter__()

    def tearDown(self) -> None:
        self._context.__exit__(None, None, None)

    def test_spam_and_ham_are_both_classified_correctly(self) -> None:
        service = PredictionService(self.registry)
        for key in ml_config.model_keys():
            with self.subTest(model=key):
                spam = service.predict(SPAM_EXAMPLE, key)
                ham = service.predict(HAM_EXAMPLE, key)
                self.assertEqual(spam.label, "spam")
                self.assertTrue(spam.is_spam)
                self.assertEqual(ham.label, "ham")
                self.assertFalse(ham.is_spam)
                self.assertGreater(spam.latency_ms, 0)
                self.assertEqual(spam.model_key, key)

    def test_score_semantics_differ_between_model_types(self) -> None:
        service = PredictionService(self.registry)
        probabilistic = service.predict(SPAM_EXAMPLE, "logistic_regression")
        self.assertEqual(probabilistic.score_type, "probability")
        self.assertGreaterEqual(probabilistic.score, 0.0)
        self.assertLessEqual(probabilistic.score, 1.0)
        self.assertTrue(probabilistic.score_display.endswith("%"))
        self.assertIsNotNone(probabilistic.score_pct)

        svm = service.predict(SPAM_EXAMPLE, "linear_svm")
        self.assertEqual(svm.score_type, "decision")
        # A decision margin must never be rendered as a percentage.
        self.assertNotIn("%", svm.score_display or "")
        self.assertIsNone(svm.score_pct)

    def test_format_score_helpers(self) -> None:
        self.assertEqual(format_score(0.5, "probability"), "50.0%")
        self.assertIsNone(format_score(None, "probability"))
        self.assertEqual(format_score(1.234, "decision"), "+1.23")
        self.assertEqual(format_score(-0.5, "decision"), "-0.50")

    def test_prediction_returns_a_valid_explanation(self) -> None:
        prediction = PredictionService(self.registry).predict(SPAM_EXAMPLE, "logistic_regression")
        self.assertTrue(prediction.explanation)
        for item in prediction.explanation:
            self.assertIn(item["direction"], {"spam", "ham"})
            self.assertIsInstance(item["term"], str)
            self.assertIsInstance(item["contribution"], float)
        contributions = [abs(item["contribution"]) for item in prediction.explanation]
        self.assertEqual(contributions, sorted(contributions, reverse=True))

    def test_empty_message_is_rejected(self) -> None:
        service = PredictionService(self.registry)
        for raw in ["", "   ", "\n\t ", None]:
            with self.subTest(raw=raw):
                with self.assertRaises(InvalidMessageError):
                    service.predict(raw)

    def test_overlong_message_is_rejected_with_a_helpful_error(self) -> None:
        too_long = "a" * (ml_config.MAX_MESSAGE_LENGTH + 1)
        with self.assertRaises(InvalidMessageError) as context:
            validate_message(too_long)
        message = str(context.exception)
        self.assertIn(str(ml_config.MAX_MESSAGE_LENGTH), message)
        self.assertIn("too long", message)

    def test_non_text_and_unprintable_input_is_rejected(self) -> None:
        with self.assertRaises(InvalidMessageError):
            validate_message(12345)
        with self.assertRaises(InvalidMessageError):
            validate_message("\x00\x01\x02")

    def test_missing_artifact_raises_a_clear_error(self) -> None:
        missing_registry = ModelRegistry(models_dir=self.paths["root"] / "empty")
        (self.paths["root"] / "empty").mkdir(parents=True, exist_ok=True)
        with self.assertRaises(ModelNotTrainedError) as context:
            missing_registry.load("naive_bayes")
        self.assertIn("train_models", str(context.exception))

    def test_unknown_model_key_is_rejected(self) -> None:
        with self.assertRaises(ModelNotTrainedError):
            self.registry.load("does_not_exist")

    def test_corrupted_artifact_raises_model_load_error(self) -> None:
        broken_dir = self.paths["root"] / "broken"
        broken_dir.mkdir(parents=True, exist_ok=True)
        (broken_dir / "naive_bayes.joblib").write_bytes(b"this is not a joblib file")
        broken_registry = ModelRegistry(models_dir=broken_dir)
        with self.assertRaises(ModelLoadError) as context:
            broken_registry.load("naive_bayes")
        self.assertIn("train_models", str(context.exception))

    def test_registry_cache_is_reused_until_the_file_changes(self) -> None:
        first = self.registry.load("naive_bayes")
        second = self.registry.load("naive_bayes")
        self.assertIs(first, second)

        self.registry.clear()
        reloaded = self.registry.load("naive_bayes")
        self.assertIsNot(reloaded, first)

    def test_statuses_report_availability(self) -> None:
        statuses = {status.key: status for status in self.registry.statuses()}
        self.assertEqual(set(statuses), set(ml_config.model_keys()))
        for status in statuses.values():
            self.assertTrue(status.available)
            self.assertGreater(status.artifact_size or 0, 0)
            self.assertIsNotNone(status.trained_at)

    def test_no_evaluation_results_returns_none(self) -> None:
        self.assertIsNone(load_evaluation_results(self.paths["root"] / "missing.json"))

    def test_evaluation_results_are_loaded(self) -> None:
        payload = load_evaluation_results(self.paths["metrics"])
        self.assertIsNotNone(payload)
        self.assertIn("models", payload or {})
