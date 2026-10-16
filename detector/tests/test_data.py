"""Tests for dataset loading, label normalisation and train/test splitting."""

from __future__ import annotations

import tempfile
from pathlib import Path

from django.test import SimpleTestCase

from detector.ml import config as ml_config
from detector.ml.data import (
    DatasetError,
    load_dataset,
    normalize_label,
    split_dataset,
)


class LabelNormalisationTests(SimpleTestCase):
    """Requirement: dataset label normalisation."""

    def test_known_spellings_are_mapped(self) -> None:
        cases = {
            "ham": "ham",
            "HAM": "ham",
            " ham ": "ham",
            "0": "ham",
            "normal": "ham",
            "spam": "spam",
            "Spam": "spam",
            "1": "spam",
            "junk": "spam",
            "spam.": "spam",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_label(raw), expected)

    def test_unknown_label_returns_none(self) -> None:
        for raw in ["", "   ", None, "promotion", "42", "unclassified"]:
            with self.subTest(raw=raw):
                self.assertIsNone(normalize_label(raw))


class DatasetLoadingTests(SimpleTestCase):
    """Requirement: successful loading of the dataset."""

    def test_provided_dataset_loads_and_is_audited(self) -> None:
        bundle = load_dataset()
        stats = bundle.stats

        self.assertGreater(len(bundle), 5000)
        self.assertEqual(
            stats["rows_in_file"],
            stats["rows_after_cleaning"]
            + stats["rows_removed_as_duplicates"]
            + stats["unknown_label_rows_dropped"]
            + stats["missing_messages"],
        )
        self.assertEqual(set(stats["class_counts"]), {"ham", "spam"})
        self.assertEqual(stats["missing_labels"], 0)
        self.assertGreater(stats["class_counts"]["ham"], stats["class_counts"]["spam"])
        self.assertEqual(stats["class_distribution"]["spam"], round(
            stats["class_counts"]["spam"] / stats["rows_after_cleaning"], 4
        ))

        # The dataframe exposes exactly the two canonical columns plus the
        # integer encoding used for modelling.
        self.assertEqual(
            list(bundle.frame.columns), ["message", "label", "label_int"]
        )
        self.assertTrue(bundle.frame["label"].isin(["ham", "spam"]).all())
        self.assertTrue(bundle.frame["message"].str.len().gt(0).all())
        self.assertEqual(
            bundle.frame["label_int"].sum(), stats["class_counts"]["spam"]
        )

    def test_duplicates_are_handled_and_counted(self) -> None:
        with_duplicates = load_dataset(drop_duplicates=False)
        without_duplicates = load_dataset(drop_duplicates=True)

        self.assertGreater(
            len(with_duplicates), len(without_duplicates)
        )
        self.assertEqual(
            without_duplicates.stats["rows_removed_as_duplicates"],
            with_duplicates.stats["duplicate_rows"],
        )
        self.assertTrue(
            with_duplicates.frame.duplicated(subset=["label", "message"]).any()
        )
        self.assertFalse(
            without_duplicates.frame.duplicated(subset=["label", "message"]).any()
        )

    def test_missing_file_raises_a_clear_error(self) -> None:
        with self.assertRaises(DatasetError) as context:
            load_dataset(path=Path(tempfile.gettempdir()) / "does-not-exist.csv")
        self.assertIn("not found", str(context.exception))

    def test_semicolon_delimited_file_with_header_is_parsed(self) -> None:
        """The loader detects the delimiter and the header instead of assuming them."""
        content = (
            "label;message\n"
            "ham;Hello there\n"
            "spam;WIN £1000 now\n"
            "ham;Another normal message\n"
            "spam;Call 09061701461 to claim your prize\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.csv"
            path.write_text(content, encoding="utf-8")
            bundle = load_dataset(path=path)
        self.assertEqual(len(bundle), 4)
        self.assertEqual(bundle.stats["class_counts"], {"ham": 2, "spam": 2})

    def test_unrecognised_labels_are_dropped_and_reported(self) -> None:
        content = (
            "ham\tHello there\n"
            "spam\tWIN £1000 now\n"
            "promotion\tBuy this\n"
            "\tEmpty label row\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.txt"
            path.write_text(content, encoding="utf-8")
            bundle = load_dataset(path=path)
        self.assertEqual(len(bundle), 2)
        # "promotion" is unrecognised and the last row carries an empty label.
        self.assertEqual(bundle.stats["unknown_label_rows_dropped"], 2)
        self.assertIn("promotion", bundle.stats["unknown_labels"])


class SplitTests(SimpleTestCase):
    """Requirement: reproducible train/test splitting with stratification."""

    def setUp(self) -> None:
        self.bundle = load_dataset()

    def test_split_is_reproducible(self) -> None:
        first = split_dataset(self.bundle)
        second = split_dataset(self.bundle)
        self.assertListEqual(list(first.X_train), list(second.X_train))
        self.assertListEqual(list(first.y_test), list(second.y_test))

    def test_split_is_stratified_and_disjoint(self) -> None:
        split = split_dataset(self.bundle)
        stats = split.stats

        self.assertTrue(stats["stratified"])
        self.assertEqual(
            stats["train_rows"] + stats["test_rows"], len(self.bundle)
        )
        self.assertEqual(
            stats["train_class_counts"]["spam"] + stats["test_class_counts"]["spam"],
            self.bundle.stats["class_counts"]["spam"],
        )
        # Class ratio preserved within a small tolerance.
        train_ratio = stats["train_class_distribution"]["spam"]
        test_ratio = stats["test_class_distribution"]["spam"]
        self.assertLess(abs(train_ratio - test_ratio), 0.01)

        # No message may appear in both splits.
        self.assertEqual(
            set(split.X_train).intersection(set(split.X_test)), set()
        )

    def test_positive_class_is_spam(self) -> None:
        split = split_dataset(self.bundle)
        self.assertEqual(ml_config.POSITIVE_LABEL, "spam")
        self.assertEqual(split.stats["positive_class"], "spam")
        self.assertEqual(ml_config.LABEL_TO_INT["spam"], 1)
        self.assertEqual(ml_config.LABEL_TO_INT["ham"], 0)
