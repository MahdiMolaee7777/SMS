"""Unit tests for the form layer (validation rules shared with the ML service)."""

from __future__ import annotations

from django.test import SimpleTestCase

from detector.forms import AnalyzeMessageForm
from detector.ml import config as ml_config


class AnalyzeMessageFormTests(SimpleTestCase):
    """Requirement: robust validation of user-submitted text."""

    def test_valid_submission_is_accepted(self) -> None:
        form = AnalyzeMessageForm({"message": "Hello, see you at 8pm.", "model": "naive_bayes"})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["model"], "naive_bayes")

    def test_message_is_stripped(self) -> None:
        form = AnalyzeMessageForm({"message": "   padded message   ", "model": "linear_svm"})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["message"], "padded message")

    def test_empty_and_whitespace_messages_are_rejected(self) -> None:
        for raw in ["", " ", "\n\t  "]:
            with self.subTest(raw=repr(raw)):
                form = AnalyzeMessageForm({"message": raw, "model": "linear_svm"})
                self.assertFalse(form.is_valid())
                self.assertIn("message", form.errors)

    def test_missing_message_field_is_rejected(self) -> None:
        form = AnalyzeMessageForm({"model": "linear_svm"})
        self.assertFalse(form.is_valid())
        self.assertIn("Please enter a message", str(form.errors["message"]))

    def test_message_at_the_length_limit_is_accepted(self) -> None:
        form = AnalyzeMessageForm(
            {"message": "a" * ml_config.MAX_MESSAGE_LENGTH, "model": "linear_svm"}
        )
        self.assertTrue(form.is_valid(), form.errors)

    def test_message_over_the_length_limit_is_rejected(self) -> None:
        form = AnalyzeMessageForm(
            {"message": "a" * (ml_config.MAX_MESSAGE_LENGTH + 1), "model": "linear_svm"}
        )
        self.assertFalse(form.is_valid())
        self.assertIn("too long", str(form.errors["message"]))

    def test_invalid_model_is_rejected(self) -> None:
        form = AnalyzeMessageForm({"message": "hello", "model": "random_forest"})
        self.assertFalse(form.is_valid())
        self.assertIn("classifier", str(form.errors["model"]).lower())

    def test_model_defaults_when_not_provided(self) -> None:
        form = AnalyzeMessageForm({"message": "hello"})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertIn(form.cleaned_data["model"], ml_config.model_keys())

    def test_picker_can_be_restricted_to_trained_models(self) -> None:
        form = AnalyzeMessageForm(available_keys=["naive_bayes"])
        self.assertEqual([key for key, _ in form.fields["model"].choices], ["naive_bayes"])
        # A model that is not offered must be rejected, not silently accepted.
        bound = AnalyzeMessageForm(
            {"message": "hello", "model": "linear_svm"}, available_keys=["naive_bayes"]
        )
        self.assertFalse(bound.is_valid())

    def test_widget_declares_the_maximum_length(self) -> None:
        form = AnalyzeMessageForm()
        self.assertEqual(int(form.fields["message"].widget.attrs["maxlength"]), ml_config.MAX_MESSAGE_LENGTH)
