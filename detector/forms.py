"""Django forms: robust, server-side validation of everything the user submits."""

from __future__ import annotations

from typing import Any

from django import forms
from django.conf import settings

from .ml import config as ml_config
from .ml.prediction import InvalidMessageError, validate_message


def model_choices() -> list[tuple[str, str]]:
    """Return ``(key, display name)`` pairs for the model selector."""
    return [(spec.key, spec.display_name) for spec in ml_config.MODEL_SPECS.values()]


class AnalyzeMessageForm(forms.Form):
    """Validates the SMS body and the selected classifier.

    Validation lives here (and in :func:`detector.ml.prediction.validate_message`)
    so the same rules apply to the HTML form, to the JSON endpoint and to
    programmatic calls.
    """

    message = forms.CharField(
        label="SMS message",
        required=True,
        strip=True,
        widget=forms.Textarea(
            attrs={
                "rows": 6,
                "maxlength": ml_config.MAX_MESSAGE_LENGTH,
                "placeholder": (
                    "Paste the English SMS text you want to analyse, e.g. "
                    "\"Congratulations! You have won a £500 prize. Call 09061701461 to claim.\""
                ),
                "class": "detector-textarea",
                "aria-describedby": "message-help message-counter",
                "autocomplete": "off",
                "spellcheck": "false",
            }
        ),
        error_messages={
            "required": "Please enter a message before analysing it.",
        },
    )

    model = forms.ChoiceField(
        label="Classifier",
        required=False,
        choices=[],  # filled in __init__ so it always reflects the registry
        widget=forms.Select(attrs={"class": "detector-select"}),
        error_messages={"invalid_choice": "Please choose one of the available classifiers."},
    )

    max_length = ml_config.MAX_MESSAGE_LENGTH

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # ``available_keys`` lets the view restrict the picker to trained models.
        available_keys = kwargs.pop("available_keys", None)
        super().__init__(*args, **kwargs)

        choices = model_choices()
        if available_keys is not None:
            allowed = set(available_keys)
            choices = [choice for choice in choices if choice[0] in allowed] or choices

        self.fields["model"].choices = choices
        default_model = getattr(settings, "DETECTOR_DEFAULT_MODEL", ml_config.DEFAULT_MODEL_KEY)
        if not self.is_bound:
            keys = {key for key, _ in choices}
            self.fields["model"].initial = default_model if default_model in keys else choices[0][0]

    def clean_message(self) -> str:
        """Apply the shared message validation rules and produce helpful errors."""
        raw = self.cleaned_data["message"]
        try:
            return validate_message(raw, max_length=self.max_length)
        except InvalidMessageError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_model(self) -> str:
        """Fall back to the configured default when no classifier was selected."""
        selected = self.cleaned_data.get("model") or getattr(
            settings, "DETECTOR_DEFAULT_MODEL", ml_config.DEFAULT_MODEL_KEY
        )
        valid_keys = {key for key, _ in self.fields["model"].choices}
        if selected not in valid_keys:
            raise forms.ValidationError("Please choose one of the available classifiers.")
        return selected
