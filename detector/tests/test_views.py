"""HTTP-level tests for every page, the form submission and the JSON endpoint."""

from __future__ import annotations

import json
from pathlib import Path

from django.test import SimpleTestCase
from django.urls import reverse

from detector.ml import config as ml_config
from detector.ml.training import train_all_models

from .utils import HAM_EXAMPLE, SPAM_EXAMPLE, temporary_model_directory, use_model_directory


class PublicPageTests(SimpleTestCase):
    """Requirements: main page, detector page and comparison page availability."""

    def test_all_pages_respond_with_http_200(self) -> None:
        for name in ("detector:home", "detector:detector", "detector:model_comparison", "detector:about"):
            with self.subTest(page=name):
                response = self.client.get(reverse(name))
                self.assertEqual(response.status_code, 200)
                self.assertTemplateUsed(response, "base.html")

    def test_navigation_links_are_present_and_reachable(self) -> None:
        response = self.client.get(reverse("detector:home"))
        content = response.content.decode()
        for name in ("detector:detector", "detector:model_comparison", "detector:about"):
            url = reverse(name)
            with self.subTest(link=url):
                self.assertIn(f'href="{url}"', content)

    def test_home_page_has_no_input_form(self) -> None:
        """The specification puts the form on the detector page only."""
        content = self.client.get(reverse("detector:home")).content.decode()
        self.assertNotIn('id="analyze-form"', content)
        self.assertIn("Open the SMS Detector", content)

    def test_detector_page_contains_the_form_and_controls(self) -> None:
        content = self.client.get(reverse("detector:detector")).content.decode()
        self.assertIn('id="analyze-form"', content)
        self.assertIn('name="message"', content)
        self.assertIn('name="model"', content)
        self.assertIn("Analyze Message", content)
        self.assertIn('id="clear-button"', content)
        self.assertIn("csrfmiddlewaretoken", content)

    def test_unknown_url_returns_branded_404(self) -> None:
        response = self.client.get("/this-page-does-not-exist/")
        self.assertEqual(response.status_code, 404)
        self.assertIn("does not exist", response.content.decode())

    def test_pages_render_without_javascript_assets_on_disk(self) -> None:
        """The layout must not depend on a CDN: every asset is served locally."""
        content = self.client.get(reverse("detector:model_comparison")).content.decode()
        self.assertNotIn("cdn.jsdelivr.net", content)
        self.assertNotIn("unpkg.com", content)


class ModelComparisonPageTests(SimpleTestCase):
    """Requirement: comparison page behaviour with and without evaluation results."""

    def test_setup_message_when_results_are_missing(self) -> None:
        with temporary_model_directory() as paths:
            with use_model_directory(paths["models"], paths["metrics"]):
                response = self.client.get(reverse("detector:model_comparison"))
        content = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertIn("Evaluation results are not available yet", content)
        self.assertIn("python manage.py train_models", content)
        self.assertNotIn('id="metrics-chart"', content)

    def test_real_metrics_are_rendered_when_results_exist(self) -> None:
        with temporary_model_directory() as paths:
            result = train_all_models(
                artifacts_dir=paths["root"],
                models_dir=paths["models"],
                write_report=False,
            )
            with use_model_directory(paths["models"], paths["metrics"]):
                response = self.client.get(reverse("detector:model_comparison"))
        content = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="metrics-chart"', content)
        # Every model name and its measured accuracy must appear on the page.
        for model in result.metrics["models"].values():
            with self.subTest(model=model["key"]):
                self.assertIn(model["display_name"], content)
                self.assertIn(f"{model['metrics']['accuracy']:.4f}", content)
        # Chart data is injected server-side as JSON.
        self.assertIn('id="chart-data"', content)
        chart_payload = json.loads(
            content.split('<script id="chart-data" type="application/json">')[1].split("</script>")[0]
        )
        self.assertEqual(len(chart_payload["labels"]), 3)
        self.assertEqual(len(chart_payload["datasets"]), 4)
        for dataset in chart_payload["datasets"]:
            for value in dataset["data"]:
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 100)

    def test_table_values_are_not_hardcoded(self) -> None:
        """Rendering with no results must not leak numbers from a previous run."""
        with temporary_model_directory() as paths:
            with use_model_directory(paths["models"], paths["metrics"]):
                content = self.client.get(reverse("detector:model_comparison")).content.decode()
        self.assertNotIn("0.9913", content)
        self.assertNotIn("0.9647", content)


class DetectorFormTests(SimpleTestCase):
    """Requirement: prediction through the web form, plus input validation."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmp = temporary_model_directory()
        paths = cls._tmp.__enter__()
        cls.paths = paths
        train_all_models(
            artifacts_dir=paths["root"],
            models_dir=paths["models"],
            write_report=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.__exit__(None, None, None)
        super().tearDownClass()

    def setUp(self) -> None:
        self._context = use_model_directory(self.paths["models"], self.paths["metrics"])
        self._context.__enter__()

    def tearDown(self) -> None:
        self._context.__exit__(None, None, None)

    def test_get_renders_empty_state(self) -> None:
        content = self.client.get(reverse("detector:detector")).content.decode()
        self.assertIn("No analysis yet", content)
        self.assertNotIn("Predicted class", content)

    def test_submitting_spam_shows_the_spam_result(self) -> None:
        response = self.client.post(
            reverse("detector:detector"),
            {"message": SPAM_EXAMPLE, "model": "linear_svm"},
        )
        content = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertIn("Predicted class", content)
        self.assertIn("Spam", content)
        self.assertIn('data-prediction-label="spam"', content)
        self.assertIn("Linear Support Vector Machine", content)
        self.assertIn("not a probability", content)  # honest score explanation

    def test_submitting_ham_shows_the_ham_result(self) -> None:
        response = self.client.post(
            reverse("detector:detector"),
            {"message": HAM_EXAMPLE, "model": "naive_bayes"},
        )
        content = response.content.decode()
        self.assertIn('data-prediction-label="ham"', content)
        self.assertIn("Multinomial Naive Bayes", content)

    def test_every_model_can_be_selected(self) -> None:
        for key in ml_config.model_keys():
            with self.subTest(model=key):
                response = self.client.post(
                    reverse("detector:detector"), {"message": SPAM_EXAMPLE, "model": key}
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn(ml_config.MODEL_SPECS[key].display_name, response.content.decode())

    def test_empty_message_shows_a_helpful_error_and_no_prediction(self) -> None:
        response = self.client.post(reverse("detector:detector"), {"message": "", "model": "linear_svm"})
        content = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertIn("Please enter a message", content)
        self.assertNotIn("Predicted class", content)

    def test_overlong_message_is_rejected(self) -> None:
        response = self.client.post(
            reverse("detector:detector"),
            {"message": "x" * (ml_config.MAX_MESSAGE_LENGTH + 50), "model": "linear_svm"},
        )
        content = response.content.decode()
        self.assertIn("too long", content)
        self.assertNotIn("Predicted class", content)

    def test_unknown_model_value_is_rejected(self) -> None:
        response = self.client.post(
            reverse("detector:detector"), {"message": SPAM_EXAMPLE, "model": "not_a_model"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Please choose one of the available classifiers.", response.content.decode())

    def test_missing_model_field_falls_back_to_the_default(self) -> None:
        response = self.client.post(reverse("detector:detector"), {"message": SPAM_EXAMPLE})
        self.assertIn(
            ml_config.MODEL_SPECS[ml_config.DEFAULT_MODEL_KEY].display_name,
            response.content.decode(),
        )

    def test_user_text_is_escaped_in_the_response(self) -> None:
        payload = '<script>alert("xss")</script> free prize'
        response = self.client.post(
            reverse("detector:detector"), {"message": payload, "model": "naive_bayes"}
        )
        content = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("<script>alert", content)
        self.assertNotIn(payload, content)

    def test_setup_message_when_no_model_is_trained(self) -> None:
        with temporary_model_directory() as empty_paths:
            with use_model_directory(empty_paths["models"], empty_paths["metrics"]):
                get_content = self.client.get(reverse("detector:detector")).content.decode()
                post_response = self.client.post(
                    reverse("detector:detector"), {"message": SPAM_EXAMPLE, "model": "linear_svm"}
                )
        self.assertIn("No trained model found", get_content)
        self.assertIn("train_models", post_response.content.decode())


class AnalyzeApiTests(SimpleTestCase):
    """Requirement: the prediction endpoint used by the asynchronous UI."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmp = temporary_model_directory()
        paths = cls._tmp.__enter__()
        cls.paths = paths
        train_all_models(
            artifacts_dir=paths["root"],
            models_dir=paths["models"],
            model_keys=["logistic_regression"],
            write_report=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.__exit__(None, None, None)
        super().tearDownClass()

    def setUp(self) -> None:
        self._context = use_model_directory(self.paths["models"], self.paths["metrics"])
        self._context.__enter__()

    def tearDown(self) -> None:
        self._context.__exit__(None, None, None)

    def test_json_request_returns_a_prediction(self) -> None:
        response = self.client.post(
            reverse("detector:analyze_api"),
            data=json.dumps({"message": SPAM_EXAMPLE, "model": "logistic_regression"}),
            content_type="application/json",
        )
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["label"], "spam")
        self.assertEqual(payload["score_type"], "probability")
        self.assertLessEqual(payload["score"], 1.0)
        self.assertTrue(payload["score_display"].endswith("%"))

    def test_form_encoded_request_is_supported(self) -> None:
        response = self.client.post(
            reverse("detector:analyze_api"), {"message": HAM_EXAMPLE, "model": "logistic_regression"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["label"], "ham")

    def test_empty_message_returns_400_with_a_message(self) -> None:
        response = self.client.post(
            reverse("detector:analyze_api"), {"message": "  ", "model": "logistic_regression"}
        )
        payload = response.json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("message", payload["errors"])

    def test_malformed_json_returns_400(self) -> None:
        response = self.client.post(
            reverse("detector:analyze_api"), data="{not json", content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    def test_overlong_message_returns_400(self) -> None:
        response = self.client.post(
            reverse("detector:analyze_api"),
            {"message": "y" * (ml_config.MAX_MESSAGE_LENGTH + 1), "model": "logistic_regression"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("too long", json.dumps(response.json()["errors"]))

    def test_get_is_not_allowed(self) -> None:
        response = self.client.get(reverse("detector:analyze_api"))
        self.assertEqual(response.status_code, 405)

    def test_unavailable_model_returns_503_with_setup_hint(self) -> None:
        with temporary_model_directory() as empty_paths:
            with use_model_directory(empty_paths["models"], empty_paths["metrics"]):
                response = self.client.post(
                    reverse("detector:analyze_api"),
                    {"message": SPAM_EXAMPLE, "model": "linear_svm"},
                )
        payload = response.json()
        self.assertEqual(response.status_code, 503)
        self.assertTrue(payload["setup_required"])
        self.assertIn("train_models", payload["training_hint"])

    def test_csrf_protection_is_active(self) -> None:
        """A POST without a CSRF token must be rejected by Django."""
        client = self.client_class(enforce_csrf_checks=True)
        response = client.post(
            reverse("detector:analyze_api"), {"message": SPAM_EXAMPLE, "model": "logistic_regression"}
        )
        self.assertEqual(response.status_code, 403)
