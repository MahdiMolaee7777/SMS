"""URL configuration of the detector app (namespace: ``detector``)."""

from django.urls import path

from . import views

app_name = "detector"

urlpatterns = [
    path("", views.HomeView.as_view(), name="home"),
    path("detector/", views.DetectorView.as_view(), name="detector"),
    path("detector/analyze/", views.AnalyzeApiView.as_view(), name="analyze_api"),
    path("model-comparison/", views.ModelComparisonView.as_view(), name="model_comparison"),
    path("about/", views.AboutView.as_view(), name="about"),
]
