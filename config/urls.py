"""Root URL configuration for the SMS Spam Detector project."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("detector.urls", namespace="detector")),
]

# Branding for the bundled Django admin (optional internal tool, not part of the
# public pages).
admin.site.site_header = "SMS Spam Detector - Administration"
admin.site.site_title = "SMS Spam Detector"
admin.site.index_title = "Project administration"

# Branded error pages (see detector/views.py).
handler404 = "detector.views.page_not_found"
handler500 = "detector.views.server_error"
