# Delivery checklist — what was actually executed

All checks below were run in the delivery environment (Linux, Python 3.13, CPU only, no GPU).
Nothing in this list is an assumption: each line states the command and its observed outcome.

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 1 | Django system checks | `python manage.py check` | ✅ `System check identified no issues (0 silenced)` |
| 2 | Database migrations | `python manage.py migrate` | ✅ Django's own tables created (the app has no models by design) |
| 3 | Dataset loading + audit | `python manage.py train_models` | ✅ 5 574 lines → 5 160 unique messages (4 518 ham / 642 spam), 414 duplicates removed, 0 missing values |
| 4 | Training all three models | `python manage.py train_models` | ✅ Naive Bayes (0.9845 acc / 0.9355 F1), Logistic Regression (0.9893 / 0.9569), Linear SVM (0.9913 / 0.9644); 4.4 s total |
| 5 | Artefacts written | `ls artifacts artifacts/models` | ✅ 3 × `.joblib` + 3 × `.meta.json`, `metrics.json`, `training_report.md` |
| 6 | Metrics reproducible from artefacts | covered by `test_metrics_are_identical_when_recomputed_from_artifacts` | ✅ stored metrics equal re-measured accuracy |
| 7 | No data leakage | `test_vectorizer_is_fitted_on_training_data_only` | ✅ vocabulary ⊆ training n-grams; test-only n-grams absent |
| 8 | Test suite | `python manage.py test` | ✅ **72 tests, all pass** (21 s) |
| 9 | Pages over HTTP (dev server) | `runserver` + GET `/`, `/detector/`, `/model-comparison/`, `/about/` | ✅ HTTP 200 for all four |
| 10 | Prediction through the form | POST `/detector/` with a spam message | ✅ HTTP 200, panel shows `data-prediction-label="spam"`, model name, score explanation, latency |
| 11 | Prediction through the JSON API | POST `/detector/analyze/` (JSON, CSRF token) | ✅ `{"label": "spam", "score_display": "99.7%", "score_type": "probability", ...}` |
| 12 | CSRF protection | POST `/detector/analyze/` without token | ✅ HTTP 403 (and covered by a test with `enforce_csrf_checks=True`) |
| 13 | Input validation | POST with empty / oversized / invalid model | ✅ user-facing error messages, no prediction shown |
| 14 | Production static files | `DJANGO_DEBUG=0 manage.py collectstatic` | ✅ 135 files collected and compressed (WhiteNoise) |
| 15 | Production server | `gunicorn config.wsgi:application` (2 workers) | ✅ all pages 200, static CSS served, API prediction OK |
| 16 | Tailwind build | `tailwindcss -c tailwind.config.js -i assets/css/input.css -o assets/css/tailwind.css --minify` | ✅ compiled locally (≈ 38 kB, committed); components, custom palette, gradients and arbitrary grids verified in the output |
| 17 | Chart.js vendored | `assets/vendor/chart.umd.min.js` | ✅ present (206 kB); page falls back to an HTML table if absent |
| 18 | Model Comparison without training | test `test_setup_message_when_results_are_missing` + manual check | ✅ setup instructions, no fake charts |
| 19 | Docker Compose file | YAML parsed and inspected | ✅ valid YAML, health check, volumes, env defaults; **not executed** (no Docker daemon in the delivery sandbox) |
| 20 | Dockerfile | static review; `sh -n docker/entrypoint.sh` | ✅ valid shell syntax, executable mode 755, referenced files exist; **image build not executed** (no Docker daemon) |
| 21 | Makefile targets | reviewed against the actual file layout | ✅ command paths verified; targets not run end-to-end (they call the same commands verified above) |
| 22 | Git repository prepared | `git init -b main`, feature branch, `git add -A`, commit | ✅ `main` = placeholder base, `feat/sms-spam-detector` = 74 files / 13 767 insertions committed; working tree clean (`git status --porcelain` empty) |
| 23 | Push to GitHub | `git push` | ❌ **not possible from the build environment**: no GitHub app is connected to the conversation (`list_connector_tools("github")` returns *unsupported*), no repository remote exists, and the sandbox holds no token or SSH key (checked `env`, `~/.git-credentials`, `~/.netrc`, `~/.ssh`). GitHub is reachable anonymously only |
| 24 | Downloadable archive | `zip -r sms-spam-detector.zip sms-spam-detector` | ✅ 2.7 MB, 308 entries, includes the `.git` directory with both branches; verified by extracting to a clean directory: `git log --all` shows both commits and the app classified a message correctly |

## Not executed, with reasons

* **`docker compose build` / `docker compose up`** — the delivery sandbox has no Docker daemon.
  The compose file and Dockerfile were reviewed line by line (paths, volumes, permissions,
  healthcheck, environment variables) and the equivalent commands were executed directly
  (`migrate`, `collectstatic`, `gunicorn`, `runserver`) so the application logic itself is verified.
* **`npm install && npm run css:build`** — the compiled CSS is already committed, so the Node
  toolchain is not needed to run the project. The Tailwind CLI (standalone binary, same
  version 3.4.17) was used to produce `assets/css/tailwind.css`, and that build was verified.
* **Browser rendering / screenshots** — HTML, CSS and JavaScript were verified by fetching the
  pages and inspecting the markup and the compiled stylesheet; no headless browser was available
  to take screenshots.

## Exact commands to launch the application

**Docker (recommended for a clean machine)**

```bash
cd sms-spam-detector
cp .env.example .env                        # optional
docker compose build
docker compose run --rm web python manage.py train_models
docker compose up
# open http://localhost:8000
```

**Local Python**

```bash
cd sms-spam-detector
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py train_models
python manage.py runserver
# open http://127.0.0.1:8000
```

**Tests**

```bash
python manage.py test                      # local
docker compose run --rm web python manage.py test   # in the container
```
