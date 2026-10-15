# Engineering decisions

This document records **why** the project is built the way it is. It complements the README, which
explains *how* to run and use it. Anything marked *(specification)* is required by
`docs/BA-Project-2.pdf`; everything else is an engineering choice made here.

---

## 1. Scope and source of truth

| Decision | Reason |
| --- | --- |
| The provided file `SMSSpamCollection.txt` is the only dataset, used unchanged | *(specification)* The assignment supplies the corpus; replacing it would change the task |
| The three algorithms are exactly Naive Bayes, Logistic Regression and SVM | *(specification)* |
| Metrics are accuracy, precision, recall and F1-score | *(specification)* |
| No Dataset Analysis or Error Analysis page | *(specification)* explicitly excludes them; the statistics live in the training report |
| Nothing about real-time protection, URL scanning, e-mail integration or accounts appears in the UI | None of it is implemented, and claiming it would be dishonest |

The specification is in Persian and short; where it leaves a detail open (vectoriser parameters,
cleaning rules, interface design, deployment), this project takes a documented decision rather than
inventing a requirement.

## 2. Dataset handling

| Decision | Reason |
| --- | --- |
| Delimiter, header and encoding are **detected**, not assumed | The loader must also work if the professor's file is exported differently (`;`, CSV header, BOM, `latin-1`) |
| Exact duplicate rows are dropped (414 rows, counted and reported) | Duplicates would otherwise appear in both the train and the test split and inflate every score. `--keep-duplicates` exists to compare |
| Labels are normalised through an explicit alias map; unknown labels are dropped **and counted** | Silent coercion hides data problems |
| The raw text column is never modified on disk; cleaning is part of the pipeline | Keeps the corpus verifiable and reproducible |
| `test_size=0.2`, `random_state=42`, `stratify=y` | Reproducible and keeps the 12 % spam ratio in both parts |
| Messages longer than 1 600 characters are rejected | That is the practical upper bound of a concatenated SMS; the model was never trained on more |

## 3. Text cleaning

| Decision | Reason |
| --- | --- |
| URLs, e-mails, phone numbers, money amounts and other numbers become placeholder tokens (`__url__`, ...) instead of being deleted | "contains a premium-rate number" is a strong spam signal; deleting the entity destroys it |
| Placeholders use underscores | scikit-learn's default token pattern keeps `_`, so `__url__` survives as one feature |
| Unicode NFKC normalisation, lower-casing, non-ASCII removal | The corpus contains mojibake (`ú1.20`, `fine‚`) that would otherwise create duplicate vocabulary entries |
| **English stop words are *not* removed** | Spam/ham differences live in exactly the short function words and bigrams ("call now", "text to") that stop-word removal destroys; TF-IDF already down-weights uninformative terms |
| Cleaning is implemented as a scikit-learn transformer inside the pipeline, not as a pre-processing step | Training and serving can never drift apart |
| No stemming/lemmatisation (NLTK is not a dependency) | Feature quality is already good (~0.96 F1); extra dependencies and processing time are not justified for a course project |

## 4. Features and models

| Decision | Reason |
| --- | --- |
| `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.98, sublinear_tf=True)` | Bigrams capture call-to-action phrases; `min_df=2` removes one-off typos; `sublinear_tf` prevents very long messages from dominating |
| Identical vectoriser configuration for all three models | The comparison must be fair |
| `class_weight="balanced"` for Logistic Regression and LinearSVC | Spam is only ~13 % of the data; without balancing, recall on spam suffers |
| `MultinomialNB(alpha=0.1)` | Slight smoothing reduction suits TF-IDF weighted counts and measurably beats the default `alpha=1.0` here |
| `LinearSVC` rather than kernel `SVC` | Text data is high-dimensional and almost linearly separable; the linear variant is far faster and equally accurate. `SVC(probability=True)` was rejected (slow internal 5-fold Platt scaling) |
| No probability calibration for the SVM | The project shows the margin as a margin. Fabricating probabilities with a sigmoid would be worse than reporting what the model actually outputs |
| The vectoriser is fitted inside each pipeline on the training split only | *(specification: avoid data leakage)* A test asserts the vocabulary contains no n-gram that appears only in the test split |

## 5. Evaluation

| Decision | Reason |
| --- | --- |
| Positive class = **spam** (`pos_label=1`) | *(specification)* Precision/recall/F1 must describe spam detection |
| Accuracy, precision, recall, F1 are all reported | The dataset is imbalanced (~87 % ham): accuracy alone is misleading |
| Confusion matrices for every model | Shows *which kind* of error each model makes (false alarm vs. missed spam) |
| ROC-AUC added | Cheap, model-agnostic and informative; computed from `predict_proba`, or from the decision margin for the SVM (AUC only needs a ranking, which is stated on the page) |
| 5-fold stratified CV on the training split | Stability check; the test split remains untouched for the headline numbers |
| Metrics are written to `metrics.json` and rendered from there | *(requirement)* No number is typed into a template. With no file, the page shows setup instructions instead of fake charts |
| A Markdown training report is generated | Makes the dataset audit and the numbers easy to hand in |

## 6. Prediction service

| Decision | Reason |
| --- | --- |
| One `ModelRegistry` per process with a lock, lazily loading and caching pipelines | Predictions must not re-read a 150-300 kB joblib file from disk per request |
| Cache key = `(mtime, size)` of the artefact file | Retraining while the server runs is picked up automatically; no restart rituals |
| Documented limits: per-process memory, no cross-worker invalidation | Honest about the trade-off instead of pretending it is a model server |
| Loading steps are validated (`clean`, `tfidf`, `clf` must exist) and scikit-learn versions are compared | Produces a clear "retrain" error instead of `AttributeError` deep inside scikit-learn |
| Missing artefacts raise `ModelNotTrainedError`; corrupt artefacts raise `ModelLoadError` | The UI can differentiate "not set up yet" (setup instructions) from "something is broken" |
| Validation lives in `validate_message` and is reused by the form, the API and the service | One source of truth for the empty/too-long/non-text rules |
| Scores: probability for NB/LR, decision margin for the SVM, with an explanation next to it | *(requirement)* Never present a margin as a calibrated probability |
| Per-request latency is measured around the actual `predict` call | The number shown is real, not an estimate |

## 7. Web application architecture

| Decision | Reason |
| --- | --- |
| One Django project, one app, function-free class-based views, no DRF, no Celery, no Redis, no Postgres | *(requirement: avoid unnecessary complexity)* Everything the assignment needs is served by Django itself |
| The ML layer (`detector/ml/`) is framework-independent (settings mirrored from `detector/ml/config.py`) | It can be trained from a plain Python shell or imported by tests without Django models; the Django layer only presents results |
| No database models | The specification requires no stored data, and storing user SMS text creates a privacy liability for no benefit. Django's own tables (sessions, admin) still run through migrations - documented in `detector/models.py` |
| Training is an explicit management command | *(requirement)* Startup must never retrain; `apps.py` only logs whether artefacts exist |
| Artefacts are copied into a temporary directory and moved into place atomically | A failed run cannot leave a half-usable model set |
| Templates render results server-side; JavaScript swaps the server-rendered fragment | No divergent client-side rendering, no HTML duplication, escaping stays in Django. Without JS the plain form still works |
| CSRF protection stays on (including for the JSON endpoint, token in `X-CSRFToken`) | *(requirement)* The endpoint is same-origin only |
| User text is only ever rendered through Django's auto-escaping; a test posts an XSS payload | *(requirement)* |
| Errors are handled per case (setup / load / unexpected) with user-facing messages; no tracebacks leak | Predictable UX and no information disclosure |

## 8. Front end

| Decision | Reason |
| --- | --- |
| Tailwind CSS compiled to `assets/css/tailwind.css` and committed | *(requirement)* No CDN as the styling mechanism, no fragile build step for the grader: `python manage.py runserver` shows the finished design |
| Chart.js vendored into `assets/vendor/` | Charting library is needed; a CDN copy would break offline/restricted environments. The page degrades to a table if it is missing |
| Dark navy palette with blue/violet accents, system font stack | *(requirement suggestion)* Lightweight (no web-font download) and readable; contrast was chosen for accessibility |
| Responsive layout, mobile navigation, visible focus rings, `aria-live` on the result panel, skip link | *(requirement: responsive and accessible)* |
| The Home page has no input form | *(specification: the detector page owns submission)* |
| No invented capabilities in the copy | *(requirement: no real-time protection / URL scanning claims)* |

## 9. Deployment and configuration

| Decision | Reason |
| --- | --- |
| `python:3.12-slim` multi-stage image, dependencies in a venv, non-root user, healthcheck | *(requirement: production-appropriate base image)* Keep the runtime image small and least-privileged |
| `artifacts/` and `data/` are Compose volumes, `.dockerignore` excludes them from the image | *(requirement)* A model trained on the host is not silently assumed to exist in the container; the training command and the web container share the same volume |
| Training is documented as a separate Docker step | Same reason as the management command: explicit, cheap, reproducible |
| SQLite | *(suggested)* Sufficient for the assignment's data volume |
| Sensible defaults in `settings.py` + optional `.env` + `.env.example` | Works out of the box, configurable without editing code |
| Security switches (HSTS, secure cookies, SSL redirect, allowed hosts, secret key requirement) are opt-in and documented | *(requirement)* Development stays frictionless, production hardening is explicit |
| Tailwind's standalone binary documented as a Node-free alternative | The grader may not have Node.js installed |

## 10. Testing

| Decision | Reason |
| --- | --- |
| Tests train into a temporary artefacts directory and repoint the registry | They pass on a fresh clone, never touch the delivered artefacts, and cannot damage a real training run |
| Behavioural assertions (accuracy > 0.9, F1 > 0.8) instead of exact numbers | Exact numbers are environment-specific; the check is that the pipeline genuinely works |
| A dedicated no-leakage test | The most common silent mistake in student NLP projects |
| Chart payload is parsed from the response | Proves the page renders measured data rather than hardcoded values |
| Tests cover "metrics missing" and "artefact missing/corrupt" paths | These are exactly the situations a grader will hit on a fresh machine |

## 11. Known trade-offs (accepted, not hidden)

1. **Old, UK-centric corpus.** Accuracy on modern traffic will be lower than the test scores.
2. **Per-process model cache.** Each WSGI worker loads its own copy; fine at this size, not a
   shared model service.
3. **No probability calibration for the SVM.** Deliberate: the margin is reported as a margin.
4. **No rate limiting or caching layer for the endpoint.** Out of scope; noted in the README.
5. **`max_features` is unlimited**, so the vocabulary (~9 956 features) depends on `min_df`/`max_df`
   rather than a hard cap. This keeps every model trained on exactly the same feature space.
6. **Single test split** (80/20) rather than nested cross-validation for model selection. The
   specification asks for train/test training and comparison; the CV scores are reported as a
   stability check instead.
