# SMS Spam Detector

A complete, runnable university machine-learning project: a **Django web application** that
classifies an English SMS message as **Ham** (normal) or **Spam** (unwanted, promotional or
fraudulent), using three real scikit-learn models trained on the provided *SMS Spam Collection*
dataset.

Nothing in this project is simulated: the training command fits actual TF-IDF pipelines, the
evaluation numbers are measured on a held-out test split, and the web pages display those stored
results. There are no hardcoded predictions, no keyword rules and no fabricated metrics.

---

## Table of contents

1. [What you get](#1-what-you-get)
2. [Technology stack](#2-technology-stack)
3. [Project structure](#3-project-structure)
4. [Dataset](#4-dataset)
5. [Quick start with Docker](#5-quick-start-with-docker)
6. [Quick start with local Python](#6-quick-start-with-local-python)
7. [Training the models](#7-training-the-models)
8. [Using the website](#8-using-the-website)
9. [How the machine learning works](#9-how-the-machine-learning-works)
10. [Evaluation metrics explained](#10-evaluation-metrics-explained)
11. [Where the artefacts live](#11-where-the-artefacts-live)
12. [Configuration](#12-configuration)
13. [Styling and front-end assets](#13-styling-and-front-end-assets)
14. [Tests](#14-tests)
15. [Troubleshooting](#15-troubleshooting)
16. [Limitations and responsible use](#16-limitations-and-responsible-use)
17. [Production notes](#17-production-notes)

---

## 1. What you get

**Pages**

| URL | Page | Purpose |
| --- | --- | --- |
| `/` | Home | What the tool does, how the pipeline works, the three algorithms |
| `/detector/` | SMS Detector | Paste a message, pick a classifier, get Ham/Spam |
| `/model-comparison/` | Model Comparison | Accuracy, precision, recall, F1, bar chart, confusion matrices |
| `/about/` | About Project | Objective, dataset, TF-IDF, algorithms, stack, limits |
| `/admin/` | Django admin | Framework administration (optional, not part of the assignment) |

**Machine learning**

* Multinomial Naive Bayes, Logistic Regression and a Linear SVM (`LinearSVC`).
* TF-IDF features (unigrams + bigrams) built **inside** a scikit-learn `Pipeline`, so the
  vectorizer is fitted on the training split only - no data leakage.
* Metrics for all three models: accuracy, precision, recall, F1 (spam = positive class), ROC-AUC,
  confusion matrix and inference latency.
* 5-fold stratified cross-validation on the training split as a stability check.
* Models persisted with `joblib`, loaded once per process and cached.

**Engineering**

* Django 5.2, server-rendered templates, CSRF protection, form validation on the server and in the
  ML service layer.
* Tailwind CSS compiled to a local file, Chart.js vendored locally - the site needs **no CDN** and
  keeps working offline.
* Docker + Docker Compose with a non-root user, health check and pinned dependencies.
* 70+ automated tests covering the ML layer and every page/endpoint.
* Explicit training command (`python manage.py train_models`); the app never retrains on startup.

---

## 2. Technology stack

| Layer | Choice | Version (pinned in `requirements.txt`) |
| --- | --- | --- |
| Language | Python | 3.12 (Docker) / 3.11+ supported |
| Web framework | Django | 5.2.18 |
| ML library | scikit-learn | 1.6.1 |
| Features | `TfidfVectorizer` (1-2 grams) | - |
| Data | pandas, NumPy | 2.2.3 / 2.3.5 |
| Persistence | joblib | 1.5.3 |
| Database | SQLite (Django's own tables only) | bundled |
| Frontend | Django templates, Tailwind CSS 3.4.17, vanilla JS | compiled locally |
| Charts | Chart.js | 4.4.7 (vendored) |
| Static files (prod) | WhiteNoise | 6.12.0 |
| App server (prod) | Gunicorn | 26.2.0 |
| Containers | Docker, Docker Compose | any recent version |

CPU only. No GPU, no paid API, no external AI service, no Redis/Celery/Kubernetes.

---

## 3. Project structure

```
sms-spam-detector/
├── manage.py                     # Django entry point
├── config/                       # Project configuration
│   ├── settings.py               # Settings driven by environment variables
│   ├── urls.py                   # Root URLconf + error handlers
│   ├── wsgi.py / asgi.py
├── detector/                     # The single Django application
│   ├── apps.py                   # Startup check: are the models trained?
│   ├── forms.py                  # AnalyzeMessageForm (server-side validation)
│   ├── views.py                  # Home / Detector / Analyze API / Comparison / About
│   ├── urls.py                   # App URLconf (namespace "detector")
│   ├── context_processors.py     # Project name, version, message limit
│   ├── models.py                 # Deliberately empty (documented in the file)
│   ├── ml/                       # Framework-independent machine-learning layer
│   │   ├── config.py             # Model specs, TF-IDF params, paths, labels
│   │   ├── cleaning.py           # TextCleaner (step 2 of the specification)
│   │   ├── data.py               # Loading, auditing, normalising, splitting
│   │   ├── evaluation.py         # Metrics, confusion matrix, feature introspection
│   │   ├── training.py           # Orchestrates training + artefact publishing
│   │   └── prediction.py         # ModelRegistry, PredictionService, validation
│   ├── management/commands/
│   │   ├── train_models.py       # `python manage.py train_models`
│   │   └── model_status.py       # `python manage.py model_status`
│   ├── templates/detector/       # home.html, detector.html, _result.html, ...
│   └── tests/                    # data / models / forms / views test modules
├── templates/                    # base.html, 404.html, 500.html
├── assets/                       # Source assets served as Django static files
│   ├── css/input.css             # Tailwind source (compiled -> tailwind.css)
│   ├── css/tailwind.css          # Compiled stylesheet (committed)
│   ├── js/app.js, detector.js, charts.js
│   ├── img/favicon.svg
│   └── vendor/chart.umd.min.js   # Chart.js, vendored (no CDN)
├── data/raw/SMSSpamCollection.txt  # Provided dataset (read-only input)
├── artifacts/                    # Generated: models + metrics (see §11)
├── docs/BA-Project-2.pdf         # The professor's specification (reference copy)
├── docs/engineering-decisions.md # Why things are built the way they are
├── docker/entrypoint.sh
├── Dockerfile / docker-compose.yml / .dockerignore
├── requirements.txt / requirements-dev.txt
├── tailwind.config.js / package.json / Makefile
├── .env.example / .gitignore
└── README.md
```

---

## 4. Dataset

**Source:** the *SMS Spam Collection* file supplied with the assignment
(`data/raw/SMSSpamCollection.txt`, copied unchanged from the provided material).

**Format:** plain text, one message per line, two tab-separated columns:

```
ham     Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam    Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 ...
```

The loader does **not** assume this shape blindly - it detects the delimiter (`\t`, `;`, `,`) and
whether a header row exists, and it reports an error instead of guessing when the file is missing.

**What the audit reports** (numbers from the delivered run, reproduced by
`python manage.py train_models`):

| Property | Value |
| --- | --- |
| Lines in file | 5 574 |
| Unique messages used | 5 160 |
| Ham / Spam | 4 518 / 642 |
| Spam ratio | ≈ 12.4 % (imbalanced - accuracy alone is misleading) |
| Exact duplicate rows removed | 414 |
| Missing messages / labels | 0 / 0 |
| Same text with conflicting labels | reported in `metrics.json` |

**Decisions taken (and why):**

* **Duplicates are dropped.** 414 rows are exact repeats of another message. Keeping them would put
  identical texts into both the training and the test split, inflating the scores. The count is
  printed by the training command and stored in `metrics.json`. Use
  `--keep-duplicates` to compare both behaviours.
* **Labels are normalised** (`ham`, `0`, `normal` → `ham`; `spam`, `junk`, `1` → `spam`). Anything
  unrecognised is dropped *and counted* rather than silently coerced.
* **The text itself is never rewritten in the dataset file.** Cleaning happens inside the model
  pipeline, so the raw corpus stays reproducible.
* **Split:** `train_test_split(test_size=0.2, random_state=42, stratify=y)` → 4 128 training
  messages / 1 032 test messages, with the spam ratio preserved in both parts.

Newer public versions of this corpus exist, but the assignment provides this file, so this file is
used - unchanged.

---

## 5. Quick start with Docker

Requirements: Docker with the Compose plugin. No Python, Node or GPU needed on the host.

```bash
# 1. enter the project directory
cd sms-spam-detector

# 2. (optional) create your local configuration
cp .env.example .env          # edit DJANGO_SECRET_KEY etc. if you like

# 3. build the image
docker compose build

# 4. train the three models ONCE (writes into ./artifacts on your host)
docker compose run --rm web python manage.py train_models

# 5. database migrations run automatically at container start; to be explicit:
docker compose run --rm web python manage.py migrate

# 6. start the application
docker compose up

# 7. open http://localhost:8000
```

Shortcut for steps 3-6:

```bash
make docker-run        # build + train + start (see Makefile)
```

Useful commands:

```bash
docker compose logs -f web                     # follow the log
docker compose run --rm web python manage.py model_status   # artefact health check
docker compose run --rm web python manage.py test           # run the test suite in the container
docker compose down                            # stop
```

**Why training is a separate step:** training takes a few seconds and must be *explicit* - the web
container never trains on startup, so a bad dataset or a broken environment cannot turn into a
silently slow first request. The `artifacts` directory is a volume shared by the one-off training
container and the web container, so models trained in step 4 are immediately visible in step 6
(restart the web container if it was already running).

---

## 6. Quick start with local Python

Python 3.11+ (3.12 recommended). No Node.js needed for running - the compiled Tailwind CSS is
already committed.

```bash
cd sms-spam-detector

# 1. virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. dependencies
pip install --upgrade pip
pip install -r requirements.txt      # add -r requirements-dev.txt for coverage

# 3. configuration (optional, sensible defaults are built in)
cp .env.example .env

# 4. database
python manage.py migrate

# 5. train the models once
python manage.py train_models

# 6. run the development server
python manage.py runserver           # http://127.0.0.1:8000
```

Or simply:

```bash
make setup     # venv + dependencies + migrate + train
make run
```

---

## 7. Training the models

```bash
python manage.py train_models
```

The command performs the specification's steps in order and prints progress:

```
Step 1/6  Loading and auditing the dataset ...        (rows, classes, duplicates)
Step 2/6  Cleaning text and building TF-IDF ...       (train/test split summary)
Step 3/6  Training Multinomial Naive Bayes ...        (metrics per model)
Step 4/6  Training Logistic Regression ...
Step 5/6  Training Linear Support Vector Machine ...
Step 6/6  Saved 3 models, metrics and report ...
```

Useful options:

| Option | Effect |
| --- | --- |
| `--model naive_bayes` | Train a single algorithm (repeatable) |
| `--dataset path/to/file` | Use another copy of the corpus |
| `--keep-duplicates` | Keep exact duplicates (for comparison in the report) |
| `--no-report` | Skip `training_report.md` |
| `--quiet` | Print only the final table |

**Verified result of the delivered run** (test split of 1 032 messages, positive class = spam):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | ms / message |
| --- | --- | --- | --- | --- | --- | --- |
| Multinomial Naive Bayes | 0.9845 | 0.9667 | 0.9062 | 0.9355 | 0.9917 | 0.040 |
| Logistic Regression | 0.9893 | 0.9606 | 0.9531 | 0.9569 | 0.9980 | 0.039 |
| Linear SVM | 0.9913 | 0.9760 | 0.9531 | 0.9644 | 0.9972 | 0.038 |

5-fold cross-validated F1 on the training split: 0.9444 (Naive Bayes), 0.9552 (Logistic
Regression), 0.9578 (Linear SVM) - the ranking is stable between the two protocols.

(These are the values written to `artifacts/metrics.json` by the training command in this
delivery; the Model Comparison page reads that file, so it always shows the real, current numbers.
`python manage.py train_models` reproduces them exactly - same seed, same split.)

### Why the models are not retrained automatically

Retraining on every request (or every startup) would be slow, would make results depend on server
load and could race between workers. Instead:

* training happens once, explicitly;
* `artifacts/models/*.joblib` are loaded **lazily** by `ModelRegistry` and cached per process;
* the cache is invalidated automatically when the artefact file changes (`mtime` + size), so
  retraining while the server runs is picked up on the next prediction;
* every artefact records the scikit-learn version, the feature count, the hyper-parameters and the
  dataset hash, so incompatibilities can be diagnosed (`python manage.py model_status`).

The cache lives in the process. With several WSGI workers each worker holds its own copy of the
models: acceptable here (each pipeline is under 300 kB) but it is the reason the project does not
claim a shared "model server" architecture.

---

## 8. Using the website

### Home (`/`)

Explains the tool, shows live dataset statistics when models have been trained, describes the
four-step pipeline and the three algorithms, and includes a capabilities/limits panel. It
deliberately has **no input form** - the specification puts the classifier on its own page.

### SMS Detector (`/detector/`)

1. Paste or type an English SMS into the text area (a character counter shows the 1 600-character
   limit; empty input is rejected before the model runs).
2. Pick the classifier: Naive Bayes, Logistic Regression or Linear SVM. The default is the model
   with the best F1-score on the test split (`linear_svm` here).
3. Press **Analyze Message**. The result panel shows:
   * the predicted class **Spam** or **Ham** in a red/green panel,
   * the model's name,
   * the inference time measured for that request,
   * the model score with an honest explanation: an estimated spam probability for the
     probabilistic models, or a signed decision margin for the SVM - never a fake percentage,
   * an optional breakdown of the n-grams that influenced that single prediction.
4. **Clear** resets the input and the panel. **Insert an example** fills the box with a sample
   message (the classification still comes from the model).

The form works without JavaScript (normal Django POST). With JavaScript enabled the same view is
called through `fetch`, showing a loading state and swapping in the server-rendered result, so the
markup and escaping rules stay in one place.

A JSON endpoint is also available for scripting:

```bash
curl -s -X POST http://localhost:8000/detector/analyze/ \
  -H "Content-Type: application/json" \
  -H "X-CSRFToken: <token from a browser cookie>" \
  -d '{"message": "You have won a prize, call 09061701461", "model": "linear_svm"}'
```

It returns the same structured data the page shows (label, model, score, score type, latency).

### Model Comparison (`/model-comparison/`)

A table of the measured metrics for the three models, a grouped bar chart, the confusion matrices,
the strongest learned n-grams and a description of the evaluation protocol. Everything is read
from `artifacts/metrics.json`. **If training has not been run, the page shows setup instructions
instead of empty charts.**

### About Project (`/about/`)

Objective, dataset description, a short explanation of TF-IDF, the three algorithms, the technology
stack and an explicit statement that this is an educational classifier that can be wrong.

---

## 9. How the machine learning works

### Step 1 - Loading and auditing (`detector/ml/data.py`)

The corpus is read, labels normalised, missing values and duplicates counted, and the class
distribution reported. Nothing is deleted silently.

### Step 2 - Cleaning (`detector/ml/cleaning.py`)

Text messages contain noise that confuses a bag-of-words model. Cleaning is deliberately light and
*information-preserving*:

| Input pattern | Replaced by | Why |
| --- | --- | --- |
| `http://x.y`, `www.x.y`, bare domains | `__url__` | "contains a URL" is a strong spam signal; the exact URL is not |
| `someone@example.com` | `__email__` | same idea for addresses |
| `09061701461`, `+44 800 123` | `__phone__` | premium-rate numbers are a spam marker |
| `£1.50`, `500 pounds`, `150p` | `__money__` | amounts vary, the *presence* of money matters |
| other numbers | `__num__` | keeps the pattern "text 87121 to ..." visible |
| non-ASCII leftovers, emoji | removed | avoids hundreds of one-off vocabulary entries |

Unicode is normalised (NFKC), everything is lower-cased and whitespace is collapsed. The result is
stored as a transformer **inside the pipeline**, so training and prediction apply exactly the same
transformation.

> **Deliberate decision:** English stop words are *not* removed. The differences between ham and
> spam live precisely in short function words and call-to-action bigrams ("call now", "text to",
> "claim your"), which stop-word removal would destroy. This is documented in `detector/ml/config.py`.

### Step 3 - Vectorisation (`TfidfVectorizer`)

Inside every pipeline: `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)`
→ 9 956 features. See §10.

### Step 4 - Splitting

`train_test_split` with `stratify=y` and `random_state=42`. The vectorizer is fitted on
`X_train` only; a test in the suite verifies that no n-gram appearing only in the test split is
present in the fitted vocabulary.

### Step 5 - Training the three algorithms

| Model | Implementation | Configuration | Score reported |
| --- | --- | --- | --- |
| Multinomial Naive Bayes | `MultinomialNB` | `alpha=0.1` | estimated probability (`predict_proba`) |
| Logistic Regression | `LogisticRegression` | `C=10`, `liblinear`, `class_weight="balanced"` | estimated probability (`predict_proba`) |
| Linear SVM | `LinearSVC` | `C=1.0`, `squared_hinge`, `class_weight="balanced"` | decision margin (not a probability) |

`class_weight="balanced"` compensates for the 87/13 ham-spam imbalance for the two models where it
matters; Naive Bayes handles the class prior internally via `fit_prior=True`.

### Step 6 - Evaluation

All three models are evaluated on the same untouched test split with the same metric set (see §10),
and 5-fold stratified cross-validation is computed on the training split only.

---

## 10. Evaluation metrics explained

Labels are encoded as `ham = 0`, `spam = 1`, therefore **spam is the positive class** in every
precision/recall/F1 calculation (`pos_label=1`). This matches the project specification, which
evaluates the detection of spam.

| Metric | Meaning | Watch out for |
| --- | --- | --- |
| **Accuracy** | share of all messages classified correctly | misleading with imbalanced data: a model that always says "ham" already scores ≈ 87.6 % |
| **Precision (spam)** | of the messages flagged as spam, how many really were spam | low precision = legitimate messages get blocked (false positives) |
| **Recall (spam)** | of all real spam messages, how many were caught | low recall = spam reaches the user (false negatives) |
| **F1 (spam)** | harmonic mean of precision and recall | the single number used to rank the models here |
| **ROC-AUC** | how well the model *ranks* spam above ham | computed from `predict_proba` or, for the SVM, from the decision margin (`roc_auc_score` only needs a ranking, not a probability) |
| **Confusion matrix** | counts of true/false positives and negatives | shows *which kind* of error each model makes |

**Reading the confusion matrix** (rows = truth, columns = prediction):

```
                     predicted Ham    predicted Spam
actual Ham            TN                 FP
actual Spam           FN                 TP
```

For the delivered Linear SVM: `TN = 901, FP = 3, FN = 6, TP = 122` → 3 false alarms and 6 missed
spam messages out of 1 032 test messages (Logistic Regression: 5 FP / 6 FN, Naive Bayes: 4 FP /
12 FN).

**Interactive inference speed** is measured per request (`latency_ms`) and shown on the detector
page; the table on the comparison page shows the average per-message time measured over the whole
test split.

---

## 11. Where the artefacts live

```
artifacts/
├── metrics.json                    # machine-readable evaluation results (all models)
├── training_report.md              # human-readable report (dataset, split, metrics, limits)
└── models/
    ├── naive_bayes.joblib          # full pipeline: cleaning -> TF-IDF -> classifier
    ├── naive_bayes.meta.json       # versions, hyper-parameters, metrics, dataset hash
    ├── logistic_regression.joblib
    ├── logistic_regression.meta.json
    ├── linear_svm.joblib
    └── linear_svm.meta.json
```

* A `.joblib` file contains the **whole fitted pipeline**, so the exact cleaning rules and the exact
  vectorizer are used at prediction time. There is no way to load a model with a mismatched
  vectorizer.
* `metrics.json` is the only source of truth for the Model Comparison page; templates contain no
  numbers.
* Artefacts are written into a temporary directory and atomically moved into place, so a crashed
  training run cannot leave a half-written model behind.
* **In this delivered repository the trained artefacts are committed** (≈ 570 kB in total), so a
  fresh clone of the branch runs immediately - the comparison page shows the real metrics and the
  detector answers without any further step. They are *generated* data and are regenerated by
  `python manage.py train_models`; if you prefer a source-only repository, delete the artefacts and
  re-enable the ignore rules that are already present (commented) in `.gitignore`.
* Retrain any time. Interpreting or moving model files manually is never necessary; if a file is
  missing or corrupted, the app shows a setup message and `python manage.py model_status` explains
  what is wrong.

---

## 12. Configuration

Everything is driven by environment variables (see `.env.example`). Copy it to `.env` - the file is
git-ignored and loaded automatically by `config/settings.py`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DJANGO_DEBUG` | `1` | Development mode. Set `0` for production behaviour |
| `DJANGO_SECRET_KEY` | dev fallback | **Required** when `DJANGO_DEBUG=0` (startup fails otherwise) |
| `DJANGO_ALLOWED_HOSTS` | `*` in debug | Comma-separated hostnames; required in production |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | empty | Required when served over HTTPS behind a proxy |
| `DJANGO_X_FRAME_OPTIONS` | empty in debug, `DENY` otherwise | Clickjacking protection |
| `DJANGO_LOG_LEVEL` | `INFO` | Log verbosity |
| `DJANGO_DB_PATH` | `data/db.sqlite3` | SQLite location |
| `MODEL_ARTIFACTS_DIR` | `artifacts` | Where models/metrics are read and written |
| `DATA_DIR` | `data/raw` | Where the dataset file is looked up |
| `DETECTOR_DEFAULT_MODEL` | `linear_svm` | Pre-selected classifier |
| `DETECTOR_MAX_MESSAGE_LENGTH` | `1600` | Upper bound for a submitted message |
| `DJANGO_SKIP_STARTUP_CHECK` | `0` | Skip the "models trained?" log line |

---

## 13. Styling and front-end assets

* **Tailwind CSS is compiled locally into `assets/css/tailwind.css`** (≈ 38 kB) and committed.
  Django serves it like any other static file. There is **no CDN dependency**: the design works
  offline, in a sandboxed preview and behind a strict CSP.
* Source: `assets/css/input.css` (design tokens, component classes) plus the utility classes used in
  the templates. Config: `tailwind.config.js` (dark navy palette, custom shadows).
* Rebuild after changing templates or styles:

  ```bash
  npm install                 # once: installs tailwindcss 3.4.17 + chart.js 4.4.7 locally
  npm run css:build           # or: make css
  npm run css:watch           # rebuild on change while developing
  ```

  Without Node.js you can use the standalone Tailwind binary:

  ```bash
  ./tailwindcss -c tailwind.config.js -i assets/css/input.css -o assets/css/tailwind.css --minify
  ```

* **Chart.js is vendored** into `assets/vendor/chart.umd.min.js` (`npm run vendor:chart`). If it is
  ever unavailable, `assets/js/charts.js` detects that and the page falls back to the HTML table
  that contains the same measured values - the interface stays usable.
* JavaScript is limited to progressive enhancements: mobile navigation, character counter, sample
  buttons, asynchronous form submission, and the chart. Every page works with JS disabled.

---

## 14. Tests

```bash
python manage.py test                   # 72 tests
python manage.py test detector.tests.test_models -v 2      # one module
coverage run --source=detector,config manage.py test && coverage report -m   # with coverage
```

What is covered:

| Area | Examples |
| --- | --- |
| Dataset loading | real file loads, row counts consistent, duplicate handling, non-existent file raises a clear error, `;`-delimited file with header is detected, unknown labels dropped and counted |
| Label normalisation | `ham/HAM/" ham "/0/normal` → `ham`, `spam/junk/1/spam.` → `spam`, unknown → rejected |
| Splitting | reproducible, stratified, disjoint, class ratios preserved, spam = positive class |
| Training | all three models train, artefacts + meta files written, metrics in range and clearly better than chance, metrics reproducible from the saved models, cross-validation recorded |
| Leakage | vocabulary ⊆ training n-grams; n-grams unique to the test split are absent |
| Prediction | spam and ham classified correctly by every model, deterministic output, explanation list valid, latency measured |
| Score semantics | probabilities stay within 0-1 and render as %, SVM margins never render as %, `format_score` helpers |
| Input validation | empty/whitespace/`None`/non-text/invisible input rejected, length limit enforced with a helpful message, unknown model rejected |
| Artefact errors | missing artefact → clear "run train_models" error, corrupted artefact → `ModelLoadError`, cache reuse and clearing, status list |
| Pages | all four pages return 200, navigation links resolve, form present on the detector page and absent from Home, branded 404, no CDN references |
| Comparison page | setup instructions when `metrics.json` is missing, real metrics/model names rendered when present, chart JSON well-formed, nothing hardcoded |
| Form endpoint | spam and ham submissions both render the right panel, every model selectable, empty/too-long/invalid model rejected, XSS payload escaped |
| JSON API | JSON and form-encoded posts, 400 on empty/malformed/too long, 405 on GET, 503 with setup hint when untrained, CSRF enforced |

The tests **train into a temporary directory** and repoint the model registry, so they neither need
pre-existing artefacts nor overwrite the ones the application uses. No test needs the internet.

---

## 15. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| The detector shows *"No trained model found"* / the comparison page asks for training | `artifacts/` is empty (fresh clone) | `python manage.py train_models` (Docker: `docker compose run --rm web python manage.py train_models`) |
| `Dataset file not found: .../SMSSpamCollection.txt` | dataset not in `data/raw` | put the provided file there or set `DATA_DIR` |
| `Could not load the model ... incompatible` | artefacts built with another scikit-learn version | retrain (`python manage.py train_models`); check versions with `python manage.py model_status` |
| `DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is disabled` | production mode without a key | generate one (see `.env.example`) and set it |
| `DisallowedHost` in production | `DJANGO_ALLOWED_HOSTS` empty | list the real hostnames |
| `CSRF verification failed` behind a proxy | origin not trusted | set `DJANGO_CSRF_TRUSTED_ORIGINS=https://your-host` |
| Port 8000 already in use | another service | `python manage.py runserver 8080` or set `WEB_PORT=8080` for Compose |
| Container starts but pages show *"train_models"* | models were trained before the volume existed | run the Docker training command again (artefacts persist in `./artifacts`) |
| Chart does not appear | `assets/vendor/chart.umd.min.js` missing | `npm install && npm run vendor:chart`; the table still works meanwhile |
| Styles look unstyled after editing a template | new Tailwind classes not compiled | `npm run css:build` |
| `PermissionError: /app/artifacts/...` in Docker | the mounted host directory is not writable by the container user (uid 1000) | `sudo chown -R 1000:1000 artifacts data`, or remove the bind mounts so Docker named/anonymous volumes are used |

---

## 16. Limitations and responsible use

* **Educational model, not a security product.** Both error types happen: legitimate messages get
  flagged as spam, and well-crafted spam gets through. Never delete or block a message based only
  on this tool.
* **Old corpus.** The SMS Spam Collection dates from around 2011 and is dominated by UK English.
  Modern wording, modern scam patterns, other languages, emoji-heavy text and non-UK phone/currency
  formats are underrepresented, so real-world accuracy will be lower than the test scores.
* **Text only.** Sender reputation, sending frequency, URL reputation and delivery metadata are not
  available to the model.
* **Scores are estimates.** Naive Bayes and Logistic Regression report model probabilities (not
  calibrated guarantees); the Linear SVM reports a decision margin and *no* probability is claimed
  for it. A large margin does not mean "definitely safe/insecure".
* **English only.** Non-English messages are out of scope; the dataset contains none.
* **Single-message classification.** There is no batch endpoint, no history and no user accounts.
* **SQLite.** Fine for a course project and single-user demos; concurrent writes are limited.
* **No rate limiting.** Deliberately out of scope, which matters if this were ever exposed publicly.

---

## 17. Production notes

The Docker defaults (`DJANGO_DEBUG=1`, `ALLOWED_HOSTS=*`) are for local development. For a real
deployment:

1. `DJANGO_DEBUG=0` and a strong, secret `DJANGO_SECRET_KEY`;
2. `DJANGO_ALLOWED_HOSTS=your-host` and, behind a proxy, `DJANGO_CSRF_TRUSTED_ORIGINS=https://your-host`;
3. enable `DJANGO_SECURE_SSL_REDIRECT=1`, `DJANGO_SESSION_COOKIE_SECURE=1`,
   `DJANGO_CSRF_COOKIE_SECURE=1` and HSTS (`DJANGO_SECURE_HSTS_SECONDS=31536000`) once HTTPS is
   confirmed working;
4. keep `DJANGO_X_FRAME_OPTIONS=DENY`;
5. serve behind a reverse proxy that terminates TLS and forwards `X-Forwarded-Proto`;
6. WhiteNoise serves the collected static files (`collectstatic` runs in the container entrypoint);
7. add rate limiting and request logging if the endpoint is exposed publicly;
8. re-train on recent, domain-relevant data before any real use, and keep the training run
   reproducible (the dataset hash and library versions are recorded in `metrics.json`).

---

## Further documentation

| File | Content |
| --- | --- |
| [`docs/engineering-decisions.md`](docs/engineering-decisions.md) | Why the dataset handling, cleaning, model choices, metrics, caching and architecture look the way they do (plus accepted trade-offs) |
| [`docs/delivery-checklist.md`](docs/delivery-checklist.md) | The exact commands that were run during delivery, their observed results, and what could not be executed (Docker daemon, headless browser) |
| [`docs/BA-Project-2.pdf`](docs/BA-Project-2.pdf) | The professor's original specification, included as a reference copy |
| `artifacts/training_report.md` | Generated by `python manage.py train_models`: dataset audit, split, metrics, confusion matrices, learned indicators |
