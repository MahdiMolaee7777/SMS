# ---------------------------------------------------------------------------
# SMS Spam Detector - convenience shortcuts
#   make help        list every target
#   make setup       virtualenv + dependencies + migrate + train  (local Python)
#   make docker-run  build + train + start through Docker Compose
# ---------------------------------------------------------------------------

PYTHON ?= python3
VENV   ?= .venv
BIN     = $(VENV)/bin
MANAGE  = $(BIN)/python manage.py
COMPOSE = docker compose

.DEFAULT_GOAL := help
.PHONY: help venv install migrate train run test test-verbose coverage css css-watch \
        vendor check status clean docker-build docker-train docker-up docker-down \
        docker-test docker-logs docker-shell setup docker-run

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- local Python workflow -------------------------------------------------
venv: ## Create the virtual environment
	$(PYTHON) -m venv $(VENV)

install: venv ## Install dependencies into the virtual environment
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -r requirements-dev.txt

migrate: ## Apply database migrations
	$(MANAGE) migrate

train: ## Train the three models and write artifacts/metrics.json + report
	$(MANAGE) train_models

status: ## Show which models and evaluation results are available
	$(MANAGE) model_status

run: ## Start the development server on http://127.0.0.1:8000
	$(MANAGE) runserver 0.0.0.0:8000

test: ## Run the Django test suite
	$(MANAGE) test -v 1

test-verbose: ## Run the tests with per-test output
	$(MANAGE) test -v 2

coverage: ## Measure test coverage of the detector app
	$(BIN)/coverage run --source=detector,config manage.py test
	$(BIN)/coverage report -m

check: ## Run Django's system checks
	$(MANAGE) check

setup: install migrate train ## Full local setup: install, migrate, train

# --- front-end assets ------------------------------------------------------
css: ## Compile Tailwind CSS into assets/css/tailwind.css
	npx --yes tailwindcss@3.4.17 -c tailwind.config.js -i assets/css/input.css -o assets/css/tailwind.css --minify

css-watch: ## Recompile Tailwind CSS on every template change
	npx --yes tailwindcss@3.4.17 -c tailwind.config.js -i assets/css/input.css -o assets/css/tailwind.css --watch

vendor: ## Refresh the vendored Chart.js copy into assets/vendor/
	npm install --no-save chart.js@4.4.7
	cp node_modules/chart.js/dist/chart.umd.js assets/vendor/chart.umd.min.js

# --- Docker workflow -------------------------------------------------------
docker-build: ## Build the Docker image
	$(COMPOSE) build

docker-train: ## Train the models inside the container (writes into ./artifacts)
	$(COMPOSE) run --rm web python manage.py train_models

docker-up: ## Start the stack (migrations run automatically)
	$(COMPOSE) up -d

docker-run: docker-build docker-train docker-up ## Build, train, start
	@echo "SMS Spam Detector is available on http://localhost:8000"

docker-down: ## Stop the stack
	$(COMPOSE) down

docker-logs: ## Follow the container logs
	$(COMPOSE) logs -f web

docker-test: ## Run the test suite inside a one-off container
	$(COMPOSE) run --rm web python manage.py test -v 1

docker-shell: ## Open a shell inside a one-off container
	$(COMPOSE) run --rm web /bin/bash

clean: ## Remove caches and generated files (keeps trained models)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage staticfiles
