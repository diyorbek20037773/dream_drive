# DreamDrive developer, QA and delivery entry points. GNU make + bash: Git Bash on Windows, or Linux/WSL.
# PowerShell users: run these from Git Bash or WSL (docs/OPERATIONS.md). Every Python/Node gate runs inside a
# container, so Docker is the only required runtime. `make help` lists the targets.
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DELETE_ON_ERROR:
.DEFAULT_GOAL := help
MAKEFLAGS += --no-builtin-rules --no-print-directory

include scripts/tool-images.env

# Git Bash would rewrite container paths such as /evidence into C:/Program Files/Git/evidence.
export MSYS_NO_PATHCONV := 1
export MSYS2_ARG_CONV_EXCL := *

GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
BUILD_DIRTY := $(shell if [ -z "$$(git status --porcelain 2>/dev/null)" ]; then echo 0; else echo 1; fi)
BUILD_VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
DD_TAG ?= dev
export GIT_COMMIT BUILD_DIRTY BUILD_VERSION DD_TAG

# Host path for `docker run -v` (D:/... on Git Bash) and host uid for evidence ownership (0 = leave as is).
HOST_PWD := $(shell pwd -W 2>/dev/null || pwd)
ifneq (,$(findstring MINGW,$(shell uname -s)))
DD_HOST_UID := 0
DD_HOST_GID := 0
else
DD_HOST_UID := $(shell id -u)
DD_HOST_GID := $(shell id -g)
endif
export DD_HOST_UID DD_HOST_GID

DD_HTTP_PORT ?= $(or $(shell sed -n 's/^DD_HTTP_PORT=\([0-9][0-9]*\)$$/\1/p' .env 2>/dev/null),8080)
# Evidence directory for junit/reports. Must start with ./ (or /) so Compose treats it as a bind mount.
EVIDENCE_DIR ?= ./artifacts/qa/$(GIT_COMMIT)/manual
export DD_EVIDENCE_DIR := $(EVIDENCE_DIR)
SEEDS ?= 1,2,3

COMPOSE_FILES := -f compose.yaml
ifeq ($(GPU),1)
COMPOSE_FILES += -f compose.gpu.yaml
endif
ifeq ($(OPS),1)
COMPOSE_FILES += -f compose.ops.yaml
endif
COMPOSE := docker compose $(COMPOSE_FILES)
TEST_COMPOSE := docker compose -f compose.yaml -f compose.test.yaml --profile test
TEST_RUN := $(TEST_COMPOSE) run --rm --no-deps
LIVE_RUN := $(TEST_COMPOSE) run --rm --no-deps -e DD_REQUIRE_LIVE=1
# GPU/CARLA tests are deselected (not skipped) in CPU gates; their support is claimed only from their own runs.
PYTEST := python -m pytest -q -p no:cacheprovider -m "not gpu and not carla"
TOOL_RUN := docker run --rm --user 10001:10001 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1

.PHONY: help doctor init-env build build-test up down logs ps model-init wait-ready lint lint-python lint-frontend \
	lint-shell typecheck test \
	integration model-smoke scenarios fixtures security e2e eval-scenarios gen-types gen-types-check qa backup \
	backup-test restore-test rollback deploy-check smoke smoke-published sbom ops-check deploy-selftest \
	mlflow-selftest mlflow-log evidence-dir

help: ## List targets
	@grep -hE '^[a-z][a-z0-9-]+:.*## ' Makefile | sed -E 's/^([a-z0-9-]+):.*## (.*)$$/  \1\t\2/' | expand -t 22
	echo
	echo "  Options: GPU=1 (CUDA inference override)  OPS=1 (Prometheus/Grafana/MLflow)  DD_TAG=dev"

evidence-dir:
	@mkdir -p "$(EVIDENCE_DIR)" tests/fixtures/cache
	# Linux: container uid 10001 writes junit files into this bind mount (no effect on Windows mounts).
	if ! chmod 0777 "$(EVIDENCE_DIR)" 2>/dev/null; then echo "note: could not chmod $(EVIDENCE_DIR)"; fi

# ---------------------------------------------------------------------------------------------------------------
# Environment and stack
# ---------------------------------------------------------------------------------------------------------------
doctor: ## Check Docker/Compose/git/repo prerequisites
	@bash scripts/doctor.sh

init-env: ## Create .env with generated secrets (never printed)
	@bash scripts/init-env.sh

build: ## Build runtime images (web, backend, inference) with GIT_COMMIT/BUILD_VERSION labels
	@$(COMPOSE) build

build-test: ## Build the backend/inference test images
	@$(TEST_COMPOSE) build backend-test inference-test

up: ## Build if needed, start the stack (volume-init, model-init, services) and wait until ready
	@test -f .env || bash scripts/init-env.sh
	$(COMPOSE) up -d --build
	OPS=$(OPS) bash scripts/wait-ready.sh

wait-ready: ## Wait for a healthy stack and a working gateway -> api path
	@OPS=$(OPS) bash scripts/wait-ready.sh

down: ## Stop the stack (named volumes are kept; never -v)
	@$(COMPOSE) down

logs: ## Follow logs (SERVICE=api to filter)
	@$(COMPOSE) logs -f --tail=200 $(SERVICE)

ps: ## Service status
	@$(COMPOSE) ps

model-init: ## Download + SHA256-verify the model checkpoint into the model-cache volume
	@$(COMPOSE) run --rm model-init

fixtures: build-test evidence-dir ## Download/verify licensed test media into tests/fixtures/cache (git-ignored)
	@docker run --rm --user "$(if $(filter 0,$(DD_HOST_UID)),10001,$(DD_HOST_UID)):$(if $(filter 0,$(DD_HOST_GID)),10001,$(DD_HOST_GID))" \
	  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -v "$(HOST_PWD):/repo" -w /repo --entrypoint python \
	  dreamdrive-backend-test:$(DD_TAG) scripts/fetch_fixtures.py

# ---------------------------------------------------------------------------------------------------------------
# Static gates
# ---------------------------------------------------------------------------------------------------------------
lint: lint-python lint-frontend lint-shell ## All linters: ruff, eslint, shellcheck, actionlint

lint-python: build-test ## ruff: backend, inference, ops/deploy scripts (in the test images)
	@$(TEST_RUN) backend-test sh -c 'set -- app; for d in ../tests/unit/backend ../tests/integration ../tests/scenarios ../tests/security; do [ -d "$$d" ] && set -- "$$@" "$$d"; done; exec python -m ruff check --no-cache --config pyproject.toml "$$@"'
	$(TEST_RUN) inference-test sh -c 'set -- app; for d in ../tests/unit/inference ../tests/model_smoke; do [ -d "$$d" ] && set -- "$$@" "$$d"; done; exec python -m ruff check --no-cache --line-length 120 "$$@"'
	$(TOOL_RUN) -v "$(HOST_PWD):/repo:ro" -w /repo --entrypoint python dreamdrive-backend-test:$(DD_TAG) \
	  -m ruff check --no-cache --config backend/pyproject.toml scripts deploy

lint-frontend: ## eslint (frontend, in a Node container)
	@$(TEST_RUN) node lint

lint-shell: ## shellcheck (scripts, deploy) and actionlint (.github/workflows)
	@docker run --rm -v "$(HOST_PWD):/mnt:ro" -w /mnt $(SHELLCHECK_IMAGE) -x \
	  scripts/*.sh scripts/container/*.sh deploy/*.sh deploy/lib/common.sh
	docker run --rm -v "$(HOST_PWD):/repo:ro" -w /repo $(ACTIONLINT_IMAGE) -color

typecheck: build-test ## mypy (backend, inference) and tsc (frontend)
	@$(TEST_RUN) backend-test python -m mypy --cache-dir /tmp/mypy app
	$(TEST_RUN) inference-test python -m mypy --cache-dir /tmp/mypy app
	$(TEST_RUN) node typecheck

gen-types: ## Regenerate frontend/src/api/{openapi.json,schema.d.ts} from the backend
	@DD_BACKEND_IMAGE=dreamdrive-backend:$(DD_TAG) bash scripts/gen-types.sh

gen-types-check: ## Fail if the generated API types are stale
	@DD_BACKEND_IMAGE=dreamdrive-backend:$(DD_TAG) bash scripts/gen-types.sh --check

ops-check: ## promtool config/rules/unit tests and Grafana dashboard drift
	@docker run --rm -v "$(HOST_PWD)/ops/prometheus:/etc/prometheus:ro" --entrypoint promtool $(PROMETHEUS_IMAGE) \
	  check config /etc/prometheus/prometheus.yml
	docker run --rm -v "$(HOST_PWD)/ops/prometheus:/etc/prometheus:ro" --entrypoint promtool $(PROMETHEUS_IMAGE) \
	  check rules /etc/prometheus/alerts.yml
	docker run --rm -v "$(HOST_PWD)/ops/prometheus:/etc/prometheus:ro" -w /etc/prometheus --entrypoint promtool \
	  $(PROMETHEUS_IMAGE) test rules alerts_test.yml
	$(TOOL_RUN) -v "$(HOST_PWD):/repo:ro" -w /repo $(PYTHON_IMAGE) python scripts/gen_grafana_dashboards.py --check

deploy-check: ## Validate every compose combination + deploy.sh guards (placeholder values only)
	@bash scripts/deploy-check.sh

deploy-selftest: ## Unit tests of the backup/restore helper and the smoke client
	@$(TOOL_RUN) -v "$(HOST_PWD)/deploy:/opt/dd/deploy:ro" -w /tmp $(PYTHON_IMAGE) \
	  python -m unittest discover -s /opt/dd/deploy/tests -v

mlflow-selftest: ## Unit tests of scripts/log_evaluations_to_mlflow.py inside the MLflow image
	@$(TOOL_RUN) -v "$(HOST_PWD)/scripts:/opt/dd/scripts:ro" -v "$(HOST_PWD)/config:/opt/dd/config:ro" -w /tmp \
	  $(MLFLOW_IMAGE) python -m unittest discover -s /opt/dd/scripts/tests -v

# ---------------------------------------------------------------------------------------------------------------
# Test gates (Python in the test images; live gates fail instead of skipping)
# ---------------------------------------------------------------------------------------------------------------
test: build-test evidence-dir ## Unit tests: backend + inference
	@$(TEST_RUN) backend-test $(PYTEST) ../tests/unit/backend --junitxml=/evidence/junit-unit-backend.xml
	$(TEST_RUN) inference-test $(PYTEST) ../tests/unit/inference --junitxml=/evidence/junit-unit-inference.xml

scenarios: build-test up evidence-dir ## LiteRoad closed-loop scenarios (incl. live learned-sim validation via inference)
	@$(LIVE_RUN) backend-test $(PYTEST) ../tests/scenarios --junitxml=/evidence/junit-scenarios.xml

model-smoke: build-test model-init evidence-dir ## Real-weight CPU inference smoke (inference test image)
	@$(LIVE_RUN) inference-test $(PYTEST) ../tests/model_smoke --junitxml=/evidence/junit-model-smoke.xml

integration: build-test up evidence-dir ## tests/integration against the running stack (real redis + inference)
	@$(LIVE_RUN) backend-test $(PYTEST) ../tests/integration --junitxml=/evidence/junit-integration.xml

e2e: up evidence-dir ## Playwright (official image matching package-lock) against the gateway
	@PLAYWRIGHT_VERSION=$$(bash scripts/playwright-version.sh)
	export PLAYWRIGHT_VERSION
	echo "Playwright $$PLAYWRIGHT_VERSION (mcr.microsoft.com/playwright:v$$PLAYWRIGHT_VERSION-noble)"
	$(TEST_RUN) e2e

eval-scenarios: build-test up evidence-dir ## Run the synthetic scenario suite via the API; save Evaluation JSON
	@$(LIVE_RUN) backend-test python /opt/dreamdrive/scripts/eval_scenarios.py --base http://gateway:8080 \
	  --seeds $(SEEDS) --out /evidence/evaluations

security: build-test up evidence-dir ## tests/security, pip-audit, npm audit, gitleaks, trivy
	@bash scripts/security.sh

smoke: evidence-dir ## Smoke the local gateway (UPLOAD=clip.mp4 adds a real end-to-end video job)
	@bash deploy/smoke.sh "http://127.0.0.1:$(DD_HTTP_PORT)" --evidence "$(EVIDENCE_DIR)" $(if $(UPLOAD),--upload "$(UPLOAD)")

qa: ## Full release QA: every gate, evidence in artifacts/qa/<commit>/ (fails on any failed/skipped gate)
	@bash scripts/qa.sh

sbom: ## SPDX SBOMs of the local images (Syft)
	@bash scripts/sbom.sh

# ---------------------------------------------------------------------------------------------------------------
# Data, delivery, operations
# ---------------------------------------------------------------------------------------------------------------
backup: ## Coherent backup of the local stack (SQLite online backup + artifacts) into ./backups
	@bash deploy/backup.sh --dev

backup-test: evidence-dir ## Backup, then restore into a separate compose project and verify job rows/reports
	@bash deploy/backup.sh --dev
	bash deploy/restore-test.sh --dev --require-job --boot-api --evidence "$(EVIDENCE_DIR)"

restore-test: evidence-dir ## Restore the newest (or BACKUP=file) backup into a separate project and verify it
	@bash deploy/restore-test.sh --dev --require-job --boot-api --evidence "$(EVIDENCE_DIR)" $(if $(BACKUP),--backup "$(BACKUP)")

rollback: ## Production: redeploy the previous recorded release (ENV_FILE=/etc/dreamdrive/dreamdrive.env [TO=<commit>])
	@test -n "$(ENV_FILE)" || { echo "usage: make rollback ENV_FILE=/etc/dreamdrive/dreamdrive.env [TO=<commit>]" >&2; exit 64; }
	bash deploy/rollback.sh --env-file "$(ENV_FILE)" $(if $(TO),--to "$(TO)")

smoke-published: ## Pull-only smoke of published digests (RELEASE_ENV=artifacts/release/<sha>/release.env [UPLOAD=clip])
	@test -n "$(RELEASE_ENV)" || { echo "usage: make smoke-published RELEASE_ENV=artifacts/release/<commit>/release.env" >&2; exit 64; }
	bash scripts/smoke-published.sh --release "$(RELEASE_ENV)" $(if $(UPLOAD),--upload "$(UPLOAD)")

mlflow-log: ## Log local evaluation JSON to the MLflow service (OPS stack)
	@docker compose -f compose.yaml -f compose.ops.yaml --profile mlflow-log run --rm mlflow-log
