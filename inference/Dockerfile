# syntax=docker/dockerfile:1.7
# DreamDrive inference service, CPU default (linux/amd64). Build context = repository root:
#   docker build -f inference/Dockerfile --target runtime -t dreamdrive-inference:dev .
#   docker build -f inference/Dockerfile --target test    -t dreamdrive-inference:test .
# Checkpoints (YOLOPv2, RF-DETR Nano) are NOT baked in: run `python -m app.model_init` (one-shot) into /models.
# Runtime is offline: HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set and rfdetr downloads are refused in code.
# Context filtering: inference/Dockerfile.dockerignore (BuildKit uses it instead of the root .dockerignore).

ARG PYTHON_IMAGE=python:3.11-slim-bookworm
# Locked but never imported by the service (checked 2026-10-10: not in sys.modules after loading both models,
# predicting and importing app.main): supervision's video/plotting deps (av, matplotlib and its deps) and the
# transformers / huggingface_hub CLI and download-accelerator deps. Skipping them saves ~200 MB; the lock is unchanged
# and the unit + real-weight smoke run in an image built the same way.
ARG UV_SKIP="--no-install-package av --no-install-package matplotlib --no-install-package fonttools \
--no-install-package kiwisolver --no-install-package contourpy --no-install-package cycler \
--no-install-package pyparsing --no-install-package python-dateutil --no-install-package six \
--no-install-package hf-xet --no-install-package typer --no-install-package rich --no-install-package shellingham \
--no-install-package markdown-it-py --no-install-package mdurl"

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# The base image's own setuptools/wheel carry fixable CVEs and are unused (the app runs from the uv venv).
RUN python -m pip uninstall -y setuptools wheel \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /models /app/inference /app/config \
    && chown 10001:10001 /models

# ---------------------------------------------------------------- dependencies (frozen lock, no dev)
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.11 \
    UV_PROJECT_ENVIRONMENT=/app/inference/.venv
WORKDIR /app/inference
COPY inference/pyproject.toml inference/uv.lock ./
ARG UV_SKIP
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${UV_SKIP} --no-install-package pygments

# ---------------------------------------------------------------- dependencies incl. dev group (tests)
FROM builder AS builder-dev
ARG UV_SKIP
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project ${UV_SKIP}

# ---------------------------------------------------------------- runtime
FROM base AS runtime
ARG GIT_COMMIT=unknown
ARG BUILD_VERSION=dev
LABEL org.opencontainers.image.title="dreamdrive-inference" \
      org.opencontainers.image.description="DreamDrive internal inference service: YOLOPv2 + RF-DETR Nano (CPU, linux/amd64)" \
      org.opencontainers.image.source="https://github.com/diyorbek20037773/dream_drive" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.licenses="MIT"
WORKDIR /app/inference
COPY --from=builder /app/inference/.venv /app/inference/.venv
COPY config/models.resolved.yaml /app/config/models.resolved.yaml
COPY inference/pyproject.toml /app/inference/pyproject.toml
COPY inference/app /app/inference/app
ENV PATH="/app/inference/.venv/bin:${PATH}" \
    PYTHONPATH=/app/inference \
    MODEL_ROOT=/models \
    MODEL_MANIFEST_PATH=/app/config/models.resolved.yaml \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    INFERENCE_DEVICE=cpu \
    INFERENCE_PRECISION=fp32 \
    GIT_COMMIT=${GIT_COMMIT} \
    BUILD_VERSION=${BUILD_VERSION}
USER 10001:10001
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]

# ---------------------------------------------------------------- test (unit + model smoke; weights mounted at /models)
FROM runtime AS test
USER root
COPY --from=builder-dev /app/inference/.venv /app/inference/.venv
COPY tests/unit/inference /app/tests/unit/inference
COPY tests/model_smoke /app/tests/model_smoke
COPY tests/fixtures/images /app/tests/fixtures/images
USER 10001:10001
# No pytest cache writes into the (root-owned) test tree.
ENV PYTEST_ADDOPTS="-p no:cacheprovider"
CMD ["pytest", "/app/tests/unit/inference", "-q"]
