# syntax=docker/dockerfile:1.7
# DreamDrive backend image: API (uvicorn) and worker (python -m app.jobs.worker_main), linux/amd64, no model inside.
# Build context = repository root:
#   docker build -f backend/Dockerfile --target runtime -t dreamdrive-backend:dev .
#   docker build -f backend/Dockerfile --target test    -t dreamdrive-backend:test .
# Context filtering: backend/Dockerfile.dockerignore (BuildKit uses it instead of the root .dockerignore).

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# The base image's own setuptools/wheel carry fixable CVEs and are unused (the app runs from the uv venv).
RUN python -m pip uninstall -y setuptools wheel     && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /app/backend /app/config /app/contracts

# ---------------------------------------------------------------- dependencies (frozen lock, no dev)
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.11 \
    UV_PROJECT_ENVIRONMENT=/app/backend/.venv
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------- dependencies incl. dev group (tests)
FROM builder AS builder-dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# ---------------------------------------------------------------- runtime
FROM base AS runtime
ARG GIT_COMMIT=unknown
ARG BUILD_VERSION=dev
ARG BUILD_DIRTY=0
# ffmpeg provides the ffprobe CLI used to validate uploads before decoding (argument arrays, timeouts).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data/artifacts /data/tmp /data/evaluations \
    && chown -R 10001:10001 /data
LABEL org.opencontainers.image.title="dreamdrive-backend" \
      org.opencontainers.image.description="DreamDrive API and worker (uploaded-video analysis jobs, LiteRoad simulation runs); no model weights" \
      org.opencontainers.image.source="https://github.com/diyorbek20037773/dream_drive" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="${BUILD_VERSION}"
WORKDIR /app/backend
COPY --from=builder /app/backend/.venv /app/backend/.venv
COPY config /app/config
COPY contracts /app/contracts
COPY evaluations/baseline /app/evaluations/baseline
COPY datasets/manifests /app/datasets/manifests
COPY backend/pyproject.toml backend/uv.lock /app/backend/
COPY backend/app /app/backend/app
ENV PATH="/app/backend/.venv/bin:${PATH}" \
    PYTHONPATH=/app/backend \
    DATA_ROOT=/data \
    SQLITE_PATH=/data/dreamdrive.db \
    ARTIFACT_ROOT=/data/artifacts \
    GIT_COMMIT=${GIT_COMMIT} \
    BUILD_VERSION=${BUILD_VERSION} \
    BUILD_DIRTY=${BUILD_DIRTY}
# Fail the build if the runtime cannot import the application (missing system libraries, broken lock).
RUN python -c "import app.main, app.jobs.worker_main, cv2, av; import shutil; assert shutil.which('ffprobe')"
USER 10001:10001
VOLUME ["/data"]
EXPOSE 8000 9101
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

# ---------------------------------------------------------------- test (unit, security, scenarios; integration needs a live stack)
FROM runtime AS test
USER root
COPY --from=builder-dev /app/backend/.venv /app/backend/.venv
COPY tests /app/tests
# The OpenAPI contract test compares the generated schema with the committed frontend copies and docs/API.md.
COPY docs/API.md /app/docs/API.md
COPY frontend/src/api/openapi.json frontend/src/api/schema.d.ts /app/frontend/src/api/
USER 10001:10001
# No pytest cache writes into the (root-owned) test tree.
ENV PYTEST_ADDOPTS="-p no:cacheprovider" \
    APP_ENV=test
CMD ["python", "-m", "pytest", "/app/tests/unit/backend", "/app/tests/security", "-q"]
