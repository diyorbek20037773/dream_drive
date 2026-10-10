# syntax=docker/dockerfile:1.7
# DreamDrive inference service (CPU default). GPU variant: Dockerfile.gpu
FROM python:3.11-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY scripts ./scripts

FROM python:3.11-slim-bookworm AS runtime
ARG GIT_COMMIT=unknown
ARG BUILD_VERSION=dev
LABEL org.opencontainers.image.title="dreamdrive-inference" \
      org.opencontainers.image.source="https://github.com/diyorbek20037773/dream_drive" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.licenses="MIT"
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 app && useradd -u 10001 -g app -m -s /usr/sbin/nologin app \
    && mkdir -p /models /app/config && chown -R app:app /models
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 GIT_COMMIT=${GIT_COMMIT} BUILD_VERSION=${BUILD_VERSION}
USER 10001:10001
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
