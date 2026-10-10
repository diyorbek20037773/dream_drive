# syntax=docker/dockerfile:1
# DreamDrive web image. Build context = repository root:
#   docker build -f frontend/Dockerfile -t dreamdrive-web:dev .
# frontend/Dockerfile.dockerignore (BuildKit) limits the context to the frontend sources.

FROM node:22-bookworm-slim AS build
WORKDIR /src/frontend
ENV CI=1 \
    NPM_CONFIG_AUDIT=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_UPDATE_NOTIFIER=false
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Type-checks, then builds static assets. No API host is injected: the bundle only uses relative /api/v1 URLs.
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.30-alpine AS web
ARG GIT_COMMIT=unknown
ARG BUILD_VERSION=dev
ARG BUILD_DIRTY=0
ARG BUILD_DATE=unknown
# Base-image labels (created, url, licenses) are overridden so they describe this image, not nginx.
LABEL org.opencontainers.image.title="dreamdrive-web" \
      org.opencontainers.image.description="DreamDrive AI static web UI (nginx, SPA)" \
      org.opencontainers.image.source="https://github.com/diyorbek20037773/dream_drive" \
      org.opencontainers.image.url="https://github.com/diyorbek20037773/dream_drive" \
      org.opencontainers.image.licenses="NOASSERTION" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.dreamdrive.build.dirty="${BUILD_DIRTY}"
# Pick up Alpine security fixes published after the base image was built.
USER root
RUN apk upgrade --no-cache
COPY frontend/nginx-main.conf /etc/nginx/nginx.conf
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY frontend/nginx-security-headers.conf /etc/nginx/dreamdrive-security-headers.conf
COPY --from=build /src/frontend/dist /usr/share/nginx/html
USER 101
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
