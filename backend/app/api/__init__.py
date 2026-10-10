"""HTTP routers. `api_router` holds every /api/v1 route; `health_router` is mounted at the root."""

from __future__ import annotations

from fastapi import APIRouter

from app.api import admin, capabilities, evaluations, events, jobs, runs, session, simulators, videos
from app.api.health import router as health_router

API_PREFIX = "/api/v1"

api_router = APIRouter(prefix=API_PREFIX)
for _module in (session, capabilities, videos, jobs, simulators, runs, evaluations, events, admin):
    api_router.include_router(_module.router)

__all__ = ["API_PREFIX", "api_router", "health_router"]
