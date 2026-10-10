"""HTTP client for the live-stack integration tests (cookie session + CSRF header, like the SPA)."""

from __future__ import annotations

import os
from typing import Any

import httpx

BASE_URL = os.environ.get("DD_API_BASE_URL", "http://localhost:8080").rstrip("/")
REQUIRE_LIVE = os.environ.get("DD_REQUIRE_LIVE") == "1"


class Api:
    """Cookie-authenticated client that adds the CSRF header to mutations, like the SPA."""

    def __init__(self, base_url: str) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=httpx.Timeout(60.0, connect=10.0))
        response = self.http.post("/api/v1/session")
        response.raise_for_status()
        self.session: dict[str, Any] = response.json()

    @property
    def csrf(self) -> dict[str, str]:
        return {"X-CSRF-Token": self.session["csrf_token"]}

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.http.get(url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        headers = {**self.csrf, **kwargs.pop("headers", {})}
        return self.http.post(url, headers=headers, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.http.delete(url, headers=self.csrf, **kwargs)

    def close(self) -> None:
        self.http.close()
