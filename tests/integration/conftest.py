"""Integration tests against a LIVE DreamDrive stack (gateway URL ``DD_API_BASE_URL``, default http://localhost:8080).

They use only HTTP (httpx with a cookie jar), exactly like the browser. When the stack is unreachable they are
skipped, unless ``DD_REQUIRE_LIVE=1`` (``make qa``/``make integration``), in which case they fail. The uploaded
video comes from ``DD_FIXTURE_VIDEO`` (default: the licensed fixture fetched by scripts/fetch_fixtures.py).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from live_client import BASE_URL, REQUIRE_LIVE, Api

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "cache" / "commons_dashcam_daylight_cc0_96s_10s.mp4"


def _unavailable(reason: str) -> None:
    if REQUIRE_LIVE:
        pytest.fail(f"live stack required (DD_REQUIRE_LIVE=1): {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def base_url() -> str:
    try:
        response = httpx.get(f"{BASE_URL}/health/ready", timeout=5.0)
    except httpx.HTTPError as exc:
        _unavailable(f"{BASE_URL} unreachable ({type(exc).__name__})")
    if response.status_code != 200:
        _unavailable(f"{BASE_URL}/health/ready answered {response.status_code}")
    return BASE_URL


@pytest.fixture
def api(base_url: str) -> Iterator[Api]:
    client = Api(base_url)
    yield client
    client.close()


@pytest.fixture(scope="session")
def fixture_video() -> Path:
    path = Path(os.environ.get("DD_FIXTURE_VIDEO") or DEFAULT_FIXTURE)
    if not path.is_file():
        _unavailable(f"fixture video missing: {path} (run scripts/fetch_fixtures.py or set DD_FIXTURE_VIDEO)")
    return path
