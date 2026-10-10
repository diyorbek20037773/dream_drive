"""Security tests (ownership, CSRF, traversal, upload abuse, admin). Run with the backend interpreter.

Puts ``<root>/backend`` (repo) or ``/app/backend`` (image) on sys.path, plus ``tests/unit/backend`` for the shared
API test harness (test_api_harness.py). No root-level conftest exists: both packages are named ``app``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["APP_ENV"] = "test"

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "backend", ROOT / "tests" / "unit" / "backend"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
