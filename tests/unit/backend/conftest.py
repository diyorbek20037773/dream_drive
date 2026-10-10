"""Backend unit tests. Run inside the backend test image or with backend/.venv on a dev host.

Puts ``<root>/backend`` (repo) or ``/app/backend`` (image) on sys.path; identical relative layout. There is no
root-level conftest: the backend and inference packages are both named ``app`` and never share an interpreter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before app.config.get_settings() is first called (it is cached).
os.environ["APP_ENV"] = "test"

ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
