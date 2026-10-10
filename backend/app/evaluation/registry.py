"""Evaluation artifact registry: ``<root>/<32-hex id>.json`` files written by the scenario suite.

``list_evaluations`` / ``load_evaluation`` return the Evaluation shape (docs/IMPLEMENTATION_SPEC.md 3.4) without the
bulky per-run ``details``; ``load_evaluation_details`` returns them. Ids are validated to block path traversal.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_ID = re.compile(r"^[a-f0-9]{32}$")


def _strip(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k != "details"}


def _read(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and _ID.match(str(doc.get("id", ""))) else None


def write_evaluation(root: Path, doc: dict[str, Any]) -> Path:
    """Atomic write (temp file in the same directory + os.replace)."""
    eval_id = str(doc.get("id", ""))
    if not _ID.match(eval_id):
        raise ValueError("evaluation id must be 32 lowercase hex characters")
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{eval_id}.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=False)
        final = root / f"{eval_id}.json"
        os.replace(tmp, final)
        return final
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def list_evaluations(root: Path) -> list[dict[str, Any]]:
    """All readable evaluations, newest first."""
    if not root.is_dir():
        return []
    docs = [doc for p in root.glob("*.json") if _ID.match(p.stem) and (doc := _read(p)) is not None]
    docs.sort(key=lambda d: str(d.get("created_utc", "")), reverse=True)
    return [_strip(d) for d in docs]


def load_evaluation(root: Path, eval_id: str) -> dict[str, Any] | None:
    if not _ID.match(eval_id):
        return None
    doc = _read(root / f"{eval_id}.json")
    return _strip(doc) if doc is not None else None


def load_evaluation_details(root: Path, eval_id: str) -> dict[str, Any] | None:
    if not _ID.match(eval_id):
        return None
    doc = _read(root / f"{eval_id}.json")
    return doc.get("details") if doc is not None else None
