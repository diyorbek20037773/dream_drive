"""Structured JSON logs on stdout (docs/OBSERVABILITY.md).

Every record carries: UTC timestamp, level, component, logger, message, and — when known — request_id, job_id,
run_id, frame_id, code and duration_ms. Correlation ids come from context variables set by the request middleware
and the worker jobs, or from ``extra=``. Cookie values, CSRF tokens, passwords and bearer tokens are redacted from
messages; uploaded file names are never passed to the logger in the first place.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("dd_request_id", default=None)
job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("dd_job_id", default=None)
run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("dd_run_id", default=None)

_CONTEXT_VARS = {"request_id": request_id_var, "job_id": job_id_var, "run_id": run_id_var}
# Extra attributes copied into the JSON object when present on the record.
_EXTRA_KEYS = (
    "request_id",
    "job_id",
    "run_id",
    "eval_id",
    "frame_id",
    "code",
    "method",
    "route",
    "status",
    "duration_ms",
    "state",
    "outcome",
    "version",
    "slot",
    "queue",
    "pid",
    "removed_jobs",
    "removed_runs",
    "removed_bytes",
)

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(dd_session|dd_csrf)=[^;\s,]+", re.IGNORECASE), r"\1=[redacted]"),
    (re.compile(r"(x-csrf-token['\"]?\s*[:=]\s*['\"]?)[^'\"\s,;]+", re.IGNORECASE), r"\1[redacted]"),
    (re.compile(r"((?:admin_)?password['\"]?\s*[:=]\s*['\"]?)[^'\"\s,;]+", re.IGNORECASE), r"\1[redacted]"),
    (re.compile(r"(authorization['\"]?\s*[:=]\s*['\"]?)(?:bearer\s+)?[^'\"\s,;]+", re.IGNORECASE), r"\1[redacted]"),
    (re.compile(r"(cookie['\"]?\s*[:=]\s*['\"]?)[^'\"\n]+", re.IGNORECASE), r"\1[redacted]"),
    (re.compile(r"(APP_SECRET['\"]?\s*[:=]\s*['\"]?)[^'\"\s,;]+"), r"\1[redacted]"),
)


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class JsonFormatter(logging.Formatter):
    def __init__(self, component: str) -> None:
        super().__init__()
        self.component = component

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "component": getattr(record, "component", None) or self.component,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        for key, var in _CONTEXT_VARS.items():
            value = var.get()
            if value is not None:
                payload[key] = value
        for key in _EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            # Type and a redacted traceback stay server-side; responses never contain them.
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exc"] = redact("".join(traceback.format_exception(*record.exc_info))[-4000:])
        return json.dumps(payload, ensure_ascii=False, default=str)


_configured_component: str | None = None


def configure_logging(component: str, level: str = "INFO") -> None:
    """Install the JSON handler on the root logger (idempotent; later calls update level/component)."""
    global _configured_component
    root = logging.getLogger()
    numeric = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric, int):
        numeric = logging.INFO
    handler: logging.Handler | None = next((h for h in root.handlers if getattr(h, "_dd_json", False)), None)
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler._dd_json = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    handler.setFormatter(JsonFormatter(component))
    root.setLevel(numeric)
    # Uvicorn's own handlers would print plain text; route its error logger through ours and silence the access log
    # (the HTTP middleware writes a JSON access line with the route template instead of the raw URL).
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False
    access.disabled = True
    for noisy in ("httpx", "httpcore", "rq.worker", "rq.queue"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING) if noisy.startswith("http") else numeric)
    _configured_component = component


@contextmanager
def log_context(**values: str | None) -> Iterator[None]:
    """Bind request_id/job_id/run_id for the duration of a block (worker jobs, background tasks)."""
    tokens = []
    for key, value in values.items():
        var = _CONTEXT_VARS.get(key)
        if var is not None:
            tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
