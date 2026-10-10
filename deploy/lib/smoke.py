#!/usr/bin/env python3
"""DreamDrive HTTP(S) smoke test against the public gateway (stdlib only; Python 3.9+).

    python3 deploy/lib/smoke.py https://dreamdrive.example.org [--expect-commit SHA] [--upload clip.mp4]

Checks: /health/live, /health/ready (waits), web root, /metrics denied (404), capabilities (+ build commit),
session cookie + CSRF token, an authenticated read, SSE stream headers/first bytes, and optionally a real video
upload processed end-to-end (job succeeded, results listed, annotated.mp4 served with HTTP Range).
Prints a JSON summary; never prints cookies, CSRF tokens or secrets. Exit 0 only if every check passed.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import mimetypes
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled", "interrupted"}


class Client:
    def __init__(self, base: str, insecure: bool, timeout: float) -> None:
        if not base.startswith(("http://", "https://")):
            raise ValueError("base URL must be http:// or https://")
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.jar = http.cookiejar.CookieJar()
        handlers: list[Any] = [urllib.request.HTTPCookieProcessor(self.jar)]
        if insecure:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            handlers.append(urllib.request.HTTPSHandler(context=ctx))
        self.opener = urllib.request.build_opener(*handlers)
        self.csrf: str | None = None

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None,
                timeout: float | None = None) -> tuple[int, dict[str, str], bytes]:
        hdrs = {"Accept": "application/json", "User-Agent": "dreamdrive-smoke/1"}
        hdrs.update(headers or {})
        if self.csrf and method in {"POST", "DELETE", "PUT", "PATCH"}:
            hdrs["X-CSRF-Token"] = self.csrf
        # The scheme is restricted to http(s) in __init__; paths are fixed API routes.
        req = urllib.request.Request(self.base + path, data=body, headers=hdrs, method=method)  # noqa: S310
        try:
            with self.opener.open(req, timeout=timeout or self.timeout) as resp:
                return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()

    def json(self, method: str, path: str, payload: Any = None) -> tuple[int, Any]:
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"} if body is not None else {}
        status, _, raw = self.request(method, path, body, headers)
        try:
            return status, json.loads(raw) if raw else None
        except ValueError:
            return status, None


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, **detail: Any) -> bool:
        self.checks.append({"check": name, "ok": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, sort_keys=True)}", file=sys.stderr)
        return ok

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c["ok"] for c in self.checks)


def multipart(field: str, path: Path) -> tuple[bytes, str]:
    boundary = f"dd-smoke-{uuid.uuid4().hex}"
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"smoke{path.suffix}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode()
    return head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def wait_ready(c: Client, r: Report, wait_s: float) -> None:
    deadline = time.monotonic() + wait_s
    status = 0
    last_error: str | None = None
    while True:
        try:
            status, _, _ = c.request("GET", "/health/ready")
        except OSError as exc:  # connection refused / TLS not ready yet
            status = 0
            last_error = type(exc).__name__
        else:
            last_error = None
        if status == 200 or time.monotonic() >= deadline:
            break
        time.sleep(3)
    r.add("health_ready", status == 200, status=status, error=last_error)


def check_sse(c: Client, r: Report, wait_s: float) -> None:
    """The stream must open through the gateway and deliver bytes (an event or the 15 s ping) unbuffered."""
    req = urllib.request.Request(c.base + "/api/v1/events", headers={"Accept": "text/event-stream"})  # noqa: S310
    started = time.monotonic()
    try:
        with c.opener.open(req, timeout=wait_s) as resp:
            ctype = resp.headers.get("Content-Type", "")
            first = resp.readline()
            r.add("sse_stream", resp.status == 200 and ctype.startswith("text/event-stream") and bool(first),
                  status=resp.status, content_type=ctype, first_bytes_after_s=round(time.monotonic() - started, 2))
    except urllib.error.HTTPError as exc:
        r.add("sse_stream", False, status=exc.code)
    except OSError as exc:
        r.add("sse_stream", False, error=type(exc).__name__, waited_s=round(time.monotonic() - started, 2))


def upload_flow(c: Client, r: Report, video: Path, caps: dict[str, Any] | None, job_timeout: float) -> None:
    available = bool(caps and caps.get("video_analysis", {}).get("available"))
    if not r.add("video_analysis_available", available,
                 reason=(caps or {}).get("video_analysis", {}).get("reason")):
        return
    body, ctype = multipart("file", video)
    status, _, raw = c.request("POST", "/api/v1/videos", body, {"Content-Type": ctype}, timeout=300)
    job = json.loads(raw) if raw and status == 202 else None
    if not r.add("video_upload_accepted", status == 202 and bool(job and job.get("id")), status=status):
        return
    assert job is not None
    job_id = job["id"]
    deadline = time.monotonic() + job_timeout
    state = job.get("state")
    while state not in TERMINAL_JOB_STATES and time.monotonic() < deadline:
        time.sleep(3)
        status, job = c.json("GET", f"/api/v1/jobs/{job_id}")
        state = (job or {}).get("state")
    if not r.add("video_job_succeeded", state == "succeeded", state=state,
                 error_code=(job or {}).get("error_code"), frames_done=(job or {}).get("frames_done")):
        return
    status, results = c.json("GET", f"/api/v1/jobs/{job_id}/results")
    names = {a.get("name"): a for a in (results or {}).get("artifacts", [])}
    r.add("video_results", status == 200 and {"annotated.mp4", "report.json"} <= names.keys(),
          status=status, artifacts=sorted(names))
    mp4 = names.get("annotated.mp4")
    if mp4:
        status, headers, chunk = c.request("GET", mp4["url"], headers={"Range": "bytes=0-99", "Accept": "*/*"})
        r.add("annotated_mp4_range", status == 206 and len(chunk) == 100 and "content-range" in headers,
              status=status, content_type=headers.get("content-type"))


def main() -> int:
    ap = argparse.ArgumentParser(description="DreamDrive gateway smoke test")
    ap.add_argument("base_url")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification (local internal CA only)")
    ap.add_argument("--expect-commit", help="fail unless /capabilities reports this build commit")
    ap.add_argument("--upload", type=Path, help="video file for an end-to-end analysis job")
    ap.add_argument("--ready-wait", type=float, default=300)
    ap.add_argument("--job-timeout", type=float, default=900)
    ap.add_argument("--no-sse", action="store_true")
    ap.add_argument("--evidence", type=Path, help="write smoke.json into this directory")
    ap.add_argument("--timeout", type=float, default=30)
    args = ap.parse_args()

    c = Client(args.base_url, args.insecure, args.timeout)
    r = Report()
    started = time.time()
    try:
        status, _, _ = c.request("GET", "/health/live")
        r.add("health_live", status == 200, status=status)
        wait_ready(c, r, args.ready_wait)

        status, headers, _ = c.request("GET", "/", headers={"Accept": "text/html"})
        r.add("web_root", status == 200 and "text/html" in headers.get("content-type", ""), status=status)
        status, _, _ = c.request("GET", "/metrics", headers={"Accept": "*/*"})
        r.add("metrics_not_public", status == 404, status=status)

        status, caps = c.json("GET", "/api/v1/capabilities")
        commit = ((caps or {}).get("build") or {}).get("commit")
        r.add("capabilities", status == 200 and isinstance(caps, dict), status=status, commit=commit,
              video_analysis=(caps or {}).get("video_analysis", {}).get("available"))
        if args.expect_commit:
            r.add("build_commit_matches", commit == args.expect_commit, expected=args.expect_commit, actual=commit)

        status, info = c.json("POST", "/api/v1/session", {})
        cookies = {ck.name for ck in c.jar}
        c.csrf = (info or {}).get("csrf_token")
        r.add("session", status in (200, 201) and bool(c.csrf) and "dd_session" in cookies,
              status=status, role=(info or {}).get("role"), cookies=sorted(cookies))
        status, _ = c.json("GET", "/api/v1/jobs")
        r.add("session_authenticated_read", status == 200, status=status)
        if not args.no_sse:
            check_sse(c, r, 25)
        if args.upload:
            upload_flow(c, r, args.upload, caps if isinstance(caps, dict) else None, args.job_timeout)
    except OSError as exc:
        r.add("connection", False, error=f"{type(exc).__name__}: {exc}")

    summary = {"base_url": args.base_url, "ok": r.ok, "started_unix": int(started),
               "elapsed_s": round(time.time() - started, 2), "checks": r.checks}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.evidence:
        args.evidence.mkdir(parents=True, exist_ok=True)
        (args.evidence / "smoke.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
