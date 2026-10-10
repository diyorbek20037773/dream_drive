"""Self-test of the smoke CLIENT (deploy/lib/smoke.py) against a tiny local stub of the gateway contract.

This verifies the client's own logic (cookies, CSRF header, polling, Range check, failure detection). It says
nothing about the real application; the real check is running deploy/smoke.sh against the actual stack.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SMOKE = Path(__file__).resolve().parents[1] / "lib" / "smoke.py"
COMMIT = "1" * 40
CSRF = "csrf-test-token"


class Stub(BaseHTTPRequestHandler):
    metrics_status = 404
    job_polls = 0

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def _send(self, status: int, body: bytes = b"", ctype: str = "application/json", **headers: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers.items():
            self.send_header(k.replace("_", "-"), v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: object, **headers: str) -> None:
        self._send(status, json.dumps(obj).encode(), **headers)

    def _authed(self) -> bool:
        return "dd_session=abc" in (self.headers.get("Cookie") or "")

    def do_GET(self) -> None:  # noqa: N802
        p = self.path
        if p in ("/health/live", "/health/ready"):
            return self._json(200, {"status": "ok"})
        if p == "/":
            return self._send(200, b"<!doctype html>", "text/html; charset=utf-8")
        if p == "/metrics":
            return self._send(type(self).metrics_status, b"")
        if p == "/api/v1/capabilities":
            return self._json(200, {"build": {"commit": COMMIT}, "video_analysis": {"available": True}})
        if p == "/api/v1/jobs":
            return self._json(200 if self._authed() else 401, {"items": []})
        if p == "/api/v1/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": ping\n\n")
            self.wfile.flush()
            return None
        if p == "/api/v1/jobs/j1":
            type(self).job_polls += 1
            state = "succeeded" if type(self).job_polls > 1 else "running"
            return self._json(200, {"id": "j1", "state": state, "frames_done": 3})
        if p == "/api/v1/jobs/j1/results":
            return self._json(200, {"artifacts": [
                {"name": "annotated.mp4", "url": "/api/v1/jobs/j1/artifacts/annotated.mp4"},
                {"name": "report.json", "url": "/api/v1/jobs/j1/artifacts/report.json"}]})
        if p == "/api/v1/jobs/j1/artifacts/annotated.mp4" and self.headers.get("Range") == "bytes=0-99":
            return self._send(206, b"x" * 100, "video/mp4", Content_Range="bytes 0-99/1000")
        return self._json(404, {"code": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        if self.path == "/api/v1/session":
            return self._json(200, {"csrf_token": CSRF, "role": "anonymous"},
                              Set_Cookie="dd_session=abc; Path=/; HttpOnly")
        if self.path == "/api/v1/videos":
            ok = self._authed() and self.headers.get("X-CSRF-Token") == CSRF and b'name="file"' in body
            return self._json(202 if ok else 403, {"id": "j1", "state": "queued"})
        return self._json(404, {"code": "not_found"})


class SmokeClientTests(unittest.TestCase):
    def setUp(self) -> None:
        Stub.metrics_status = 404
        Stub.job_polls = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def smoke(self, *extra: str) -> tuple[int, dict]:
        proc = subprocess.run([sys.executable, str(SMOKE), self.base, "--ready-wait", "5", *extra],
                              capture_output=True, text=True, timeout=120, check=False)
        return proc.returncode, json.loads(proc.stdout)

    def test_all_checks_pass_with_upload(self) -> None:
        clip = Path(self.tmp.name) / "clip.mp4"
        clip.write_bytes(b"\x00" * 64)
        code, summary = self.smoke("--expect-commit", COMMIT, "--upload", str(clip), "--evidence", self.tmp.name)
        self.assertEqual(code, 0, summary)
        names = {c["check"] for c in summary["checks"]}
        self.assertTrue({"session", "sse_stream", "video_job_succeeded", "annotated_mp4_range"} <= names)
        self.assertTrue((Path(self.tmp.name) / "smoke.json").is_file())
        self.assertNotIn(CSRF, json.dumps(summary))

    def test_public_metrics_fails_the_smoke(self) -> None:
        Stub.metrics_status = 200
        code, summary = self.smoke("--no-sse")
        self.assertEqual(code, 1)
        failed = [c["check"] for c in summary["checks"] if not c["ok"]]
        self.assertEqual(failed, ["metrics_not_public"])

    def test_wrong_commit_fails_the_smoke(self) -> None:
        code, summary = self.smoke("--no-sse", "--expect-commit", "2" * 40)
        self.assertEqual(code, 1)
        self.assertIn("build_commit_matches", [c["check"] for c in summary["checks"] if not c["ok"]])

    def test_unreachable_gateway_fails(self) -> None:
        self.base = "http://127.0.0.1:9"
        code, summary = self.smoke("--no-sse")
        self.assertEqual(code, 1)
        self.assertFalse(summary["ok"])


if __name__ == "__main__":
    unittest.main()
