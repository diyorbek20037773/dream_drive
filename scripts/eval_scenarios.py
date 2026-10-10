#!/usr/bin/env python3
"""Run the synthetic LiteRoad scenario suite through the public API and save the resulting Evaluation JSON.

    make eval-scenarios        # runs this inside the backend test image on the compose network:
    python /opt/dreamdrive/scripts/eval_scenarios.py --base http://gateway:8080 --seeds 1,2,3 --out /evidence

Uses the same path as a browser: session cookie + CSRF header, POST /api/v1/evaluations, then polls
GET /api/v1/evaluations/{id}. The application computes every metric; this client only records what it returns.
Exit 0 only when the evaluation finished with state "succeeded". Stdlib only.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TERMINAL = {"succeeded", "failed", "interrupted"}


class Api:
    def __init__(self, base: str, timeout: float) -> None:
        if not base.startswith(("http://", "https://")):
            raise ValueError("base URL must be http:// or https://")
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.csrf = ""

    def call(self, method: str, path: str, payload: Any = None) -> tuple[int, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if method != "GET" and self.csrf:
            headers["X-CSRF-Token"] = self.csrf
        # The scheme is restricted to http(s) in __init__; paths are fixed API routes.
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)  # noqa: S310
        try:
            with self.opener.open(req, timeout=self.timeout) as resp:
                body = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            body, status = exc.read(), exc.code
        try:
            return status, json.loads(body) if body else None
        except ValueError:
            return status, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://gateway:8080")
    ap.add_argument("--seeds", default="1,2,3", help="comma-separated integer seeds")
    ap.add_argument("--out", type=Path, required=True, help="directory for evaluation-<id>.json")
    ap.add_argument("--timeout", type=float, default=1800, help="max seconds to wait for the suite")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    api = Api(args.base, 30)
    status, session = api.call("POST", "/api/v1/session", {})
    if status not in (200, 201) or not session or not session.get("csrf_token"):
        print(f"session failed: HTTP {status}", file=sys.stderr)
        return 1
    api.csrf = session["csrf_token"]

    status, evaluation = api.call("POST", "/api/v1/evaluations", {"kind": "synthetic_scenario_suite", "seeds": seeds})
    if status not in (200, 201, 202) or not evaluation or not evaluation.get("id"):
        print(f"could not start the scenario suite: HTTP {status} {json.dumps(evaluation)}", file=sys.stderr)
        return 1
    eval_id = evaluation["id"]
    print(f"evaluation {eval_id} started for seeds {seeds}", file=sys.stderr)

    deadline = time.monotonic() + args.timeout
    state = evaluation.get("state")
    while state not in TERMINAL:
        if time.monotonic() > deadline:
            print(f"timed out after {args.timeout:.0f}s in state {state}", file=sys.stderr)
            return 1
        time.sleep(5)
        status, evaluation = api.call("GET", f"/api/v1/evaluations/{eval_id}")
        if status != 200 or not evaluation:
            print(f"poll failed: HTTP {status}", file=sys.stderr)
            return 1
        state = evaluation.get("state")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"evaluation-{eval_id}.json"
    out.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
    print(f"state={state} sample_count={evaluation.get('sample_count')} -> {out}", file=sys.stderr)
    for name, metric in sorted((evaluation.get("metrics") or {}).items()):
        print(f"  {name}: value={metric.get('value')} unit={metric.get('unit')} status={metric.get('status')} "
              f"n={metric.get('sample_count')}", file=sys.stderr)
    return 0 if state == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
