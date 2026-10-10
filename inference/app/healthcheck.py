"""Container healthcheck without curl: ``python -m app.healthcheck [--ready]`` exits 0 on HTTP 200."""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request


def main(argv: list[str]) -> int:
    path = "/health/ready" if "--ready" in argv else "/health/live"
    port = os.environ.get("INFERENCE_PORT", "8001")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as resp:  # noqa: S310 - fixed URL
            return 0 if resp.status == 200 else 1
    except (urllib.error.URLError, OSError):
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
