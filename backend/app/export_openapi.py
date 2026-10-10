"""Export the OpenAPI document as stable JSON (sorted keys, 2-space indent, LF, trailing newline).

Usage (from backend/):
    python -m app.export_openapi                      # print to stdout
    python -m app.export_openapi --output PATH        # write PATH
    python -m app.export_openapi --check PATH         # exit 1 if PATH differs from the current schema
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def openapi_document() -> dict[str, Any]:
    from app.main import create_app

    return create_app().openapi()


def render(document: dict[str, Any]) -> str:
    return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", "-o", type=Path, help="write the schema to this file")
    group.add_argument("--check", type=Path, help="compare this file with the current schema")
    args = parser.parse_args(argv)

    text = render(openapi_document())
    if args.check is not None:
        try:
            committed = json.loads(args.check.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"cannot read {args.check}: {exc}", file=sys.stderr)
            return 1
        if committed != json.loads(text):
            print(f"{args.check} is stale; regenerate with scripts/gen-types.sh", file=sys.stderr)
            return 1
        return 0
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # newline="" keeps LF on Windows hosts too, so the committed file is byte-identical across platforms.
        with args.output.open("w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        return 0
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
