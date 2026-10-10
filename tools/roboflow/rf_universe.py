"""Read-only Roboflow Universe helper for the DreamDrive lane-marking dataset audit (docs/EVALUATION_LANES.md).

Secret handling: the API key is loaded inside this process from the official SDK login store
(``roboflow.login()`` -> ``%USERPROFILE%/roboflow/config.json``) via ``roboflow.config.load_roboflow_api_key``. It is
only sent to ``https://api.roboflow.com`` as a query parameter, never printed, logged or written. Every string this
tool prints or writes is passed through ``_scrub`` (removes the key and any ``api_key=`` / signed-URL query strings).
Export download links are signed URLs: they are used once in memory and never stored.

The tool never writes to a Roboflow workspace: only GET requests (search, project/version metadata, export link).

Usage (``py -3.11 -I tools/roboflow/rf_universe.py ...``)::

    search  --out <json> "lane segmentation" "lane lines" ...
    info    --out <json> <workspace>/<project>[/<version>] ...
    export  --dest <dir> <workspace>/<project>/<version> --format coco-segmentation
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import requests

API = "https://api.roboflow.com"
_URL_QS = re.compile(r"(https?://[^\s\"']+?)\?[^\s\"']*")
_KEY_QS = re.compile(r"(api_key=)[^&\s\"']+")


def _key() -> str:
    from roboflow.config import load_roboflow_api_key

    key = load_roboflow_api_key(None)
    if not key:
        raise SystemExit('no Roboflow login found; run: py -3.11 -c "import roboflow; roboflow.login()"')
    return str(key)


def _scrub(obj: Any, key: str) -> Any:
    if isinstance(obj, str):
        s = obj.replace(key, "<redacted>") if key else obj
        s = _KEY_QS.sub(r"\1<redacted>", s)
        return s
    if isinstance(obj, dict):
        return {k: _scrub(v, key) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v, key) for v in obj]
    return obj


def _get(path: str, key: str, params: dict[str, Any] | None = None, timeout: float = 60.0) -> Any:
    p = dict(params or {})
    p["api_key"] = key
    try:
        r = requests.get(f"{API}/{path.lstrip('/')}", params=p, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(_scrub(f"{type(exc).__name__}: {exc}", key)) from None
    if r.status_code != 200:
        raise RuntimeError(_scrub(f"HTTP {r.status_code} for /{path}: {r.text[:300]}", key))
    return r.json()


def _write(out: Path | None, data: Any) -> None:
    text = json.dumps(data, indent=1, ensure_ascii=False)
    if out is None:
        print(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")


def cmd_search(args: argparse.Namespace) -> int:
    key = _key()
    results: dict[str, Any] = {}
    for q in args.queries:
        pages = []
        for page in range(1, args.pages + 1):
            try:
                data = _get("universe/search", key, {"q": q, "limit": args.limit, "page": page, "type": "dataset"})
            except RuntimeError as exc:
                pages.append({"error": str(exc)})
                break
            pages.append(data)
            time.sleep(0.3)
        results[q] = pages
    _write(args.out, _scrub(results, key))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    key = _key()
    out: dict[str, Any] = {}
    for ref in args.refs:
        try:
            out[ref] = _get(ref, key)
        except RuntimeError as exc:
            out[ref] = {"error": str(exc)}
        time.sleep(0.3)
    _write(args.out, _scrub(out, key))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Request an export of one version and download the archive to ``--dest`` (the signed link is not stored)."""
    key = _key()
    ref = args.ref.strip("/")
    if ref.count("/") != 2:
        raise SystemExit("ref must be <workspace>/<project>/<version>")
    data: Any = None
    for _ in range(60):
        data = _get(f"{ref}/{args.format}", key)
        link = (data.get("export") or {}).get("link") if isinstance(data, dict) else None
        if link:
            break
        time.sleep(5)
    else:
        raise SystemExit("export link not ready after 5 minutes")
    if args.probe:
        exp = {k: v for k, v in (data.get("export") or {}).items() if k != "link"}
        head = requests.head(link, timeout=60, allow_redirects=True)
        del link
        print(json.dumps(_scrub({"export": exp, "content_length": head.headers.get("content-length")}, key), indent=1))
        return 0
    if not args.dest:
        raise SystemExit("--dest is required unless --probe is given")
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{ref.replace('/', '__')}__{args.format}.zip"
    with requests.get(link, stream=True, timeout=600) as r:
        if r.status_code != 200:
            raise SystemExit(f"download failed: HTTP {r.status_code}")
        with zip_path.open("wb") as fh:
            shutil.copyfileobj(r.raw, fh, length=1 << 20)
    del link
    meta = _scrub({k: v for k, v in data.items() if k != "export"}, key) if isinstance(data, dict) else {}
    print(json.dumps({"zip": str(zip_path), "bytes": zip_path.stat().st_size, "meta_keys": sorted(meta)}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rf_universe")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("queries", nargs="+")
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--pages", type=int, default=1)
    s.add_argument("--out", type=Path, default=None)
    i = sub.add_parser("info")
    i.add_argument("refs", nargs="+")
    i.add_argument("--out", type=Path, default=None)
    e = sub.add_parser("export")
    e.add_argument("ref")
    e.add_argument("--format", default="coco-segmentation")
    e.add_argument("--dest", default=None)
    e.add_argument("--probe", action="store_true", help="print export metadata and archive size, do not download")
    args = p.parse_args(argv)
    return {"search": cmd_search, "info": cmd_info, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
