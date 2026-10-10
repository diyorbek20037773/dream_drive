#!/usr/bin/env python3
"""Download licensed test fixtures listed in tests/fixtures/manifest.json and derive short H.264 clips.

Videos are never committed. They are cached in tests/fixtures/cache/ (git-ignored):
  1. download `source.url` with a timeout into `<name>.part`, verify SHA1 (as published by the Wikimedia Commons API)
     and SHA256 (recorded by DreamDrive on first verified download), then atomically rename;
  2. when `derived` is present, decode the source with PyAV and write a <= 12 s, <= 1280x720 H.264 (libx264, yuv420p,
     +faststart, no audio) MP4 clip, keeping the source frame timing, and verify its SHA256.

Usage (from the repository root, with the backend virtualenv or any Python that has `av` and `numpy`):
  python scripts/fetch_fixtures.py                 # download + derive + verify
  python scripts/fetch_fixtures.py --check         # verify cached files only (no network); non-zero if missing/bad
  python scripts/fetch_fixtures.py --record        # maintainer only: write observed SHA256 values into the manifest

Fixtures under attribution licences (e.g. CC BY 3.0) carry `source.license_note` with the required attribution and
the list of changes; it is printed for every fixture and mirrored in docs/DATA_SOURCES.md.

The source SHA256 is always enforced: it is the provenance guarantee. The derived clip is reproducible byte for byte
with the pinned PyAV/libx264 build on one machine, but CI runners showed different bytes for the same source and
settings (CPU-dependent decode/convert/encode code paths). When the derived bytes differ from the manifest reference,
the clip is accepted only if its decoded properties (frame count, width, height, frame rate) match the manifest
exactly; the SHA256 observed on this machine is then recorded in <cache>/derived-observed.json so tests can verify
that the cached clip is the one this script produced. Any property mismatch fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "manifest.json"
DEFAULT_CACHE = REPO_ROOT / "tests" / "fixtures" / "cache"
USER_AGENT = "DreamDriveFixtureFetcher/1.0 (https://github.com/diyorbek20037773/dream_drive)"
MAX_DERIVED_SECONDS = 12.0
MAX_DERIVED_W, MAX_DERIVED_H = 1280, 720


def file_hashes(path: Path) -> tuple[str, str]:
    sha1, sha256 = hashlib.sha1(usedforsecurity=False), hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            sha1.update(block)
            sha256.update(block)
    return sha1.hexdigest(), sha256.hexdigest()


def download(url: str, dest: Path, timeout_s: float, max_bytes: int) -> None:
    if not url.startswith("https://"):
        raise SystemExit(f"refusing non-https fixture URL: {url}")
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 - https enforced above
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(part, "wb") as out:  # noqa: S310
        total = 0
        while chunk := resp.read(1 << 20):
            total += len(chunk)
            if total > max_bytes:
                raise SystemExit(f"download exceeded expected size {max_bytes} bytes: {url}")
            out.write(chunk)
    print(f"  downloaded {total} bytes in {time.monotonic() - started:.1f} s")
    os.replace(part, dest)


def verify_source(entry: dict[str, Any], path: Path, record: bool) -> bool:
    src = entry["source"]
    sha1, sha256 = file_hashes(path)
    ok = True
    if src.get("size_bytes") is not None and path.stat().st_size != src["size_bytes"]:
        print(f"  size mismatch: {path.stat().st_size} != {src['size_bytes']}")
        ok = False
    if src.get("sha1") and sha1 != src["sha1"]:
        print(f"  SHA1 mismatch vs Commons: {sha1} != {src['sha1']}")
        ok = False
    if src.get("sha256"):
        if sha256 != src["sha256"]:
            print(f"  SHA256 mismatch: {sha256} != {src['sha256']}")
            ok = False
    elif record and ok:
        src["sha256"] = sha256
        print(f"  recorded source SHA256 {sha256}")
    elif not record:
        print("  manifest has no source SHA256 (run --record after verifying the license)")
        ok = False
    print(f"  source sha1={sha1} sha256={sha256} -> {'OK' if ok else 'FAILED'}")
    return ok


def derive_clip(source: Path, dest: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import av  # local import: only needed for derivation
    import numpy as np

    start_s = float(spec["start_s"])
    duration_s = float(spec["duration_s"])
    if duration_s <= 0 or duration_s > MAX_DERIVED_SECONDS:
        raise SystemExit(f"derived duration must be in (0, {MAX_DERIVED_SECONDS}] s")
    max_w = min(int(spec.get("max_width", MAX_DERIVED_W)), MAX_DERIVED_W)
    max_h = min(int(spec.get("max_height", MAX_DERIVED_H)), MAX_DERIVED_H)
    part = dest.with_name(dest.name + ".part")
    frames = 0
    with av.open(str(source)) as src:
        vs = src.streams.video[0]
        rate = vs.average_rate or vs.guessed_rate or Fraction(30, 1)
        rate = Fraction(rate).limit_denominator(1001)
        w, h = vs.codec_context.width, vs.codec_context.height
        scale = min(1.0, max_w / w, max_h / h)
        out_w, out_h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
        with av.open(str(part), "w", format="mp4", options={"movflags": "+faststart"}) as dst:
            ost = dst.add_stream("libx264", rate=rate)
            ost.width, ost.height, ost.pix_fmt = out_w, out_h, "yuv420p"
            ost.time_base = Fraction(1, 1) / rate
            ost.codec_context.time_base = Fraction(1, 1) / rate
            ost.codec_context.options = {
                "crf": str(spec.get("crf", 20)),
                "preset": str(spec.get("preset", "medium")),
                "threads": "1",
            }
            first_t: float | None = None
            for frame in src.decode(vs):
                t = frame.time
                if t is None or t < start_s:
                    continue
                if first_t is None:
                    first_t = t
                if t - first_t >= duration_s - 1e-9:
                    break
                img = frame.to_ndarray(format="rgb24")
                if (img.shape[1], img.shape[0]) != (out_w, out_h):
                    import cv2

                    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
                out = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
                out.pts = round((t - first_t) * rate)
                out.time_base = ost.codec_context.time_base  # stream time base changes after the header
                for packet in ost.encode(out):
                    dst.mux(packet)
                frames += 1
            for packet in ost.encode(None):
                dst.mux(packet)
    if frames == 0:
        part.unlink(missing_ok=True)
        raise SystemExit("derived clip has no frames (start_s beyond the source?)")
    os.replace(part, dest)
    return {"frames": frames, "width": out_w, "height": out_h, "fps": float(rate), "pyav_version": av.__version__}


def clip_properties(path: Path) -> dict[str, Any]:
    """Decoded frame count, size and average frame rate of a derived clip."""
    import av  # local import: only needed for derived clips

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
        rate = stream.average_rate
        return {"frames": frames, "width": stream.width, "height": stream.height, "fps": float(rate) if rate else 0.0}


def record_observed(cache_dir: Path, file_name: str, sha256: str) -> None:
    """Remember the derived-clip SHA256 produced on this machine (tests compare the cached file against it)."""
    index = cache_dir / "derived-observed.json"
    data: dict[str, str] = {}
    if index.exists():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    data[file_name] = sha256
    index.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--check", action="store_true", help="verify cached files only; never download")
    ap.add_argument("--record", action="store_true", help="record observed SHA256 values into the manifest")
    ap.add_argument("--allow-derived-mismatch", action="store_true", help="accept any derived bytes (maintainers only)")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for entry in manifest["fixtures"]:
        print(f"[{entry['id']}] {entry['source']['title']} ({entry['source']['license']})")
        if entry["source"].get("license_note"):
            print(f"  attribution: {entry['source']['license_note']}")
        src_path = args.cache_dir / entry["source"]["file_name"]
        if not src_path.exists():
            if args.check:
                print("  missing source file (run without --check to download)")
                failures += 1
                continue
            limit = int(entry["source"].get("size_bytes") or 0) or 500 * 1024 * 1024
            download(entry["source"]["url"], src_path, args.timeout, limit)
        if not verify_source(entry, src_path, args.record):
            failures += 1
            continue
        spec = entry.get("derived")
        if not spec:
            continue
        dst_path = args.cache_dir / spec["file_name"]
        if not dst_path.exists():
            if args.check:
                print("  missing derived clip")
                failures += 1
                continue
            info = derive_clip(src_path, dst_path, spec)
            print(f"  derived {dst_path.name}: {info}")
            if args.record:
                spec.update(info)
        _, sha256 = file_hashes(dst_path)
        expected = spec.get("sha256")
        if expected and sha256 != expected:
            props = clip_properties(dst_path)
            wanted = {k: spec.get(k) for k in ("frames", "width", "height")}
            fps_ok = abs(props["fps"] - float(spec.get("fps", 0))) < 1e-3
            same = fps_ok and all(props[k] == v for k, v in wanted.items())
            if same:
                print(
                    f"  derived bytes differ from the reference ({expected[:12]}...) on this machine; source SHA256 "
                    f"verified and decoded properties match {props}"
                )
                record_observed(args.cache_dir, spec["file_name"], sha256)
            elif args.allow_derived_mismatch:
                print(f"  derived SHA256 mismatch accepted by --allow-derived-mismatch: {sha256} != {expected}")
            else:
                print(f"  derived clip mismatch: sha256 {sha256} != {expected} and properties {props} != {wanted}")
                failures += 1
        elif expected:
            record_observed(args.cache_dir, spec["file_name"], sha256)
        elif not expected:
            if args.record:
                spec["sha256"] = sha256
                print(f"  recorded derived SHA256 {sha256}")
            else:
                print("  manifest has no derived SHA256")
                failures += 1
        print(f"  derived sha256={sha256}")
    if args.record:
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"manifest updated: {args.manifest}")
    print("FAILED" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
