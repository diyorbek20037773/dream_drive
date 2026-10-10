"""Randomised agreement test: backend/app/evaluation/coco_ap.py vs pycocotools COCOeval on synthetic boxes.

Generates many images with ground truth (including iscrowd regions, duplicate/overlapping boxes, tied scores, more
than 100 detections per image) and jittered/false-positive detections, then compares per-class AP50, AP75, AP50:95,
recall50 and recall50:95. Exits non-zero when any absolute difference exceeds the tolerance.

    uv run --project tools/eval python -I tools/eval/random_crosscheck.py --trials 20 --out <json>
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]


def _load_coco_ap() -> Any:
    path = ROOT / "backend" / "app" / "evaluation" / "coco_ap.py"
    spec = importlib.util.spec_from_file_location("dd_coco_ap", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dd_coco_ap"] = mod
    spec.loader.exec_module(mod)
    return mod


def _trial(rng: np.random.Generator, cap: Any, n_images: int, n_cats: int) -> dict[str, float]:
    images, gts, dts = [], [], []
    ann_id = 1
    for img_id in range(1, n_images + 1):
        w, h = int(rng.integers(200, 800)), int(rng.integers(200, 800))
        images.append({"id": img_id, "width": w, "height": h})
        for cat in range(1, n_cats + 1):
            n_gt = int(rng.integers(0, 8))
            for _ in range(n_gt):
                bw, bh = rng.uniform(5, w / 2), rng.uniform(5, h / 2)
                x, y = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
                crowd = int(rng.random() < 0.08)
                gts.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh * rng.uniform(0.5, 1.0),
                        "iscrowd": crowd,
                    }
                )
                ann_id += 1
                for _ in range(int(rng.integers(0, 3))):  # jittered true/duplicate detections
                    j = rng.normal(0, 0.15, 4) * [bw, bh, bw, bh]
                    dts.append(
                        {
                            "image_id": img_id,
                            "category_id": cat,
                            "bbox": [x + j[0], y + j[1], max(1.0, bw + j[2]), max(1.0, bh + j[3])],
                            "score": float(np.round(rng.random(), 2)),  # rounding creates ties
                        }
                    )
            n_fp = int(rng.integers(0, 6)) if rng.random() > 0.05 else 130  # sometimes exceed maxDets=100
            for _ in range(n_fp):
                bw, bh = rng.uniform(5, w / 3), rng.uniform(5, h / 3)
                dts.append(
                    {
                        "image_id": img_id,
                        "category_id": cat,
                        "bbox": [rng.uniform(0, w - bw), rng.uniform(0, h - bh), bw, bh],
                        "score": float(np.round(rng.random(), 2)),
                    }
                )
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": gts, "categories": [{"id": c} for c in range(1, n_cats + 1)]}
        gt.createIndex()
        dt = gt.loadRes(dts)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
    prec, rec = ev.eval["precision"], ev.eval["recall"]
    max_diff = 0.0
    for k, cat in enumerate(ev.params.catIds):
        samples = []
        for im in images:
            g = [a for a in gts if a["image_id"] == im["id"] and a["category_id"] == cat]
            d = [a for a in dts if a["image_id"] == im["id"] and a["category_id"] == cat]
            samples.append(
                cap.ImageSample.build(
                    im["id"],
                    [a["bbox"] for a in g],
                    [a["iscrowd"] for a in g],
                    [a["bbox"] for a in d],
                    [a["score"] for a in d],
                )
            )
        ours = cap.evaluate_class(samples)
        p = prec[:, :, k, 0, 2]
        r = rec[:, k, 0, 2]
        if not (p > -1).any():
            if ours.ap50 is not None:
                return {"max_abs_diff": float("inf")}
            continue
        ref = [p.mean(), p[0].mean(), p[5].mean(), r[0], r.mean()]
        got = [ours.ap50_95, ours.ap50, ours.ap75, ours.recall50, ours.recall50_95]
        max_diff = max(max_diff, max(abs(a - b) for a, b in zip(ref, got, strict=True)))
    return {"max_abs_diff": float(max_diff), "gt": len(gts), "dt": len(dts)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--tolerance", type=float, default=1e-9)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cap = _load_coco_ap()
    rng = np.random.default_rng(args.seed)
    results = [_trial(rng, cap, n_images=int(rng.integers(3, 25)), n_cats=3) for _ in range(args.trials)]
    worst = max(r["max_abs_diff"] for r in results)
    report = {
        "kind": "coco_ap_random_crosscheck",
        "seed": args.seed,
        "trials": args.trials,
        "total_gt": sum(r.get("gt", 0) for r in results),
        "total_dt": sum(r.get("dt", 0) for r in results),
        "max_abs_diff": worst,
        "tolerance": args.tolerance,
        "passed": worst <= args.tolerance,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
