"""Cross-check DreamDrive's NumPy COCO AP (backend/app/evaluation/coco_ap.py) with the reference pycocotools COCOeval.

Inputs are the raw predictions written by ``app.evaluation.detector_accuracy`` (``<id>.predictions.json``), the
committed subset manifest and the verified COCO annotation archive. Ground truth for each DreamDrive eval class is
rebuilt independently here from the archive (merged categories such as vehicle = car+bus+truck become one synthetic
category), predictions are fed to ``COCO.loadRes`` in file order, and per-class AP50, AP50:95 and recall from
``COCOeval`` (iouType bbox, areaRng all, maxDets 100) are compared with the NumPy values stored in the predictions
file. The result (including the maximum absolute difference) is written as JSON.

    uv run --project tools/eval python -I tools/eval/crosscheck.py --predictions <file> --manifest <manifest> \
        --annotations datasets/raw/coco/annotations_trainval2017.zip --out artifacts/qa/spike/<id>.crosscheck.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MEMBER = "annotations/instances_val2017.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_annotations(zip_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    src = manifest["source"]
    archive_sha = _sha256(zip_path)
    if archive_sha != src["archive_sha256"]:
        raise SystemExit(f"archive sha256 mismatch: {archive_sha}")
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(MEMBER)
    member_sha = hashlib.sha256(data).hexdigest()
    if member_sha != src["member_sha256"]:
        raise SystemExit(f"member sha256 mismatch: {member_sha}")
    return json.loads(data)


def _xyxy_to_xywh(b: list[float]) -> list[float]:
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    args = ap.parse_args()

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if hashlib.sha256(manifest_bytes).hexdigest() != preds["manifest_sha256"]:
        raise SystemExit("predictions were produced from a different manifest")
    coco = _load_annotations(args.annotations, manifest)
    cat_by_name = {c["name"]: c["id"] for c in coco["categories"]}
    images_by_id = {im["id"]: im for im in coco["images"]}
    image_ids = sorted(int(i) for i in preds["image_ids"])
    wanted = set(image_ids)

    classes = [c for c in preds["eval_classes"] if c["supported_by_service"]]
    cat_ids = {c["name"]: k + 1 for k, c in enumerate(classes)}  # synthetic category ids for merged classes
    gt_anns: list[dict[str, Any]] = []
    next_id = 1
    for c in classes:
        src_cats = {cat_by_name[n] for n in c["gt_coco_categories"]}
        for ann in sorted(coco["annotations"], key=lambda a: a["id"]):
            if ann["image_id"] in wanted and ann["category_id"] in src_cats:
                gt_anns.append(
                    {
                        "id": next_id,
                        "image_id": ann["image_id"],
                        "category_id": cat_ids[c["name"]],
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                )
                next_id += 1
    gt_dict = {
        "images": [
            {"id": i, "width": images_by_id[i]["width"], "height": images_by_id[i]["height"]} for i in image_ids
        ],
        "categories": [{"id": cat_ids[c["name"]], "name": c["name"]} for c in classes],
        "annotations": gt_anns,
    }
    dt_list: list[dict[str, Any]] = []
    for item in preds["predictions"]:
        for det in item["detections"]:
            for c in classes:
                if det["class_name"] in c["pred_class_names"]:
                    dt_list.append(
                        {
                            "image_id": item["image_id"],
                            "category_id": cat_ids[c["name"]],
                            "bbox": _xyxy_to_xywh(det["xyxy_pixels"]),
                            "score": det["score"],
                        }
                    )
    if not dt_list:
        raise SystemExit("no predictions for supported classes")

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        dt = gt.loadRes(dt_list)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.imgIds = image_ids
        ev.params.catIds = [cat_ids[c["name"]] for c in classes]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    precision = ev.eval["precision"]  # [T, R, K, A, M]
    recall = ev.eval["recall"]  # [T, K, A, M]
    a_all = list(ev.params.areaRngLbl).index("all")
    m100 = list(ev.params.maxDets).index(100)

    rows: dict[str, Any] = {}
    max_diff = 0.0
    for k, c in enumerate(classes):
        p = precision[:, :, k, a_all, m100]
        r = recall[:, k, a_all, m100]
        if (p > -1).any():
            ref = {
                "ap50_95": float(np.mean(p[p > -1])),
                "ap50": float(np.mean(p[0][p[0] > -1])),
                "ap75": float(np.mean(p[5][p[5] > -1])),
                "recall50": float(r[0]),
                "recall50_95": float(np.mean(r[r > -1])),
            }
        else:
            ref = dict.fromkeys(("ap50_95", "ap50", "ap75", "recall50", "recall50_95"))
        ours = preds["numpy_results"][c["name"]]
        diffs = {}
        for key, val in ref.items():
            o = ours.get(key)
            if val is None or o is None:
                diffs[key] = None if (val is None and o is None) else "mismatch_undefined"
                if diffs[key] is not None:
                    max_diff = float("inf")
                continue
            d = abs(float(o) - val)
            diffs[key] = d
            max_diff = max(max_diff, d)
        rows[c["name"]] = {"pycocotools": ref, "numpy": {k2: ours.get(k2) for k2 in ref}, "abs_diff": diffs}

    report = {
        "schema_version": "1.0",
        "kind": "coco_ap_crosscheck",
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluation_id": preds["evaluation_id"],
        "run_label": preds.get("run_label"),
        "predictions_file": args.predictions.name,
        "predictions_sha256": _sha256(args.predictions),
        "manifest_sha256": preds["manifest_sha256"],
        "images": len(image_ids),
        "gt_annotations": len(gt_anns),
        "detections_fed": len(dt_list),
        "pycocotools_version": getattr(pycocotools, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "python": sys.version.split()[0],
        "classes": rows,
        "max_abs_diff": max_diff,
        "tolerance": args.tolerance,
        "passed": max_diff <= args.tolerance,
        "pycocotools_summary_stdout": log.getvalue().splitlines()[-12:],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("images", "detections_fed", "max_abs_diff", "passed")}, indent=2))
    for name, row in rows.items():
        print(name, {k: (row["pycocotools"][k], row["numpy"][k]) for k in ("ap50", "ap50_95")})
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
