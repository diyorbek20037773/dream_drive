"""Print the detector-accuracy results of an Evaluation JSON as a Markdown table (values copied, never recomputed).

uv run --project tools/eval python -I tools/eval/results_table.py artifacts/evaluations/<id>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CLASSES = ["person", "bicycle", "motorcycle", "bus", "truck", "traffic_light", "stop_sign", "vehicle", "vehicle_any"]


def _fmt(metric: dict | None) -> str:
    if not metric:
        return "-"
    if metric["status"] != "measured":
        return metric["status"]
    return f"{metric['value']:.4f}"


def main() -> int:
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    m = doc["metrics"]
    per_class = doc.get("details", {}).get("per_class", {})
    print(f"Evaluation `{doc['id']}` ({doc['kind']}, source `{doc['source']}`, {doc['sample_count']} images)\n")
    print("| Class | GT instances | Images with GT | Detections scored | AP50 | AP50:95 | Recall@0.50 |")
    print("|---|---|---|---|---|---|---|")
    for c in CLASSES:
        pc = per_class.get(c, {})
        n = m.get(f"ap50_{c}", {}).get("sample_count", "-")
        print(
            f"| {c} | {n} | {pc.get('images_with_gt', '-')} | {pc.get('detections_scored', '-')} | "
            f"{_fmt(m.get(f'ap50_{c}'))} | {_fmt(m.get(f'ap50_95_{c}'))} | {_fmt(m.get(f'recall50_{c}'))} |"
        )
    print()
    for k in (
        "map50_enabled_classes",
        "map50_95_enabled_classes",
        "inference_latency_p50_ms",
        "inference_latency_p95_ms",
        "request_roundtrip_p50_ms",
        "request_roundtrip_p95_ms",
    ):
        if k in m:
            print(f"- `{k}`: {_fmt(m[k])} ({m[k]['unit']}, n={m[k]['sample_count']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
