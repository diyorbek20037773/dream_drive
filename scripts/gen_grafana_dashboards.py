#!/usr/bin/env python3
"""Generate the provisioned Grafana dashboards in ops/grafana/dashboards/ (stdlib only).

The JSON files are committed; `make ops-check` regenerates them into a temp dir and fails on drift, so the
dashboards stay reviewable as code. Metric and label names follow docs/OBSERVABILITY.md.

    python scripts/gen_grafana_dashboards.py [--out ops/grafana/dashboards] [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DS = {"type": "prometheus", "uid": "dd-prometheus"}
ROOT = Path(__file__).resolve().parents[1]


class Board:
    """Accumulates panels on Grafana's 24-column grid, left to right, top to bottom."""

    def __init__(self, uid: str, title: str, description: str) -> None:
        self.uid = uid
        self.title = title
        self.description = description
        self.panels: list[dict[str, Any]] = []
        self._x = 0
        self._y = 0
        self._row_h = 0

    def _place(self, w: int, h: int) -> dict[str, int]:
        if self._x + w > 24:
            self._x = 0
            self._y += self._row_h
            self._row_h = 0
        pos = {"x": self._x, "y": self._y, "w": w, "h": h}
        self._x += w
        self._row_h = max(self._row_h, h)
        return pos

    def add(self, panel: dict[str, Any], w: int, h: int) -> None:
        panel["id"] = len(self.panels) + 1
        panel["gridPos"] = self._place(w, h)
        self.panels.append(panel)

    def to_json(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": ["dreamdrive"],
            "timezone": "utc",
            "editable": False,
            "graphTooltip": 1,
            "refresh": "30s",
            "schemaVersion": 39,
            "version": 1,
            "time": {"from": "now-6h", "to": "now"},
            "templating": {"list": []},
            "annotations": {"list": []},
            "links": [],
            "panels": self.panels,
        }


def target(expr: str, legend: str = "", ref: str = "A", instant: bool = False, fmt: str | None = None) -> dict:
    t: dict[str, Any] = {
        "refId": ref,
        "datasource": DS,
        "expr": expr,
        "legendFormat": legend,
        "range": not instant,
        "instant": instant,
    }
    if fmt:
        t["format"] = fmt
    return t


def thresholds(*steps: tuple[str, float | None]) -> dict:
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for c, v in steps]}


def timeseries(title: str, targets: list[dict], unit: str, description: str, warn: float | None = None,
               stack: bool = False) -> dict:
    defaults: dict[str, Any] = {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 10, "showPoints": "never"}}
    if stack:
        defaults["custom"]["stacking"] = {"mode": "normal", "group": "A"}
    if warn is not None:
        defaults["thresholds"] = thresholds(("green", None), ("red", warn))
        defaults["custom"]["thresholdsStyle"] = {"mode": "line"}
    return {
        "type": "timeseries",
        "title": title,
        "description": description,
        "datasource": DS,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "targets": targets,
    }


def stat(title: str, expr: str, legend: str, description: str, unit: str = "none",
         updown: bool = False, steps: dict | None = None) -> dict:
    defaults: dict[str, Any] = {"unit": unit, "noValue": "no data", "thresholds": steps or thresholds(("blue", None))}
    if updown:
        defaults["mappings"] = [{
            "type": "value",
            "options": {
                "0": {"text": "DOWN", "color": "red", "index": 0},
                "1": {"text": "UP", "color": "green", "index": 1},
            },
        }]
        defaults["thresholds"] = thresholds(("red", None), ("green", 1))
    return {
        "type": "stat",
        "title": title,
        "description": description,
        "datasource": DS,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "background",
            "graphMode": "none",
            "textMode": "value_and_name",
            "justifyMode": "auto",
            "orientation": "auto",
        },
        "targets": [target(expr, legend)],
    }


def text(title: str, content: str) -> dict:
    return {"type": "text", "title": title, "options": {"mode": "markdown", "content": content}}


def table(title: str, expr: str, description: str) -> dict:
    return {
        "type": "table",
        "title": title,
        "description": description,
        "datasource": DS,
        "fieldConfig": {"defaults": {"noValue": "none"}, "overrides": []},
        "options": {"showHeader": True},
        "targets": [target(expr, instant=True, fmt="table")],
    }


def heatmap(title: str, expr: str, description: str) -> dict:
    return {
        "type": "heatmap",
        "title": title,
        "description": description,
        "datasource": DS,
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {
            "calculate": False,
            "cellGap": 1,
            "color": {"mode": "scheme", "scheme": "Oranges", "steps": 64},
            "yAxis": {"unit": "s", "axisPlacement": "left"},
            "rowsFrame": {"layout": "auto"},
            "tooltip": {"mode": "single", "yHistogram": True},
            "legend": {"show": True},
        },
        "targets": [target(expr, "{{le}}", fmt="heatmap")],
    }


def quantiles(metric: str, by: str, selector: str = "", window: str = "5m") -> list[dict]:
    out = []
    for ref, q in (("A", "0.5"), ("B", "0.95"), ("C", "0.99")):
        expr = f"histogram_quantile({q}, sum by (le, {by}) (rate({metric}_bucket{selector}[{window}])))"
        labels = " ".join("{{" + b.strip() + "}}" for b in by.split(","))
        legend = "p" + q.split(".")[1].ljust(2, "0")[:2] + " " + labels
        out.append(target(expr, legend, ref))
    return out


def service_health() -> Board:
    b = Board("dreamdrive-service-health", "DreamDrive / Service health",
              "Availability of api, worker and inference, model readiness, HTTP traffic, alerts and storage.")
    for job in ("api", "worker", "inference"):
        b.add(stat(f"{job} scrape", f'max(up{{job="{job}"}})', job,
                   f"1 when Prometheus can scrape {job}; DOWN for > 2 min raises DreamDriveServiceDown.",
                   updown=True), 4, 4)
    b.add(stat("Model ready", "max by (model, device) (dreamdrive_model_ready)", "{{model}} {{device}}",
               "1 only after the SHA256-verified checkpoint loaded and passed warm-up.", updown=True), 6, 4)
    b.add(stat("Model cold start", "max by (model, device) (dreamdrive_model_load_seconds)", "{{model}} {{device}}",
               "Measured verify + load + warm-up seconds of the last successful start.", unit="s"), 6, 4)
    b.add(timeseries("Worker heartbeat age", [target("dreamdrive_worker_heartbeat_age_seconds", "{{slot}}")], "s",
                     "Seconds since each worker slot last reported progress. Alert threshold 120 s.", warn=120), 12, 8)
    b.add(timeseries("HTTP requests by status class",
                     [target("sum by (status_class) (rate(dreamdrive_http_requests_total[5m]))", "{{status_class}}")],
                     "reqps", "API request rate by status class (route templates only, no ids)."), 12, 8)
    b.add(timeseries("HTTP p95 latency by route",
                     [target("histogram_quantile(0.95, sum by (le, route) "
                             "(rate(dreamdrive_http_request_duration_seconds_bucket[5m])))", "{{route}}")],
                     "s", "95th percentile API handler latency per route template."), 12, 8)
    b.add(table("Firing alerts", 'ALERTS{alertstate="firing"}',
                "Alerts currently firing in Prometheus. No notification destination is configured."), 12, 8)
    b.add(timeseries("Data volume free space", [target("dreamdrive_data_volume_free_bytes", "free")], "bytes",
                     "Free bytes on the /data volume (SQLite, uploads, artifacts). Alert below 5 GB.", warn=5e9), 12, 8)
    b.add(timeseries("Artifact bytes by category", [target("sum by (category) (dreamdrive_artifact_bytes)",
                                                           "{{category}}")],
                     "bytes", "Stored artifact bytes per category, after retention cleanup.", stack=True), 12, 8)
    return b


def jobs_queue() -> Board:
    b = Board("dreamdrive-jobs-queue", "DreamDrive / Jobs and queue",
              "RQ queue depth/age, job outcomes, processed frames and per-stage video pipeline timing.")
    b.add(timeseries("Queue depth", [target("max by (queue) (dreamdrive_queue_depth)", "{{queue}}")], "short",
                     "Jobs waiting per RQ queue (video, sim, eval)."), 12, 8)
    b.add(timeseries("Oldest queued job age", [target("max by (queue) (dreamdrive_queue_oldest_age_seconds)",
                                                      "{{queue}}")],
                     "s", "Age of the oldest waiting job from real enqueue timestamps. SLA alert at 900 s.",
                     warn=900), 12, 8)
    b.add(timeseries("Finished jobs per hour by status",
                     [target("sum by (kind, status) (increase(dreamdrive_jobs_total[1h]))", "{{kind}} {{status}}")],
                     "short", "Terminal job outcomes (succeeded, failed, cancelled, interrupted) per kind."), 12, 8)
    b.add(timeseries("Frames processed per second",
                     [target("sum by (mode, source, outcome) (rate(dreamdrive_frames_processed_total[5m]))",
                             "{{mode}} {{source}} {{outcome}}")],
                     "short", "Actual analyzed frames per mode, perception source and outcome (from counts)."),
          12, 8)
    b.add(timeseries("Video pipeline stage p95",
                     [target("histogram_quantile(0.95, sum by (le, stage) "
                             "(rate(dreamdrive_video_stage_duration_seconds_bucket[15m])))", "{{stage}}")],
                     "s", "End-to-end stage timing (probe, decode, inference, annotate, encode, report) per job."),
          12, 8)
    b.add(timeseries("Video pipeline stage time share",
                     [target("sum by (stage) (rate(dreamdrive_video_stage_duration_seconds_sum[15m]))", "{{stage}}")],
                     "s", "Seconds per second spent in each stage across all jobs (where wall time goes).",
                     stack=True), 12, 8)
    return b


def inference() -> Board:
    b = Board("dreamdrive-inference", "DreamDrive / Inference",
              "Measured YOLOPv2 inference latency and throughput from the inference service.")
    b.add(text("How to read this",
               "Latency is the model forward pass plus decode/NMS/mask inversion **under the model lock**. "
               "It excludes queueing for the lock, image decode, PNG encoding and HTTP transfer "
               "(those appear in each response's `timing_ms`). GPU memory metrics are not exported on CPU."),
          24, 3)
    b.add(timeseries("Inference latency p50 / p95 / p99",
                     quantiles("dreamdrive_inference_duration_seconds", "model, device",
                               '{job="inference", outcome="ok"}'),
                     "s", "Quantiles over 5 minutes of successful inferences, measured inside the inference "
                     "service (the worker exports the same family per analyzed frame; it is excluded here)."), 12, 9)
    b.add(timeseries("Inference throughput",
                     [target('sum by (model, device) (rate(dreamdrive_inference_duration_seconds_count'
                             '{job="inference", outcome="ok"}[5m]))', "{{model}} {{device}}")],
                     "ops", "Successful forward passes per second (one at a time by design)."), 12, 9)
    b.add(heatmap("Inference latency distribution",
                  'sum by (le) (increase(dreamdrive_inference_duration_seconds_bucket'
                  '{job="inference", outcome="ok"}[1m]))',
                  "Histogram buckets per minute."), 12, 9)
    b.add(timeseries("Inference requests by outcome",
                     [target("sum by (outcome) (rate(dreamdrive_inference_requests_total[5m]))", "{{outcome}}")],
                     "reqps", "ok, image_invalid, image_too_large, model_not_ready, inference_busy, inference_failed.",
                     stack=True), 12, 9)
    return b


def simulation() -> Board:
    b = Board("dreamdrive-simulation", "DreamDrive / Simulation",
              "LiteRoad closed-loop health: real-time factor, safety overrides and camera frame age.")
    b.add(timeseries("Simulation real-time factor",
                     [target("dreamdrive_simulation_realtime_factor", "{{simulator}} {{source}}")], "none",
                     "Simulated seconds per wall second (paced to <= 1.0; lower means the loop is slower than "
                     "real time)."), 12, 8)
    b.add(timeseries("Safety overrides",
                     [target("sum by (reason, perception_source) "
                             "(increase(dreamdrive_safety_overrides_total[5m]))",
                             "{{reason}} {{perception_source}}")],
                     "short", "Safety supervisor overrides per reason code and perception source (5 min).",
                     stack=True), 12, 8)
    b.add(timeseries("Frame acquisition age", [target("dreamdrive_frame_acquisition_age_seconds", "{{source}}")],
                     "s", "Age of the camera frame consumed by perception; stale frames force a safety state."),
          12, 8)
    b.add(timeseries("Simulation frames per second",
                     [target('sum by (source, outcome) (rate(dreamdrive_frames_processed_total'
                             '{mode="simulation"}[5m]))', "{{source}} {{outcome}}")],
                     "short", "Closed-loop perception frames processed per second."), 12, 8)
    return b


BOARDS = (service_health, jobs_queue, inference, simulation)


def render() -> dict[str, str]:
    return {f"{b.uid}.json": json.dumps(b.to_json(), indent=2, sort_keys=False) + "\n" for b in (f() for f in BOARDS)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "ops" / "grafana" / "dashboards")
    parser.add_argument("--check", action="store_true", help="fail if committed files differ from generated")
    args = parser.parse_args()
    files = render()
    if args.check:
        drift = [n for n, c in files.items()
                 if not (args.out / n).is_file() or (args.out / n).read_text(encoding="utf-8") != c]
        extra = sorted(p.name for p in args.out.glob("*.json") if p.name not in files)
        if drift or extra:
            print(f"dashboard drift: changed={drift} unexpected={extra}; run scripts/gen_grafana_dashboards.py",
                  file=sys.stderr)
            return 1
        print(f"dashboards up to date ({len(files)} files)")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (args.out / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
