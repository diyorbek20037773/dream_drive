"""report.json (contracts/run-report.schema.json) and report.html builders for uploaded-video analysis.

Only measured quantities get `status: measured` (with unit and sample count). Accuracy metrics need labels that an
upload does not have, so they are `not_evaluated` with value null; TTC is `unsupported`. The HTML report is a single
self-contained file: inline CSS, the poster embedded as a data URI, no scripts, every dynamic string HTML-escaped.

report.json stays strictly contract-valid (no extra top-level keys). With the two-model service (spec section 2.1)
`model_sha256` is the ensemble sha256 and a limitation line lists every model id with its checkpoint SHA256; per-class
detection counts (`detections_<class>`), per-class confirmed track counts (`tracks_<class>`), detections per model
(`detections_from_<model_id>`), class refinements and traffic-light HSV-heuristic state counts
(`traffic_light_hsv_<state>`) are measured count metrics.

Reporting threshold (Settings.detection_report_min_score, DETECTION_REPORT_MIN_SCORE, default 0.4): `detections_<class>`
and the track list/`tracks_<class>` count only detections with score >= the threshold and tracks whose max_score
reaches it; lower-score detections are reported as `low_score_detections_<class>` and a limitation line states the
threshold and how many tracks were dropped. timeline.json is unaffected (every detection with its score).
"""

from __future__ import annotations

import base64
import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.schemas.report import Metric, RunReport

NO_LABELS_NOTE = "no labeled ground truth for this upload"


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy 'linear' method); None for an empty sample."""
    if not values:
        return None
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


@dataclass
class VideoRunFacts:
    """Everything measured during one analysis run (inputs to report.json / report.html / summary)."""

    job_id: str
    code_commit: str
    generated_utc: datetime
    input_sha256: str
    input_bytes: int
    video: dict[str, Any]
    probe_tool: str
    analysis_fps: float
    model_id: str
    model_sha256: str
    model_device: str
    model_precision: str
    model_revision: str | None
    class_map: dict[str, str]
    decoded_frames: int
    sampled_frames: int
    analyzed_frames: int
    failed_frames: int
    estimated_pts_frames: int
    output_frames: int
    roundtrip_ms: list[float]
    server_inference_ms: list[float]
    decode_ms: float
    postprocess_ms: float
    encode_ms: float
    total_wall_ms: float
    vehicle_detections: int
    lane_valid_frames: int
    decision_counts: dict[str, int]
    reason_counts: dict[str, int]
    class_counts: dict[str, int]
    tracks: list[dict[str, Any]]
    limitations: list[str] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)
    # Two-model service (spec section 2.1). `models` holds one entry per model that ran:
    # {model_id, label, checkpoint_sha256, device, precision, repository_revision, class_map}.
    models: list[dict[str, Any]] = field(default_factory=list)
    detection_counts_by_model: dict[str, int] = field(default_factory=dict)
    refined_detection_counts: dict[str, int] = field(default_factory=dict)
    track_counts_by_class: dict[str, int] = field(default_factory=dict)
    light_state_counts: dict[str, int] = field(default_factory=dict)  # per evaluated traffic-light detection
    frame_light_state_counts: dict[str, int] = field(default_factory=dict)  # aggregated relevant state per frame
    # Reporting threshold: class_counts/tracks hold findings with score >= report_min_score; the rest is listed here.
    class_counts_low_score: dict[str, int] = field(default_factory=dict)
    report_min_score: float | None = None
    low_score_tracks_dropped: int = 0

    @property
    def coverage_ratio(self) -> float:
        return self.analyzed_frames / self.sampled_frames if self.sampled_frames else 0.0

    @property
    def effective_fps(self) -> float:
        return self.analyzed_frames / (self.total_wall_ms / 1000.0) if self.total_wall_ms > 0 else 0.0


def _latency(values: list[float], q: float, label: str) -> Metric:
    p = percentile(values, q)
    if p is None:
        return Metric.missing("ms", f"no successful {label} samples", status="not_evaluated")
    return Metric.measured(round(p, 3), "ms", len(values), f"p{q:g} over analyzed frames")


def build_metrics(f: VideoRunFacts) -> dict[str, Metric]:
    analyzed = f.analyzed_frames
    metrics: dict[str, Metric] = {
        "decoded_frames": Metric.measured(f.decoded_frames, "frames", max(1, f.decoded_frames), "decoded video frames"),
        "sampled_frames": Metric.measured(
            f.sampled_frames, "frames", max(1, f.decoded_frames), f"PTS-grid samples at {f.analysis_fps:g} fps"
        ),
        "analyzed_frames": Metric.measured(analyzed, "frames", max(1, f.sampled_frames), "samples with model output"),
        "inference_failed_frames": Metric.measured(
            f.failed_frames, "frames", max(1, f.sampled_frames), "samples whose inference failed"
        ),
        "inference_coverage_ratio": Metric.measured(
            round(f.coverage_ratio, 6), "ratio", max(1, f.sampled_frames), "analyzed / sampled frames"
        ),
        "video_duration": Metric.measured(round(float(f.video.get("duration_s") or 0.0), 4), "s", 1, "probed duration"),
        "inference_roundtrip_latency_p50": _latency(f.roundtrip_ms, 50, "round-trip"),
        "inference_roundtrip_latency_p95": _latency(f.roundtrip_ms, 95, "round-trip"),
        "model_forward_latency_p50": _latency(f.server_inference_ms, 50, "server-reported forward"),
        "model_forward_latency_p95": _latency(f.server_inference_ms, 95, "server-reported forward"),
        "total_wall_time": Metric.measured(round(f.total_wall_ms / 1000.0, 4), "s", 1, "probe to published artifacts"),
        "decode_time": Metric.measured(round(f.decode_ms / 1000.0, 4), "s", max(1, f.decoded_frames), "demux+decode"),
        "postprocess_time": Metric.measured(
            round(f.postprocess_ms / 1000.0, 4),
            "s",
            max(1, f.sampled_frames),
            "mask decode, tracking, decisions, overlay",
        ),
        "encode_time": Metric.measured(round(f.encode_ms / 1000.0, 4), "s", max(1, f.output_frames), "H.264 encode"),
        "confirmed_tracks": Metric.measured(len(f.tracks), "tracks", max(1, analyzed), "ByteTrack confirmed track ids"),
    }
    light_total = sum(f.light_state_counts.values())
    if light_total:
        metrics["traffic_light_state_known_ratio"] = Metric.measured(
            round(1.0 - f.light_state_counts.get("unknown", 0) / light_total, 6),
            "ratio",
            light_total,
            "share of evaluated traffic-light detections the HSV heuristic assigned a colour; not accuracy",
        )
    if analyzed > 0:
        metrics["effective_analyzed_fps"] = Metric.measured(
            round(f.effective_fps, 4), "frames/s", analyzed, "analyzed frames / total wall time"
        )
        metrics["vehicle_detections_per_frame"] = Metric.measured(
            round(f.vehicle_detections / analyzed, 4), "detections/frame", analyzed, "class 'vehicle', all scores"
        )
        metrics["lane_valid_frame_ratio"] = Metric.measured(
            round(f.lane_valid_frames / analyzed, 6),
            "ratio",
            analyzed,
            "data-quality indicator (lane evidence on both sides), not lane accuracy",
        )
    else:
        for name, unit in (
            ("effective_analyzed_fps", "frames/s"),
            ("vehicle_detections_per_frame", "detections/frame"),
            ("lane_valid_frame_ratio", "ratio"),
        ):
            metrics[name] = Metric.missing(unit, "no analyzed frames")
    if analyzed > 0:
        metrics.update(_count_metrics(f, analyzed))
    for name in (
        "detection_map50",
        "detection_precision",
        "detection_recall",
        "lane_iou",
        "drivable_iou",
        "tracking_idf1",
    ):
        metrics[name] = Metric.missing("ratio", NO_LABELS_NOTE, status="not_evaluated")
    metrics["time_to_collision"] = Metric.missing(
        "s", "no metric depth or ego speed in an uploaded video; ttc_seconds is null", status="unsupported"
    )
    return metrics


def _metric_key(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_") or "unknown"


def _count_metrics(f: VideoRunFacts, analyzed: int) -> dict[str, Metric]:
    """Measured counts over the analyzed frames (no accuracy implied)."""
    out: dict[str, Metric] = {}
    thr = f.report_min_score
    score_rule = "all scores" if thr is None else f"score >= {thr:g}"
    for name, count in sorted(f.class_counts.items()):
        out[f"detections_{_metric_key(name)}"] = Metric.measured(
            count, "detections", analyzed, f"class '{name}' detections over analyzed frames, {score_rule}"
        )
    if thr is not None:
        for name, count in sorted(f.class_counts_low_score.items()):
            out[f"low_score_detections_{_metric_key(name)}"] = Metric.measured(
                count,
                "detections",
                analyzed,
                f"class '{name}' detections with score < {thr:g} (kept for tracking, not counted as findings)",
            )
    track_rule = "" if thr is None else f" and max_score >= {thr:g}"
    for name, count in sorted(f.track_counts_by_class.items()):
        out[f"tracks_{_metric_key(name)}"] = Metric.measured(
            count, "tracks", analyzed, f"ByteTrack confirmed tracks whose majority class is '{name}'{track_rule}"
        )
    if len(f.models) > 1:
        for model_id, count in sorted(f.detection_counts_by_model.items()):
            out[f"detections_from_{_metric_key(model_id)}"] = Metric.measured(
                count, "detections", analyzed, f"detections produced by model '{model_id}'"
            )
        refined = sum(f.refined_detection_counts.values())
        out["refined_detections"] = Metric.measured(
            refined,
            "detections",
            analyzed,
            "vehicle boxes relabelled bus/truck by the second detector (IoU >= 0.5): "
            + (", ".join(f"{k}={v}" for k, v in sorted(f.refined_detection_counts.items())) or "none"),
        )
    for state, count in sorted(f.light_state_counts.items()):
        out[f"traffic_light_hsv_{_metric_key(state)}"] = Metric.measured(
            count,
            "detections",
            analyzed,
            f"traffic-light detections the HSV heuristic labelled '{state}' (hsv_heuristic, not model output)",
        )
    for state, count in sorted(f.frame_light_state_counts.items()):
        out[f"frames_relevant_light_hsv_{_metric_key(state)}"] = Metric.measured(
            count,
            "frames",
            analyzed,
            f"frames whose relevant (upper-central) light state was '{state}' (hsv_heuristic, not model output)",
        )
    return out


def model_identity_lines(f: VideoRunFacts) -> list[str]:
    """Limitation lines naming every model that ran and its checkpoint SHA256 (ensemble runs only)."""
    if len(f.models) < 2:
        return []
    members = "; ".join(f"{m.get('model_id')} checkpoint sha256 {m.get('checkpoint_sha256')}" for m in f.models)
    return [
        f"Perception identity {f.model_id} (model_sha256 is the ensemble sha256 {f.model_sha256} = sha256 over the "
        f"'<model_id>:<sha256>' lines of its members): {members}."
    ]


def report_threshold_lines(f: VideoRunFacts) -> list[str]:
    """Limitation line stating the reporting threshold and what it moved out of the findings."""
    if f.report_min_score is None:
        return []
    low = sum(f.class_counts_low_score.values())
    low_text = ", ".join(f"{k}={v}" for k, v in sorted(f.class_counts_low_score.items())) or "none"
    return [
        f"Reporting threshold {f.report_min_score:g} (DETECTION_REPORT_MIN_SCORE): detection counts and the track "
        f"list include only detections with score >= {f.report_min_score:g} and tracks whose max_score reaches it. "
        f"{low} lower-score detection(s) ({low_text}) are reported separately as low_score_detections_<class> and "
        f"{f.low_score_tracks_dropped} low-score track(s) were left out of the track list; the inference service "
        "returns boxes down to its lower confidence threshold for ByteTrack association, and timeline.json keeps "
        "every detection with its score."
    ]


def build_run_report(f: VideoRunFacts) -> dict[str, Any]:
    report = RunReport(
        run_id=f.job_id,
        kind="video_analysis",
        status="succeeded",
        perception_source="learned_model",
        code_commit=f.code_commit,
        model_sha256=f.model_sha256,
        input_manifest_sha256=f.input_sha256,
        generated_utc=f.generated_utc,
        metrics=build_metrics(f),
        limitations=[*model_identity_lines(f), *report_threshold_lines(f), *f.limitations],
        evaluation_truth_used=False,
    )
    return report.model_dump(mode="json")


# ----------------------------------------------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------------------------------------------
_CSS = """
:root{--bg:#f7f8fa;--fg:#14171c;--muted:#5b6472;--card:#fff;--line:#dde1e7;--accent:#0b6bcb}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e8eaee;--muted:#9aa3ae;--card:#1a1e24;--line:#2c323b;
--accent:#5aa9ff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 48px}h1{font-size:1.5rem;margin:0 0 4px}
h2{font-size:1.1rem;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;border:1px solid var(--line);font-size:.8rem}
img{max-width:100%;height:auto;border-radius:6px;border:1px solid var(--line)}
code{font-size:.85rem;word-break:break-all}ul{margin:4px 0;padding-left:20px}
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1e5 else f"{value:.1f}"
    return str(value)


def _rows(pairs: Sequence[tuple[str, Any]]) -> str:
    return "".join(f"<tr><th>{_e(k)}</th><td>{_e(_fmt(v))}</td></tr>" for k, v in pairs)


def _count_table(counts: Mapping[str, int], key_label: str) -> str:
    if not counts:
        return '<p class="muted">-</p>'
    body = "".join(
        f"<tr><td><code>{_e(k)}</code></td><td class='num'>{_e(v)}</td></tr>"
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return f"<table><tr><th>{_e(key_label)}</th><th>count</th></tr>{body}</table>"


def _light_section(f: VideoRunFacts) -> str:
    if not f.light_state_counts and not f.frame_light_state_counts:
        return ""
    return (
        '<h2>Traffic-light colour (HSV heuristic) / Svetofor rangi (HSV evristika)</h2><div class="card">'
        '<p class="muted">Classical HSV colour heuristic on detected traffic-light pixels - not model output, '
        "accuracy not measured. / Klassik HSV evristika - model natijasi emas, aniqligi o&#8216;lchanmagan.</p>"
        f"{_count_table(f.light_state_counts, 'light_state (per detection)')}"
        f"{_count_table(f.frame_light_state_counts, 'relevant light_state (per frame)')}</div>"
    )


def render_report_html(f: VideoRunFacts, report: Mapping[str, Any], poster_jpeg: bytes | None) -> str:
    metrics: Mapping[str, Mapping[str, Any]] = report["metrics"]
    metric_rows = "".join(
        "<tr>"
        f"<td><code>{_e(name)}</code></td>"
        f"<td class='num'>{_e(_fmt(m['value']))}</td>"
        f"<td>{_e(m['unit'])}</td>"
        f"<td class='num'>{_e(m['sample_count'])}</td>"
        f"<td><span class='badge'>{_e(m['status'])}</span></td>"
        f"<td class='muted'>{_e(m['note'])}</td>"
        "</tr>"
        for name, m in metrics.items()
    )
    track_rows = "".join(
        "<tr>"
        f"<td class='num'>{_e(t['track_id'])}</td><td>{_e(t['class_name'])}</td>"
        f"<td class='num'>{_e(_fmt(t['first_t']))}</td><td class='num'>{_e(_fmt(t['last_t']))}</td>"
        f"<td class='num'>{_e(t['frames'])}</td><td class='num'>{_e(_fmt(t['max_score']))}</td>"
        "</tr>"
        for t in f.tracks[:200]
    )
    tracks_html = (
        "<table><tr><th>track_id</th><th>class</th><th>first_t (s)</th><th>last_t (s)</th><th>frames</th>"
        f"<th>max_score</th></tr>{track_rows}</table>"
        if f.tracks
        else '<p class="muted">-</p>'
    )
    if len(f.tracks) > 200:
        tracks_html += f"<p class='muted'>{_e(len(f.tracks) - 200)} more tracks in timeline.json</p>"
    poster_html = ""
    if poster_jpeg:
        b64 = base64.b64encode(poster_jpeg).decode("ascii")
        poster_html = (
            '<h2>Poster frame / Namuna kadr</h2><div class="card">'
            f'<img alt="annotated poster frame" src="data:image/jpeg;base64,{b64}"></div>'
        )
    limitations = "".join(f"<li>{_e(item)}</li>" for item in report["limitations"])
    thr_text = "all scores" if f.report_min_score is None else f"score &gt;= {_e(f'{f.report_min_score:g}')}"
    threshold_note = f'<p class="muted">Counted as findings: {thr_text}. / Topilma sifatida sanalgan: {thr_text}.</p>'
    low_note = (
        '<p class="muted">Below the reporting threshold; kept for ByteTrack association and listed in timeline.json, '
        "not counted as findings. / Hisobot chegarasidan past; topilma sifatida sanalmaydi.</p>"
    )
    supported = ", ".join(sorted(set(f.class_map.values()))) or "-"
    has_second = bool(set(f.class_map.values()) & {"person", "traffic_light", "stop_sign"})
    unsupported = (
        "general traffic signs other than stop_sign, road markings, animals, debris"
        if has_second
        else "person, traffic_light, traffic_sign"
    )
    report_rule = (
        "all scores" if f.report_min_score is None else f"score >= {f.report_min_score:g} (DETECTION_REPORT_MIN_SCORE)"
    )
    summary_pairs = [
        ("job_id", f.job_id),
        ("status", report["status"]),
        ("generated_utc", report["generated_utc"]),
        ("perception_source", report["perception_source"]),
        ("code_commit", f.code_commit),
        ("input_sha256", f.input_sha256),
        ("input_bytes", f.input_bytes),
        ("video", ", ".join(f"{k}={_fmt(v)}" for k, v in f.video.items())),
        ("probe_tool", f.probe_tool),
        ("analysis_fps", f.analysis_fps),
        ("detection report threshold", report_rule),
        (
            "decoded / sampled / analyzed / failed frames",
            f"{f.decoded_frames} / {f.sampled_frames} / {f.analyzed_frames} / {f.failed_frames}",
        ),
        ("audio", "removed"),
    ]
    if len(f.models) > 1:
        model_pairs = [
            ("ensemble_id", f.model_id),
            ("ensemble_sha256", f.model_sha256),
            *[(f"{m.get('model_id')} checkpoint_sha256", m.get("checkpoint_sha256")) for m in f.models],
            *[
                (f"{m.get('model_id')} repository_revision", m.get("repository_revision"))
                for m in f.models
                if m.get("repository_revision")
            ],
            ("device / precision", f"{f.model_device} / {f.model_precision}"),
            ("enabled classes", supported),
            ("unsupported examples", unsupported),
            ("traffic-light colour", "hsv_heuristic (not model output, accuracy not measured)"),
        ]
    else:
        model_pairs = [
            ("model_id", f.model_id),
            ("checkpoint_sha256", f.model_sha256),
            ("repository_revision", f.model_revision),
            ("device / precision", f"{f.model_device} / {f.model_precision}"),
            ("verified classes", supported),
            ("unsupported examples", unsupported),
        ]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DreamDrive video report</title><style>{_CSS}</style></head>
<body><main>
<h1>DreamDrive video analysis report / Video tahlili hisoboti</h1>
<p class="muted">Analytical only - no vehicle was controlled. / Faqat tahliliy - hech qanday transport vositasi
boshqarilmadi.</p>
<h2>Summary / Xulosa</h2><div class="card"><table>{_rows(summary_pairs)}</table></div>
<h2>Model / Model</h2><div class="card"><table>{_rows(model_pairs)}</table></div>
{poster_html}
<h2>Metrics / Ko&#8216;rsatkichlar</h2><div class="card"><table>
<tr><th>metric</th><th>value</th><th>unit</th><th>samples</th><th>status</th><th>note</th></tr>{metric_rows}</table></div>
<h2>Decisions / Qarorlar</h2><div class="card">{_count_table(f.decision_counts, "state")}</div>
<h2>Reason codes / Sabab kodlari</h2><div class="card">{_count_table(f.reason_counts, "reason_code")}</div>
<h2>Detected classes / Aniqlangan sinflar</h2>
<div class="card">{threshold_note}{_count_table(f.class_counts, "class_name")}</div>
<h2>Low-confidence detections / Past ishonchli aniqlashlar</h2>
<div class="card">{low_note}{_count_table(f.class_counts_low_score, "class_name")}</div>
<h2>Tracks per class / Sinf bo&#8216;yicha treklar</h2>
<div class="card">{_count_table(f.track_counts_by_class, "class_name")}</div>
{_light_section(f)}
<h2>Tracks / Treklar</h2><div class="card">{tracks_html}</div>
<h2>Limitations / Cheklovlar</h2><div class="card"><ul>{limitations}</ul></div>
</main></body></html>
"""
