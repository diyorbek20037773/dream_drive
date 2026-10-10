"""Uploaded-video analysis orchestration (docs/IMPLEMENTATION_SPEC.md section 5).

probe -> decode/PTS-grid sample -> per sample: JPEG q95 -> InferenceClient.infer -> ByteTrack -> analytical decision
-> WorldState -> overlay -> H.264 encode; then timeline.json, report.json, report.html, poster.jpg.

All outputs are written into a private temporary directory next to `out_dir` and published only after everything
succeeded: if `out_dir` does not exist the whole directory is renamed atomically; if it exists (e.g. it already holds
the uploaded `input`) each output file is moved with an atomic `os.replace`, report.json last, so a reader that sees
report.json sees a complete result. On cancellation or any error the temporary directory is removed and no output
name is left behind in `out_dir` (the `input` file is never touched).

Perception identity: with the two-model service (YOLOPv2 + RF-DETR-N, spec section 2.1) every WorldState carries
the ensemble id and ensemble sha256; the timeline/report list each model with its checkpoint sha256. Per-detection
provenance (`model_id`, `refined_from_model`) and the traffic-light HSV heuristic (`light_state`,
`light_state_method: hsv_heuristic`) are timeline extras next to each WorldState record.

Inference failures on single frames are recorded honestly (perception source `unavailable`, reason INFERENCE_FAILED)
and the job continues; it fails with `inference_failed` / `inference_unavailable` after 3 consecutive failures or
when more than 10 % of the samples failed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.perception.client import (
    InferenceClient,
    InferenceClientError,
    InferenceUnavailable,
    PerceptionIdentity,
    merged_class_map,
    perception_identity,
)
from app.perception.tracking import TrackerConfig, VideoTracker, tracked_class_names
from app.perception.traffic_light import DEFAULT_LIGHT_THRESHOLDS, LIGHT_STATE_METHOD
from app.perception.video_decisions import (
    CLASS_FAMILIES,
    DEFAULT_THRESHOLDS,
    VideoDecision,
    decide,
    inference_failed_decision,
)
from app.perception.world_state import ModelRef, TrackedDetection, build_video_world_state
from app.reporting.video_report import VideoRunFacts, build_run_report, percentile, render_report_html
from app.schemas.inference import InferenceResult, ModelInfo
from app.video import errors
from app.video.annotate import HudInfo, annotate_frame
from app.video.decode import DecodeLimits, DecodeStats, iter_sampled_frames
from app.video.encode import H264Writer, encode_jpeg_bytes, write_jpeg
from app.video.errors import JobCancelled, VideoError
from app.video.probe import ProbeLimits, VideoProbe, probe_video

ProgressCallback = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]

OUTPUT_NAMES = ("annotated.mp4", "poster.jpg", "timeline.json", "report.html", "report.json")  # publish order
TIMELINE_SCHEMA_VERSION = "1.0"
MAX_CONSECUTIVE_FAILURES = 3
MAX_FAILED_FRACTION = 0.10
PROGRESS_INTERVAL_S = 0.5
MODEL_LABELS = {"yolopv2": "YOLOPv2", "rfdetr_nano": "RF-DETR-N"}
# Classes of the second detector (spec section 2.1); used only to word the limitations honestly.
RFDETR_CLASSES = frozenset({"person", "bicycle", "motorcycle", "bus", "truck", "traffic_light", "stop_sign"})


def model_label(identity: PerceptionIdentity) -> str:
    """HUD label, e.g. "YOLOPv2" or "YOLOPv2 + RF-DETR-N" (ASCII)."""
    return " + ".join(MODEL_LABELS.get(m.model_id, m.model_id) for m in identity.models)


def _check_identity_against_info(info: ModelInfo, ident: PerceptionIdentity, result: InferenceResult) -> None:
    """The first result must come from the models /v1/model announced (no silent single-model fallback)."""
    if info.sha256 and info.sha256 not in {result.model.sha256, ident.sha256}:
        raise VideoError(errors.MODEL_CHANGED, "inference result model differs from /v1/model")
    entries = getattr(info, "models", None) or []
    announced = {(e.model_id, e.sha256) for e in entries if e.sha256}
    if len(entries) >= 2 and not ident.is_ensemble:
        raise VideoError(errors.MODEL_CHANGED, "/v1/model announced several models but the result names one")
    if ident.is_ensemble and announced and announced != {(m.model_id, m.sha256) for m in ident.models}:
        raise VideoError(errors.MODEL_CHANGED, "inference result models differ from /v1/model")
    info_ens = getattr(info, "ensemble_sha256", None)
    if ident.is_ensemble and info_ens and info_ens != ident.sha256:
        raise VideoError(errors.MODEL_CHANGED, "inference result ensemble differs from /v1/model")


def _model_entries(info: ModelInfo, ident: PerceptionIdentity) -> list[dict[str, Any]]:
    """Per-model identity for timeline/report: id, checkpoint sha256, device, precision, revision, class names."""
    by_id = {e.model_id: e for e in (getattr(info, "models", None) or [])}
    out: list[dict[str, Any]] = []
    for m in ident.models:
        entry = by_id.get(m.model_id)
        if entry is not None:
            class_map, revision = dict(entry.class_map), entry.repository_revision
        elif m.model_id == info.model_id:
            class_map, revision = dict(info.class_map), info.repository_revision
        else:
            class_map, revision = {}, None
        out.append(
            {
                "model_id": m.model_id,
                "label": MODEL_LABELS.get(m.model_id, m.model_id),
                "checkpoint_sha256": m.sha256,
                "device": m.device,
                "precision": m.precision,
                "repository_revision": revision,
                "class_map": class_map,
            }
        )
    return out


class _Progress:
    """Throttled progress reporting: at most every PROGRESS_INTERVAL_S, plus stage changes and forced calls."""

    def __init__(self, cb: ProgressCallback | None) -> None:
        self._cb = cb
        self._last_t = 0.0
        self._last_stage: str | None = None

    def __call__(self, done: int, total: int, stage: str, force: bool = False) -> None:
        if self._cb is None:
            return
        now = time.monotonic()
        if force or stage != self._last_stage or now - self._last_t >= PROGRESS_INTERVAL_S:
            self._cb(done, max(total, done), stage)
            self._last_t, self._last_stage = now, stage


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def decode_mask_png(data: str | None, width: int, height: int) -> NDArray[np.bool_] | None:
    """Base64 PNG (uint8 {0,255}, original H x W) -> bool mask; None when absent or not matching the frame."""
    if not data:
        return None
    try:
        raw = np.frombuffer(base64.b64decode(data, validate=True), dtype=np.uint8)
    except (ValueError, TypeError):
        return None
    mask = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != (height, width):
        return None
    return mask > 127


def _estimate_samples(probe: VideoProbe, fps: float) -> int:
    est = math.floor(probe.duration_s * fps) + 1
    if probe.nb_frames:
        est = min(est, probe.nb_frames)
    return max(1, est)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _publish(tmp_dir: Path, out_dir: Path) -> None:
    if not out_dir.exists():
        os.replace(tmp_dir, out_dir)
        return
    moved: list[Path] = []
    try:
        for name in OUTPUT_NAMES:
            target = out_dir / name
            os.replace(tmp_dir / name, target)
            moved.append(target)
    except OSError:
        for target in moved:
            target.unlink(missing_ok=True)
        raise
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _limitations(
    class_map: dict[str, str],
    model: PerceptionIdentity,
    fps: float,
    failed: int,
    estimated_pts: int,
    probe_tool: str,
    build_dirty: bool,
    stopped_at_decode_error: bool = False,
    decoded_frames: int = 0,
) -> list[str]:
    names = set(class_map.values())
    classes = ", ".join(sorted(names)) or "none"
    if names & RFDETR_CLASSES:
        class_items = [
            f"Detector classes enabled for this run: {classes} (YOLOPv2: vehicle; RF-DETR-N, COCO-pretrained: the "
            "others). General traffic signs other than stop signs, road markings, animals and debris are not "
            "detected; detections of other class ids are excluded from decisions.",
            f"Traffic-light colour (red/yellow/green/unknown) is a classical HSV heuristic ({LIGHT_STATE_METHOD}) on "
            "the detected light's pixels, not model output, and its accuracy is not measured; TRAFFIC_LIGHT_*_HEURISTIC"
            " codes and controlled_stop recommendations derived from it are heuristic.",
            "RF-DETR-N is trained on COCO, not on driving footage; detector accuracy for this upload is not evaluated "
            "(see the COCO street-subset evaluation for a reference measurement).",
        ]
    else:
        class_items = [
            f"Verified detector classes for this checkpoint: {classes}. Persons, traffic lights, traffic signs and "
            "other categories are not detected; detections of unverified class ids are excluded from decisions."
        ]
    items = [
        "Analytical only: the uploaded video is an observation recording; no vehicle or actuator was controlled and "
        "decisions cannot change the next frame.",
        *class_items,
        "image_obstacle_cue is an image-space heuristic from box position and height, not a distance; ttc_seconds is "
        "null because an uploaded video has no metric depth or ego speed.",
        "Accuracy metrics (mAP, precision/recall, lane/drivable IoU, IDF1) are not evaluated: no labeled ground truth "
        "for this upload.",
        f"Frames were sampled on a {fps:g} fps presentation-timestamp grid; annotated.mp4 contains only analyzed "
        f"frames at a constant {fps:g} fps, so source timing gaps are not reproduced - each output frame's original "
        "PTS is in timeline.json and in the frame HUD.",
        "Lane, corridor and quality decisions use fixed image-space thresholds that assume a forward-facing, roughly "
        "centred dash camera; side/rear or strongly off-centre mounts are outside the tested domain.",
        f"Inference ran on {model.device} ({model.precision}); latency figures describe this host only.",
        "Audio is removed from annotated.mp4.",
    ]
    if failed:
        items.append(
            f"{failed} sampled frame(s) failed inference; they are marked INFERENCE_FAILED with perception source "
            "'unavailable' and carry no detections."
        )
    if estimated_pts:
        items.append(f"{estimated_pts} decoded frame(s) had no PTS; their time was estimated from the average rate.")
    if stopped_at_decode_error:
        items.append(
            f"Decoding stopped at a corrupt packet after {decoded_frames} decoded frame(s); the rest of the video "
            "was not analyzed."
        )
    if probe_tool != "ffprobe":
        items.append(f"Container metadata was probed with {probe_tool} (ffprobe unavailable on this host).")
    if build_dirty:
        items.append("The build was marked dirty: code_commit does not fully identify the analyzed source.")
    return items


def run_video_analysis(
    job_id: str,
    input_path: Path | str,
    out_dir: Path | str,
    client: InferenceClient,
    settings: Settings,
    progress_cb: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    *,
    allow_pyav_probe_fallback: bool = False,
    decode_limits: DecodeLimits | None = None,
) -> dict[str, Any]:
    """Analyze one uploaded video; returns the `JobResults.summary` dict. See the module docstring for semantics.

    `allow_pyav_probe_fallback` exists for tests/host runs without the ffprobe binary; the worker leaves it False.
    """
    wall_start = time.perf_counter()
    if not settings.commit_is_valid:
        raise VideoError(errors.BUILD_IDENTITY_MISSING, "GIT_COMMIT must be the 40-hex commit of the running build")
    input_path, out_dir = Path(input_path), Path(out_dir)
    fps = float(settings.analysis_fps)
    progress = _Progress(progress_cb)
    cancelled = cancel_check or (lambda: False)

    progress(0, 0, "probing", force=True)
    probe = probe_video(
        input_path,
        ProbeLimits(settings.max_video_seconds, settings.max_video_width, settings.max_video_height),
        allow_pyav_fallback=allow_pyav_probe_fallback,
    )
    input_sha256 = sha256_file(input_path)
    input_bytes = input_path.stat().st_size

    try:
        info = client.model_info()
    except InferenceClientError as exc:
        raise VideoError(errors.INFERENCE_UNAVAILABLE, f"inference service unavailable ({exc.code})") from exc
    if not info.ready:
        raise VideoError(errors.INFERENCE_UNAVAILABLE, f"model not ready ({info.reason or 'unknown'})")
    try:
        class_map = merged_class_map(info)
    except InferenceClientError as exc:
        raise VideoError(errors.INFERENCE_UNAVAILABLE, f"inference service model info invalid ({exc.code})") from exc
    tracker = VideoTracker(tracked_class_names(class_map), TrackerConfig(frame_rate=fps))
    limits = decode_limits or DecodeLimits(
        max_decode_wall_s=max(60.0, 3.0 * settings.max_video_seconds),
        max_decoded_frames=int(settings.max_video_seconds * 240),
        max_width=settings.max_video_width,
        max_height=settings.max_video_height,
    )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir.parent / f".{out_dir.name}.partial-{uuid.uuid4().hex[:12]}"
    tmp_dir.mkdir()
    writer: H264Writer | None = None
    try:
        stats = DecodeStats()
        frames_total = _estimate_samples(probe, fps)
        model: PerceptionIdentity | None = None
        frames: list[dict[str, Any]] = []
        roundtrip_ms: list[float] = []
        server_ms: list[float] = []
        post_ms = 0.0
        encode_ms = 0.0
        failed = consecutive_failed = 0
        last_failure_unavailable = False
        decisions: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        classes: Counter[str] = Counter()
        model_dets: Counter[str] = Counter()
        refined_dets: Counter[str] = Counter()
        light_states: Counter[str] = Counter()
        frame_light_states: Counter[str] = Counter()
        vehicle_dets = lane_valid = 0
        poster: tuple[int, NDArray[np.uint8]] | None = None  # (score, annotated image)

        progress(0, frames_total, "analyzing", force=True)
        for sample in iter_sampled_frames(input_path, probe, fps, limits, stats):
            if cancelled():
                raise JobCancelled()
            image = sample.image
            height, width = image.shape[:2]
            t0 = time.perf_counter()
            jpeg = encode_jpeg_bytes(image, 95)
            t_jpeg = time.perf_counter()
            result: InferenceResult | None = None
            try:
                result = client.infer(
                    jpeg,
                    frame_id=sample.sample_index,
                    conf_threshold=settings.inference_conf_threshold,
                    include_masks=True,
                )
                if (result.image.width, result.image.height) != (width, height):
                    raise VideoError(errors.INFERENCE_FAILED, "inference image size does not match the frame")
            except InferenceClientError as exc:
                result = None
                last_failure_unavailable = isinstance(exc, InferenceUnavailable)
            t_infer = time.perf_counter()

            if result is not None:
                ident = perception_identity(result)
                if model is None:
                    _check_identity_against_info(info, ident, result)
                    model = ident
                elif ident != model:
                    raise VideoError(errors.MODEL_CHANGED, "model identity changed during the job")
                consecutive_failed = 0
                roundtrip_ms.append((t_infer - t_jpeg) * 1000.0)
                server_ms.append(float(result.timing_ms.inference))
                drivable = decode_mask_png(result.drivable_mask_png, width, height)
                lane = decode_mask_png(result.lane_mask_png, width, height)
                raw = result.detections
                track_ids = tracker.update(raw, sample.pts_s)
                dets = [
                    TrackedDetection(
                        d.class_id,
                        d.class_name,
                        d.score,
                        list(d.xyxy_pixels),
                        tid,
                        model_id=getattr(d, "model_id", None),
                        refined_from_model=getattr(d, "refined_from_model", None),
                    )
                    for d, tid in zip(raw, track_ids, strict=True)
                ]
                decision: VideoDecision = decide(image, drivable, lane, dets, DEFAULT_THRESHOLDS)
                if drivable is None or lane is None:
                    decision.uncertainty_reasons.append("mask_invalid")
                model_ref: ModelRef | None = ModelRef(ident.model_id, ident.sha256)
                classes.update(d.class_name for d in dets)
                # Single-model service: every box is that model's. Ensemble: never guess a missing attribution.
                fallback_id = "unattributed" if ident.is_ensemble else ident.models[0].model_id
                model_dets.update(d.model_id or fallback_id for d in dets)
                refined_dets.update(f"{d.refined_from_model}->{d.class_name}" for d in dets if d.refined_from_model)
                light_states.update(r.state for r in decision.light_states if r is not None)
                if decision.light_state is not None:
                    frame_light_states[decision.light_state] += 1
                vehicle_dets += sum(1 for d in dets if d.class_name == "vehicle")
                lane_valid += decision.lane_status == "valid"
            else:
                failed += 1
                consecutive_failed += 1
                if consecutive_failed >= MAX_CONSECUTIVE_FAILURES:
                    code = errors.INFERENCE_UNAVAILABLE if last_failure_unavailable else errors.INFERENCE_FAILED
                    raise VideoError(code, f"{consecutive_failed} consecutive frames failed inference")
                drivable = lane = None
                dets = []
                decision = inference_failed_decision(image, DEFAULT_THRESHOLDS)
                model_ref = None

            ws = build_video_world_state(
                job_id=job_id,
                frame_id=sample.sample_index,
                pts_s=sample.pts_s,
                acquired_utc=datetime.now(UTC),
                width=width,
                height=height,
                decision=decision,
                detections=dets,
                model=model_ref,
            )
            decisions[decision.state] += 1
            reasons.update(decision.reason_codes)
            hud = HudInfo(
                model_label=model_label(model) if model else "no model output",
                source=ws.perception.source,
                device=model.device.upper() if model else "-",
            )
            annotated = annotate_frame(
                image,
                drivable_mask=drivable,
                lane_mask=lane,
                detections=dets,
                decision=decision,
                hud=hud,
                pts_s=sample.pts_s,
                sample_index=sample.sample_index,
            )
            t_post = time.perf_counter()
            if writer is None:
                writer = H264Writer(tmp_dir / "annotated.mp4", width, height, fps)
            writer.write(annotated)
            t_enc = time.perf_counter()
            output_index = writer.frames_written - 1
            drawn = sum(1 for d in dets if d.track_id is not None or d.score >= 0.5)
            if poster is None or drawn > poster[0]:
                poster = (drawn, annotated)
            post_ms += (t_post - t_infer) * 1000.0
            encode_ms += (t_enc - t_post) * 1000.0

            record = ws.model_dump(mode="json")
            record.update(
                {
                    "pts_s": round(sample.pts_s, 6),
                    "pts_estimated": sample.pts_estimated,
                    "source_frame_index": sample.source_frame_index,
                    "output_frame_index": output_index,
                    "timing_ms": {
                        "jpeg_encode": round((t_jpeg - t0) * 1000.0, 3),
                        "inference_roundtrip": round((t_infer - t_jpeg) * 1000.0, 3),
                        "inference_server": None if result is None else result.timing_ms.model_dump(),
                        "postprocess": round((t_post - t_infer) * 1000.0, 3),
                        "encode": round((t_enc - t_post) * 1000.0, 3),
                    },
                    "lane_points": decision.lane_points,
                    "measurements": decision.measurements(),
                    # Aligned with perception.detections (same order).
                    "detection_sources": [
                        {"model_id": d.model_id, "refined_from_model": d.refined_from_model} for d in dets
                    ],
                    "traffic_lights": [
                        {
                            "detection_index": i,
                            "track_id": dets[i].track_id,
                            "relevant": i in decision.relevant_light_indices,
                            **res.as_dict(),
                        }
                        for i, res in enumerate(decision.light_states)
                        if res is not None
                    ],
                    "light_state": decision.light_state,
                    "light_state_method": LIGHT_STATE_METHOD if decision.light_state is not None else None,
                }
            )
            frames.append(record)
            progress(len(frames), max(frames_total, len(frames)), "analyzing")

        if cancelled():
            raise JobCancelled()
        if not frames:
            raise VideoError(errors.VIDEO_DECODE_FAILED, "no frames could be decoded")
        if model is None:
            code = errors.INFERENCE_UNAVAILABLE if last_failure_unavailable else errors.INFERENCE_FAILED
            raise VideoError(code, "no frame produced model output")
        if failed / len(frames) > MAX_FAILED_FRACTION:
            raise VideoError(errors.INFERENCE_FAILED, f"{failed} of {len(frames)} sampled frames failed inference")

        progress(len(frames), len(frames), "encoding", force=True)
        assert writer is not None
        t_close = time.perf_counter()
        writer.close()
        encode_ms += (time.perf_counter() - t_close) * 1000.0

        assert poster is not None
        write_jpeg(tmp_dir / "poster.jpg", poster[1], quality=88)
        progress(len(frames), len(frames), "writing_report", force=True)

        video_meta = probe.as_job_video()
        analyzed = len(frames) - failed
        tracks = tracker.track_summaries()
        track_counts = tracker.track_counts_by_class()
        models = _model_entries(info, model)
        timeline = {
            "schema_version": TIMELINE_SCHEMA_VERSION,
            "job_id": job_id,
            "video": {**video_meta, "start_time_s": stats.start_time_s, "probe_tool": probe.probe_tool},
            "model": {
                "model_id": model.model_id,
                "sha256": model.sha256,
                "device": model.device,
                "precision": model.precision,
                "repository_revision": info.repository_revision if not model.is_ensemble else None,
                "class_map": class_map,
                "ensemble": model.is_ensemble,
                "models": models,
            },
            "analysis_fps": fps,
            "sampling": {
                "rule": "first decoded frame with pts >= k / analysis_fps (seconds since stream start)",
                "decoded_frames": stats.decoded_frames,
                "sampled_frames": stats.sampled_frames,
                "analyzed_frames": analyzed,
                "inference_failed_frames": failed,
                "estimated_pts_frames": stats.estimated_pts_frames,
                "dropped_non_monotonic_frames": stats.dropped_non_monotonic,
                "resized_frames": stats.resized_frames,
                "stopped_at_decode_error": stats.stopped_at_decode_error,
            },
            "output": {
                "annotated_fps": fps,
                "frames": writer.frames_written,
                "width": writer.width,
                "height": writer.height,
                "audio": "removed",
            },
            "tracker": {
                "algorithm": "ByteTrack",
                "high": 0.5,
                "low": 0.1,
                "match": 0.8,
                "track_buffer_s": 1.0,
                "association": "per class family: vehicle|bus|truck, bicycle|motorcycle, every other class alone",
            },
            "decision_thresholds": DEFAULT_THRESHOLDS.as_dict(),
            "decision_class_families": dict(CLASS_FAMILIES),
            "light_state_heuristic": {
                "method": LIGHT_STATE_METHOD,
                "note": "classical HSV colour heuristic on the detected traffic-light crop; not model output",
                "thresholds": DEFAULT_LIGHT_THRESHOLDS.as_dict(),
            },
            "frames": frames,
        }
        _write_json(tmp_dir / "timeline.json", timeline)

        total_wall_ms = (time.perf_counter() - wall_start) * 1000.0
        facts = VideoRunFacts(
            job_id=job_id,
            code_commit=settings.git_commit,
            generated_utc=datetime.now(UTC),
            input_sha256=input_sha256,
            input_bytes=input_bytes,
            video=video_meta,
            probe_tool=probe.probe_tool,
            analysis_fps=fps,
            model_id=model.model_id,
            model_sha256=model.sha256,
            model_device=model.device,
            model_precision=model.precision,
            model_revision=info.repository_revision if not model.is_ensemble else None,
            class_map=class_map,
            models=models,
            detection_counts_by_model=dict(model_dets),
            refined_detection_counts=dict(refined_dets),
            track_counts_by_class=track_counts,
            light_state_counts=dict(light_states),
            frame_light_state_counts=dict(frame_light_states),
            decoded_frames=stats.decoded_frames,
            sampled_frames=len(frames),
            analyzed_frames=analyzed,
            failed_frames=failed,
            estimated_pts_frames=stats.estimated_pts_frames,
            output_frames=writer.frames_written,
            roundtrip_ms=roundtrip_ms,
            server_inference_ms=server_ms,
            decode_ms=stats.decode_wall_s * 1000.0,
            postprocess_ms=post_ms,
            encode_ms=encode_ms,
            total_wall_ms=total_wall_ms,
            vehicle_detections=vehicle_dets,
            lane_valid_frames=lane_valid,
            decision_counts=dict(decisions),
            reason_counts=dict(reasons),
            class_counts=dict(classes),
            tracks=tracks,
            limitations=_limitations(
                class_map,
                model,
                fps,
                failed,
                stats.estimated_pts_frames,
                probe.probe_tool,
                settings.build_dirty,
                stats.stopped_at_decode_error,
                stats.decoded_frames,
            ),
            thresholds=DEFAULT_THRESHOLDS.as_dict(),
        )
        report = build_run_report(facts)
        poster_bytes = (tmp_dir / "poster.jpg").read_bytes()
        (tmp_dir / "report.html").write_text(render_report_html(facts, report, poster_bytes), encoding="utf-8")
        _write_json(tmp_dir / "report.json", report)

        if cancelled():
            raise JobCancelled()
        _publish(tmp_dir, out_dir)
        progress(len(frames), len(frames), "done", force=True)

        p50, p95 = percentile(roundtrip_ms, 50), percentile(roundtrip_ms, 95)
        return {
            "analyzed_frames": analyzed,
            "decoded_frames": stats.decoded_frames,
            "coverage_ratio": round(facts.coverage_ratio, 6),
            "duration_s": video_meta["duration_s"],
            "effective_analyzed_fps": round(facts.effective_fps, 4),
            "timing_ms": {
                "decode": round(facts.decode_ms, 3),
                "inference_p50": None if p50 is None else round(p50, 3),
                "inference_p95": None if p95 is None else round(p95, 3),
                "postprocess": round(post_ms, 3),
                "encode": round(encode_ms, 3),
                "total_wall": round(total_wall_ms, 3),
            },
            "decision_counts": dict(decisions),
            "reason_counts": dict(reasons),
            "tracks": tracks,
            "class_counts": dict(classes),
            "audio": "removed",
        }
    except BaseException:
        if writer is not None:
            writer.abort()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
