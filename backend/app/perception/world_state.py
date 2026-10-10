"""Build validated `WorldState` records for uploaded-video analysis.

Uploaded video is an observation recording: mode `video_analysis`, no ego localisation (`ego=None`), no actuator
command (`command=None`), decisions are `analytical_only`, the clock is the video PTS. Perception is tagged with its
source: `learned_model` with the model id and checkpoint SHA256 when inference succeeded, `unavailable` (no model
identity) for frames whose inference failed. With the two-model service (YOLOPv2 + RF-DETR-N, spec section 2.1) the
identity is the ensemble: `model_id` = ensemble id ("yolopv2+rfdetr_nano") and `model_sha256` = ensemble sha256
(sha256 over each model's checkpoint sha256, see app.perception.client.ensemble_sha256); a single-model service keeps
its own model id and checkpoint sha256 (app.perception.client.perception_identity decides).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from app.perception.video_decisions import VideoDecision
from app.schemas.world_state import Decision, Detection, FrameInfo, Perception, WorldState


@dataclass(frozen=True)
class TrackedDetection:
    """A detector box in original image pixels with the ByteTrack id assigned in this frame (None = untracked)."""

    class_id: int
    class_name: str
    score: float
    xyxy_pixels: list[float]
    track_id: int | None = None
    # Provenance from a multi-model service (spec section 2.1); not part of the WorldState contract, emitted in the
    # timeline's per-frame `detection_sources` instead.
    model_id: str | None = None
    refined_from_model: str | None = None


@dataclass(frozen=True)
class ModelRef:
    model_id: str
    sha256: str


def clip_box(xyxy: Sequence[float], width: int, height: int) -> list[float]:
    """Clip to the image and keep x1<=x2, y1<=y2 (contract semantic rule); 2-decimal pixel precision."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    x1, x2 = sorted((min(max(x1, 0.0), float(width)), min(max(x2, 0.0), float(width))))
    y1, y2 = sorted((min(max(y1, 0.0), float(height)), min(max(y2, 0.0), float(height))))
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def build_video_world_state(
    *,
    job_id: str,
    frame_id: int,
    pts_s: float,
    acquired_utc: datetime,
    width: int,
    height: int,
    decision: VideoDecision,
    detections: Sequence[TrackedDetection],
    model: ModelRef | None,
) -> WorldState:
    """`model=None` marks a frame without learned perception (inference failed); detections must then be empty."""
    if model is None and detections:
        raise ValueError("detections without a model identity are not allowed")
    perception = Perception(
        source="learned_model" if model is not None else "unavailable",
        model_id=model.model_id if model else None,
        model_sha256=model.sha256 if model else None,
        detections=[
            Detection(
                track_id=d.track_id,
                class_id=d.class_id,
                class_name=d.class_name,
                score=round(float(d.score), 4),
                xyxy_pixels=clip_box(d.xyxy_pixels, width, height),
            )
            for d in detections
        ],
        # Masks are not persisted per frame; they are rendered into annotated.mp4 (see timeline output_frame_index).
        lane_mask_ref=None,
        drivable_mask_ref=None,
        lane_status=decision.lane_status,
        image_lane_offset_normalized=decision.image_lane_offset_normalized,
        image_obstacle_cue=decision.image_obstacle_cue,
        ttc_seconds=None,
        uncertainty_reasons=list(decision.uncertainty_reasons),
    )
    state = WorldState(
        run_id=job_id,
        mode="video_analysis",
        frame=FrameInfo(
            id=frame_id,
            timestamp_seconds=max(0.0, round(float(pts_s), 6)),
            clock="video_pts",
            acquired_utc=acquired_utc,
            width=width,
            height=height,
            quality_status=decision.quality.status,
        ),
        perception=perception,
        ego=None,
        decision=Decision(
            state=decision.state, reason_codes=list(decision.reason_codes), action_kind="analytical_only"
        ),
        command=None,
    )
    # Round-trip through validation so every emitted record satisfies the contract-level validators.
    return WorldState.model_validate(state.model_dump())
