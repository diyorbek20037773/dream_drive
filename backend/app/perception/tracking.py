"""ByteTrack adapter for uploaded-video analysis.

One `VideoTracker` per job (never shared, never global). ByteTrack itself is class-agnostic, so the adapter runs one
vendored `BYTETracker` per *association group* with a single per-job `IdCounter`: track IDs are unique within the job,
always start at 1 for a new job, and a detection can only continue a track of its own group - a person never
continues a vehicle track, a traffic light never continues a stop-sign track.

Association groups (`ASSOCIATION_GROUPS`, docs/IMPLEMENTATION_SPEC.md section 2.1):
- `vehicle`: vehicle, bus, truck - the inference service relabels a YOLOPv2 `vehicle` box as RF-DETR `bus`/`truck`
  when both models agree (IoU >= 0.5), so the label of one physical vehicle can change between frames;
- `two_wheeler`: bicycle, motorcycle (frequently confused with each other);
- every other verified class name is its own group (person, traffic_light, stop_sign, ...).
Each track keeps per-class hit counts; its reported `class_name` is the majority label (ties -> latest label) and
`class_counts` lists every label it carried, so per-class consistency stays visible.
Detections are associated at their real sampled presentation time, so VFR gaps or skipped samples are respected by
Kalman prediction and lost-track expiry.

Defaults follow docs/IMPLEMENTATION_SPEC.md section 5: high 0.5, low 0.1, match 0.8. Low-score boxes are passed to the
tracker unfiltered so ByteTrack's second association can continue existing tracks through low-confidence frames.
Detections of classes outside the verified class map are never tracked (track_id stays None).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.perception.vendored.bytetrack import BYTETracker, IdCounter


class DetectionLike(Protocol):
    @property
    def class_name(self) -> str: ...

    @property
    def score(self) -> float: ...

    @property
    def xyxy_pixels(self) -> list[float]: ...


@dataclass(frozen=True)
class TrackerConfig:
    track_thresh: float = 0.5
    low_thresh: float = 0.1
    match_thresh: float = 0.8
    track_buffer_s: float = 1.0
    frame_rate: float = 5.0


ASSOCIATION_GROUPS: dict[str, str] = {
    "vehicle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
}


def association_group(class_name: str) -> str:
    """Tracker key for a class name; classes without an entry are their own group."""
    return ASSOCIATION_GROUPS.get(class_name, class_name)


@dataclass
class TrackStats:
    track_id: int
    class_name: str
    first_t: float
    last_t: float
    frames: int = 0
    max_score: float = 0.0
    group: str = ""
    class_counts: dict[str, int] = field(default_factory=dict)
    last_class: str = ""

    def majority_class(self) -> str:
        if not self.class_counts:
            return self.class_name
        top = max(self.class_counts.values())
        winners = [c for c, n in self.class_counts.items() if n == top]
        return self.last_class if self.last_class in winners else sorted(winners)[0]

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_name": self.majority_class(),
            "first_t": round(self.first_t, 4),
            "last_t": round(self.last_t, 4),
            "frames": self.frames,
            "max_score": round(self.max_score, 4),
            "group": self.group or association_group(self.class_name),
            "class_counts": dict(sorted(self.class_counts.items())),
        }


@dataclass
class VideoTracker:
    """Per-job multi-class ByteTrack wrapper. Not thread-safe; one job = one instance."""

    tracked_classes: frozenset[str]
    config: TrackerConfig = field(default_factory=TrackerConfig)
    _ids: IdCounter = field(default_factory=IdCounter, init=False)
    _trackers: dict[str, BYTETracker] = field(default_factory=dict, init=False)
    _stats: dict[int, TrackStats] = field(default_factory=dict, init=False)
    _last_t: float | None = field(default=None, init=False)

    def reset(self) -> None:
        """Forget all tracks and restart IDs at 1 (used when a new, unrelated sequence starts)."""
        self._ids = IdCounter()
        self._trackers = {}
        self._stats = {}
        self._last_t = None

    def _tracker(self, group: str) -> BYTETracker:
        tracker = self._trackers.get(group)
        if tracker is None:
            cfg = self.config
            tracker = BYTETracker(
                track_thresh=cfg.track_thresh,
                low_thresh=cfg.low_thresh,
                match_thresh=cfg.match_thresh,
                frame_rate=cfg.frame_rate,
                track_buffer_s=cfg.track_buffer_s,
                ids=self._ids,
            )
            # A tracker created mid-sequence starts its clock at the current sequence time.
            if self._last_t is not None:
                tracker.last_time_s = self._last_t
                tracker.frame_id = 1  # frame_id==1 would auto-confirm births; later births need a second hit
            self._trackers[group] = tracker
        return tracker

    def update(self, detections: Sequence[DetectionLike], timestamp_s: float) -> list[int | None]:
        """Associate one sampled frame. Returns a track id (or None) aligned with `detections`."""
        if self._last_t is not None and not timestamp_s > self._last_t:
            raise ValueError("timestamps must strictly increase within one tracking run")
        result: list[int | None] = [None] * len(detections)
        by_group: dict[str, list[int]] = {}
        for i, det in enumerate(detections):
            if det.class_name in self.tracked_classes:
                by_group.setdefault(association_group(det.class_name), []).append(i)
        # Every existing group tracker is stepped each frame (empty input ages and expires its lost tracks).
        for group in sorted(set(self._trackers) | set(by_group)):
            idxs = by_group.get(group, [])
            rows = np.asarray(
                [[*detections[i].xyxy_pixels, detections[i].score] for i in idxs], dtype=np.float64
            ).reshape(-1, 5)
            for track in self._tracker(group).update(rows, timestamp_s):
                if 0 <= track.det_index < len(idxs):
                    det_i = idxs[track.det_index]
                    result[det_i] = track.track_id
                    det = detections[det_i]
                    self._record(track.track_id, group, det.class_name, det.score, timestamp_s)
        self._last_t = timestamp_s
        return result

    def _record(self, track_id: int, group: str, class_name: str, score: float, t: float) -> None:
        stats = self._stats.get(track_id)
        if stats is None:
            stats = self._stats[track_id] = TrackStats(track_id, class_name, first_t=t, last_t=t, group=group)
        if stats.group != group:  # defensive: IDs are per job and trackers per group, so this cannot happen
            raise RuntimeError(f"track {track_id} crossed association groups ({stats.group} -> {group})")
        stats.last_t = t
        stats.frames += 1
        stats.max_score = max(stats.max_score, float(score))
        stats.class_counts[class_name] = stats.class_counts.get(class_name, 0) + 1
        stats.last_class = class_name

    @property
    def ids_issued(self) -> int:
        return self._ids.issued

    def track_summaries(self) -> list[dict[str, object]]:
        return [s.as_dict() for s in sorted(self._stats.values(), key=lambda s: s.track_id)]

    def track_counts_by_class(self) -> dict[str, int]:
        """Confirmed tracks per (majority) class name."""
        counts: dict[str, int] = {}
        for stats in self._stats.values():
            name = stats.majority_class()
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))


def tracked_class_names(class_map: dict[str, str] | Iterable[str]) -> frozenset[str]:
    """Verified class names from a model class map ({"3": "vehicle"} -> {"vehicle"})."""
    names = class_map.values() if isinstance(class_map, dict) else class_map
    return frozenset(n for n in names if n and not n.startswith("unverified_class_"))
