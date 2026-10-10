"""Class-aware association for the two-model detector set (YOLOPv2 vehicle + RF-DETR-N classes, spec section 2.1).

A detection may only continue a track of its own association group: a person never continues a vehicle track even
when its box is identical; bus/truck relabels of a YOLOPv2 vehicle continue the same track; per-class counts are kept.
"""

from __future__ import annotations

from app.perception.tracking import (
    ASSOCIATION_GROUPS,
    TrackerConfig,
    VideoTracker,
    association_group,
    tracked_class_names,
)
from app.perception.world_state import TrackedDetection

CLASS_IDS = {
    "vehicle": 3,
    "person": 101,
    "bicycle": 102,
    "motorcycle": 104,
    "bus": 106,
    "truck": 108,
    "traffic_light": 110,
    "stop_sign": 113,
}
ENSEMBLE_CLASS_MAP = {str(v): k for k, v in CLASS_IDS.items()}


def _det(
    cls: str, cx: float, cy: float = 200.0, w: float = 60.0, h: float = 40.0, score: float = 0.9
) -> TrackedDetection:
    return TrackedDetection(CLASS_IDS.get(cls, 0), cls, score, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _tracker() -> VideoTracker:
    return VideoTracker(tracked_class_names(ENSEMBLE_CLASS_MAP), TrackerConfig(frame_rate=5.0))


def test_association_groups() -> None:
    assert {association_group(c) for c in ("vehicle", "bus", "truck")} == {"vehicle"}
    assert {association_group(c) for c in ("bicycle", "motorcycle")} == {"two_wheeler"}
    for cls in ("person", "traffic_light", "stop_sign"):
        assert association_group(cls) == cls
        assert cls not in ASSOCIATION_GROUPS
    assert tracked_class_names(ENSEMBLE_CLASS_MAP) == frozenset(CLASS_IDS)


def test_person_never_continues_a_vehicle_track() -> None:
    tr = _tracker()
    vehicle_ids = [tr.update([_det("vehicle", 100 + 5 * k)], 0.2 * k)[0] for k in range(4)]
    assert set(vehicle_ids) == {1}
    # The vehicle disappears and a person box appears at exactly the predicted vehicle position.
    person_ids = [tr.update([_det("person", 120 + 5 * k)], 0.8 + 0.2 * k)[0] for k in range(1, 4)]
    assert 1 not in person_ids
    assert all(pid is None or pid > 1 for pid in person_ids)
    summaries = {s["track_id"]: s for s in tr.track_summaries()}
    assert summaries[1]["class_name"] == "vehicle" and summaries[1]["class_counts"] == {"vehicle": 4}
    for tid, s in summaries.items():
        if tid != 1:
            assert s["class_name"] == "person" and s["group"] == "person"
            assert set(s["class_counts"]) == {"person"}  # type: ignore[arg-type]


def test_overlapping_person_and_vehicle_get_distinct_tracks_every_frame() -> None:
    tr = _tracker()
    for k in range(6):
        ids = tr.update([_det("vehicle", 300 + 4 * k), _det("person", 300 + 4 * k, score=0.8)], 0.2 * k)
        if k > 0:
            assert ids[0] is not None and ids[1] is not None and ids[0] != ids[1]
    by_class = tr.track_counts_by_class()
    assert by_class == {"person": 1, "vehicle": 1}


def test_bus_truck_relabel_continues_the_vehicle_track_and_keeps_class_counts() -> None:
    tr = _tracker()
    labels = ["vehicle", "truck", "truck", "vehicle", "truck", "bus"]
    ids = [tr.update([_det(cls, 200 + 6 * k)], 0.2 * k)[0] for k, cls in enumerate(labels)]
    assert set(ids) == {1}
    (summary,) = tr.track_summaries()
    assert summary["group"] == "vehicle"
    assert summary["class_counts"] == {"bus": 1, "truck": 3, "vehicle": 2}
    assert summary["class_name"] == "truck"  # majority label
    assert tr.track_counts_by_class() == {"truck": 1}


def test_majority_tie_prefers_latest_label() -> None:
    tr = _tracker()
    for k, cls in enumerate(["bicycle", "motorcycle"]):
        tr.update([_det(cls, 200 + 3 * k)], 0.2 * k)
    (summary,) = tr.track_summaries()
    assert summary["group"] == "two_wheeler" and summary["class_name"] == "motorcycle"


def test_traffic_light_and_stop_sign_tracks_are_separate() -> None:
    tr = _tracker()
    for k in range(4):
        ids = tr.update(
            [_det("traffic_light", 600, 80, 16, 40), _det("stop_sign", 600, 80, 30, 30, score=0.7)], 0.2 * k
        )
        if k > 0:
            assert None not in ids and ids[0] != ids[1]
    assert tr.track_counts_by_class() == {"stop_sign": 1, "traffic_light": 1}


def test_classes_outside_the_service_class_map_are_not_tracked() -> None:
    tr = VideoTracker(tracked_class_names({"3": "vehicle"}), TrackerConfig(frame_rate=5.0))  # YOLOPv2-only service
    for k in range(3):
        assert tr.update([_det("person", 200 + k)], 0.2 * k) == [None]
    assert tr.track_summaries() == []
