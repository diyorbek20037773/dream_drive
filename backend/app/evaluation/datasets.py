"""Dataset descriptors for evaluation records (the optional ``Evaluation.dataset`` object, schemas/api.py).

Metadata only. Every value is copied from the committed manifests under datasets/manifests/ and docs/DATA_SOURCES.md;
nothing here is a measurement. The LiteRoad suite uses generated scenarios and no external data.
"""

from __future__ import annotations

from typing import Any

COCO_VAL2017_STREET: dict[str, Any] = {
    "id": "coco_val2017_street_v1",
    "name": "COCO 2017 val street subset",
    "split": "val2017",
    "sample_unit": "image",
    "licence": "CC BY 4.0 (annotations); images keep their per-image Flickr licences",
    "url": "https://cocodataset.org/",
}

KITTI_ROAD_TRAINING: dict[str, Any] = {
    "id": "kitti_road_training_v1",
    "name": "KITTI Road benchmark, training split",
    "split": "training",
    "sample_unit": "image",
    "licence": "CC BY-NC-SA 3.0",
    "url": "https://www.cvlibs.net/datasets/kitti/",
}

ROBOFLOW_LANE_MARKING_V2: dict[str, Any] = {
    "id": "roboflow_lane-marking-9qzw7_v2",
    "name": "Roboflow Universe rtqi/lane-marking-9qzw7 v2",
    "split": "test",
    "sample_unit": "image",
    "licence": "CC BY 4.0",
    "url": "https://universe.roboflow.com/rtqi/lane-marking-9qzw7/dataset/2",
}

LITEROAD_SYNTHETIC_SCENARIOS: dict[str, Any] = {
    "id": "literoad_synthetic_scenarios",
    "name": "LiteRoad synthetic scenarios",
    "split": None,
    "sample_unit": "scenario_run",
    "licence": "generated (no external data)",
    "url": None,
}

_ALL = (COCO_VAL2017_STREET, KITTI_ROAD_TRAINING, ROBOFLOW_LANE_MARKING_V2, LITEROAD_SYNTHETIC_SCENARIOS)
ALL_DATASETS: dict[str, dict[str, Any]] = {d["id"]: d for d in _ALL}


def dataset_ref(dataset_id: str) -> dict[str, Any]:
    """A copy of one descriptor (records must not share mutable dicts)."""
    return dict(ALL_DATASETS[dataset_id])
