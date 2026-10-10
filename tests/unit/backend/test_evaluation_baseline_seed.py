"""Release baseline evaluation records are seeded once and never overwrite runtime evaluations."""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation import registry

BASELINE = registry.BASELINE_DIR


def test_committed_baseline_records_are_valid_evaluations() -> None:
    files = sorted(BASELINE.glob("*.json"))
    assert files, "evaluations/baseline must contain the measured release evaluation records"
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        assert doc["id"] == f.stem
        assert doc["state"] == "succeeded"
        for metric in doc["metrics"].values():
            if metric["status"] == "measured":
                assert metric["value"] is not None and metric["sample_count"] >= 1
            else:
                assert metric["value"] is None and metric["note"]


def test_seed_copies_missing_and_keeps_existing(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"
    first = registry.seed_baseline(root)
    assert first == len(list(BASELINE.glob("*.json")))
    victim = sorted(root.glob("*.json"))[0]
    edited = {**json.loads(victim.read_text(encoding="utf-8")), "state": "runtime"}
    victim.write_text(json.dumps(edited), encoding="utf-8")
    assert registry.seed_baseline(root) == 0
    assert json.loads(victim.read_text(encoding="utf-8"))["state"] == "runtime"
    assert len(registry.list_evaluations(root)) == first


def test_seed_ignores_invalid_names(tmp_path: Path) -> None:
    src = tmp_path / "baseline"
    src.mkdir()
    (src / "not-an-id.json").write_text("{}", encoding="utf-8")
    assert registry.seed_baseline(tmp_path / "root", src) == 0


EXPECTED_DATASETS = {
    "c77b30763a37492888f68c3f9e794d6d": ("coco_val2017_street_v1", "val2017", "image", 300),
    "c023994ee6c34479ac0b4000257697b4": ("kitti_road_training_v1", "training", "image", 289),
    "9e313020dd6c4f4eb89431e1a6c02840": ("roboflow_lane-marking-9qzw7_v2", "test", "image", 150),
    "26f38e26ca9348a0adebad38b14746ab": ("literoad_synthetic_scenarios", None, "scenario_run", 15),
}
MANIFESTS = registry.BASELINE_DIR.parents[1] / "datasets" / "manifests"


def test_baseline_records_carry_dataset_metadata_and_validate() -> None:
    from app.evaluation.datasets import ALL_DATASETS
    from app.schemas.api import Evaluation

    for eval_id, (dataset_id, split, unit, samples) in EXPECTED_DATASETS.items():
        doc = json.loads((BASELINE / f"{eval_id}.json").read_text(encoding="utf-8"))
        parsed = Evaluation.model_validate({k: v for k, v in doc.items() if k != "details"})
        assert parsed.dataset is not None
        assert (parsed.dataset.id, parsed.dataset.split, parsed.dataset.sample_unit) == (dataset_id, split, unit)
        assert doc["dataset"] == ALL_DATASETS[dataset_id]
        assert parsed.sample_count == samples


def test_dataset_descriptors_match_the_committed_manifests() -> None:
    from app.evaluation import datasets

    coco = json.loads((MANIFESTS / "coco_val2017_street_v1.json").read_text(encoding="utf-8"))
    assert coco["manifest_id"] == datasets.COCO_VAL2017_STREET["id"]
    assert coco["source"]["split"] == datasets.COCO_VAL2017_STREET["split"]
    assert coco["source"]["annotation_licence"].startswith("CC BY 4.0")
    kitti = json.loads((MANIFESTS / "kitti_road_training_v1.json").read_text(encoding="utf-8"))
    assert kitti["manifest_id"] == datasets.KITTI_ROAD_TRAINING["id"]
    assert kitti["source"]["licence_id"] == "CC-BY-NC-SA-3.0"
    assert kitti["source"]["licence_url"] == datasets.KITTI_ROAD_TRAINING["url"]
    lane = json.loads((MANIFESTS / "roboflow_lane-marking-9qzw7_v2.json").read_text(encoding="utf-8"))
    assert lane["manifest_id"] == datasets.ROBOFLOW_LANE_MARKING_V2["id"]
    assert lane["split"] == datasets.ROBOFLOW_LANE_MARKING_V2["split"]
    assert lane["source"]["licence"] == datasets.ROBOFLOW_LANE_MARKING_V2["licence"]
    assert lane["source"]["url"] == datasets.ROBOFLOW_LANE_MARKING_V2["url"]


def test_registry_lists_dataset_and_backfills_unmodified_old_copies(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"
    registry.seed_baseline(root)
    assert {d["dataset"]["id"] for d in registry.list_evaluations(root)} == {v[0] for v in EXPECTED_DATASETS.values()}
    # A copy seeded before the dataset field existed gets it; a modified runtime copy is never touched.
    old, edited = sorted(root.glob("*.json"))[:2]
    old_doc = {k: v for k, v in json.loads(old.read_text(encoding="utf-8")).items() if k != "dataset"}
    old.write_text(json.dumps(old_doc), encoding="utf-8")
    edited_doc = {k: v for k, v in json.loads(edited.read_text(encoding="utf-8")).items() if k != "dataset"}
    edited_doc["state"] = "runtime"
    edited.write_text(json.dumps(edited_doc), encoding="utf-8")
    assert registry.seed_baseline(root) == 0
    assert "dataset" in json.loads(old.read_text(encoding="utf-8"))
    after = json.loads(edited.read_text(encoding="utf-8"))
    assert "dataset" not in after and after["state"] == "runtime"
