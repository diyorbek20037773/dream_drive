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
