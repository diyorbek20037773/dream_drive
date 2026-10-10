"""Real-weight model smoke gate (inference test image + BOTH verified checkpoints mounted at MODEL_ROOT, default
/models, laid out as model-init writes them: /models/yolopv2.pt and /models/rf-detr-nano.pth).

Missing weights FAIL this gate; it never passes by skipping. When DD_EVIDENCE_DIR is set, a JSON evidence file
``model_smoke_<device>.json`` is written at session end, including failures.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = ROOT / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

FIXTURE_IMAGE = ROOT / "tests" / "fixtures" / "images" / "yolopv2_example.jpg"
COCO_IMAGE = ROOT / "tests" / "fixtures" / "images" / "coco_val2017_000000190923.jpg"
COCO_ANNOTATIONS = ROOT / "tests" / "fixtures" / "images" / "coco_val2017_000000190923.annotations.json"

EVIDENCE: dict[str, Any] = {
    "schema_version": "1.0",
    "kind": "model_smoke",
    "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
    "build_version": os.environ.get("BUILD_VERSION", "unknown"),
    "tests": {},
}


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _mem_total_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        return None
    return None


def _dist_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


@pytest.fixture(scope="session")
def smoke_settings() -> Any:
    from app.settings import Settings

    return Settings()  # MODEL_ROOT / INFERENCE_DEVICE / INFERENCE_PRECISION from the environment


@pytest.fixture(scope="session")
def real_engine(smoke_settings: Any) -> Any:
    import torch

    from app.manifest import load_manifest, model_path
    from app.model import YolopV2Engine

    manifest = load_manifest(smoke_settings.model_manifest_path, smoke_settings.model_id)
    rf_manifest = load_manifest(smoke_settings.model_manifest_path, smoke_settings.rfdetr_model_id)
    for m in (manifest, rf_manifest):
        weights = model_path(smoke_settings.model_root, m)
        if not weights.is_file():
            pytest.fail(
                f"REAL WEIGHTS ABSENT: {m.file_name} not found under MODEL_ROOT={smoke_settings.model_root}. "
                "Mount both verified checkpoints (or run `python -m app.model_init`). "
                "This gate never passes by skipping.",
                pytrace=False,
            )
    engine = YolopV2Engine(smoke_settings)
    t0 = time.perf_counter()
    engine.load()
    cold = time.perf_counter() - t0
    EVIDENCE["model"] = {
        "model_id": manifest.model_id,
        "file_name": manifest.file_name,
        "manifest_sha256": manifest.sha256,
        "verified_sha256": engine.verified_sha256,
        "size_bytes": engine.verified_size,
        "repository": manifest.repository,
        "repository_revision": manifest.repository_revision,
        "class_map": manifest.class_map_wire(),
    }
    EVIDENCE["rfdetr_model"] = {
        "model_id": rf_manifest.model_id,
        "file_name": rf_manifest.file_name,
        "manifest_sha256": rf_manifest.sha256,
        "verified_sha256": engine.rfdetr.verified_sha256,
        "size_bytes": engine.rfdetr.verified_size,
        "publisher_md5": rf_manifest.checkpoint_md5,
        "repository": rf_manifest.repository,
        "repository_revision": rf_manifest.repository_revision,
        "class_map": rf_manifest.class_map_wire(),
        "precision_requested": smoke_settings.inference_precision,
        "precision_effective": engine.rfdetr.precision,
        "precision_note": engine.rfdetr.precision_note,
        "load_seconds": {k: round(v, 3) for k, v in engine.rfdetr.timings_s.items()},
    }
    EVIDENCE["ensemble"] = {"ensemble_id": "yolopv2+rfdetr_nano", "ensemble_sha256": engine.ensemble_sha256()}
    EVIDENCE["runtime"] = {
        "device": smoke_settings.inference_device,
        "precision": smoke_settings.inference_precision,
        "torch_version": torch.__version__,
        "rfdetr_version": _dist_version("rfdetr"),
        "transformers_version": _dist_version("transformers"),
        "supervision_version": _dist_version("supervision"),
        "hf_hub_offline_env": os.environ.get("HF_HUB_OFFLINE"),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "accelerator": engine.accelerator_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count_visible": os.cpu_count(),
        "mem_total_gib_visible": _mem_total_gib(),
    }
    EVIDENCE["cold_start_s"] = {"wall_total": round(cold, 3), **{k: round(v, 3) for k, v in engine.timings_s.items()}}
    if not engine.ready:
        EVIDENCE["load_failure"] = engine.status()
        pytest.fail(f"model not ready with real weights: {engine.status()}", pytrace=False)
    return engine


@pytest.fixture(scope="session")
def example_bytes() -> bytes:
    """The upstream example frame (provenance: tests/fixtures/images/README.md)."""
    assert FIXTURE_IMAGE.is_file(), "fixture image missing"
    return FIXTURE_IMAGE.read_bytes()


@pytest.fixture(scope="session")
def evidence() -> dict[str, Any]:
    return EVIDENCE


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call" or report.outcome != "passed":
        name = report.nodeid.split("::")[-1]
        prev = EVIDENCE["tests"].get(name)
        if prev in (None, "passed"):
            EVIDENCE["tests"][name] = report.outcome


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out_dir = os.environ.get("DD_EVIDENCE_DIR")
    if not out_dir:
        return
    EVIDENCE["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    EVIDENCE["pytest_exit_status"] = int(exitstatus)
    EVIDENCE["passed"] = (
        int(exitstatus) == 0 and bool(EVIDENCE["tests"]) and all(v == "passed" for v in EVIDENCE["tests"].values())
    )
    device = EVIDENCE.get("runtime", {}).get("device") or os.environ.get("INFERENCE_DEVICE", "cpu")
    name = os.environ.get("DD_EVIDENCE_NAME") or f"model_smoke_{device}"
    path = Path(out_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(EVIDENCE, indent=2, sort_keys=False) + "\n", encoding="utf-8")


# The CPU-vs-CUDA agreement test only exists for a CUDA smoke; a CPU smoke does not collect it (rather than
# reporting a skip, which must never count as a pass).
collect_ignore = [] if os.environ.get("INFERENCE_DEVICE", "cpu") == "cuda" else ["test_yolopv2_cuda_agreement.py"]
