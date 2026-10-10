"""Unit tests of the inference service. Run inside the inference test image (torch is linux-only locked).

Puts ``<root>/inference`` (repo) or ``/app/inference`` (image) on sys.path; identical relative layout.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
INFERENCE_DIR = ROOT / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

REPO_MANIFEST = ROOT / "config" / "models.resolved.yaml"
FIXTURES = ROOT / "tests" / "fixtures" / "images"

# Enabled RF-DETR classes exactly as docs/IMPLEMENTATION_SPEC.md section 2.1 specifies (DreamDrive id -> name).
SPEC_RFDETR_CLASS_MAP = {
    "101": "person",
    "102": "bicycle",
    "104": "motorcycle",
    "106": "bus",
    "108": "truck",
    "110": "traffic_light",
    "113": "stop_sign",
}
ABSENT_SHA = "0" * 64


def rfdetr_entry(
    *,
    file_name: str = "rf-detr-nano.pth",
    sha256: str = ABSENT_SHA,
    size: int = 1,
    url: str = "https://example.invalid/rf.pth",
    md5: str | None = None,
    class_map: dict[str, str] | None = None,
) -> dict[str, object]:
    """RF-DETR manifest entry in the repository schema (by default naming a file tests never create)."""
    entry: dict[str, object] = {
        "id": "rfdetr_nano",
        "file_name": file_name,
        "repository": "https://github.com/roboflow/rf-detr",
        "repository_revision": "7433d5919d0f7e72cc6cb78663e0a731afd499e0",
        "checkpoint_url": url,
        "checkpoint_sha256": sha256,
        "checkpoint_size_bytes": size,
        "format": "rfdetr_checkpoint_pth",
        "source_label_space": "coco2017_category_id",
        "class_id_offset": 100,
        "input_policy_id": "rfdetr_resize384_imagenet_norm",
        "supported_class_map": class_map or dict(SPEC_RFDETR_CLASS_MAP),
    }
    if md5 is not None:
        entry["checkpoint_md5_publisher"] = md5
    return entry


def write_manifest(
    path: Path,
    *,
    file_name: str,
    sha256: str,
    size: int,
    url: str = "https://example.invalid/m.pt",
    class_map: dict[str, str] | None = None,
    rfdetr: dict[str, object] | None = None,
) -> Path:
    """A minimal manifest in the same schema as config/models.resolved.yaml (JSON is valid YAML).

    It always contains both required models; unless given, the RF-DETR entry names a file that tests never create.
    """
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "models": [
                    {
                        "id": "yolopv2",
                        "file_name": file_name,
                        "repository": "https://github.com/CAIC-AD/YOLOPv2",
                        "repository_revision": "9388c6aa55f3bf5a878679a1338bb090cff6a256",
                        "checkpoint_url": url,
                        "checkpoint_sha256": sha256,
                        "checkpoint_size_bytes": size,
                        "format": "torchscript",
                        "supported_class_map": class_map or {"3": "vehicle"},
                    },
                    rfdetr if rfdetr is not None else rfdetr_entry(),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class SyntheticYolopLike(torch.nn.Module):
    """A tiny *synthetic test* TorchScript module with YOLOPv2's output structure. NOT a model; never shipped.

    Outputs are pure functions of the input pixels so tests can exercise the service plumbing (verification, load,
    request handling, geometry inversion) deterministically: drivable channel 1 = red > 0.5, lane = green, and one
    class-3 detection from anchor 0 of the stride-8 head at cell (gy=10, gx=12): centre (100, 84) canvas px,
    size = anchor (40, 24) px. Real-weight behaviour is covered by tests/model_smoke.
    """

    def forward(self, x: torch.Tensor) -> tuple[list[list[torch.Tensor]], torch.Tensor, torch.Tensor]:
        b, h, w = x.shape[0], x.shape[2], x.shape[3]
        p0 = torch.full((b, 255, h // 8, w // 8), -20.0, dtype=x.dtype, device=x.device)
        p1 = torch.full((b, 255, h // 16, w // 16), -20.0, dtype=x.dtype, device=x.device)
        p2 = torch.full((b, 255, h // 32, w // 32), -20.0, dtype=x.dtype, device=x.device)
        p0[:, 0:4, 10, 12] = 0.0  # sigmoid(0)=0.5 -> xy = (g + 0.5) * 8, wh = anchor
        p0[:, 4, 10, 12] = 12.0  # objectness
        p0[:, 5 + 3, 10, 12] = 12.0  # class 3
        g0 = torch.tensor([40.0, 24.0, 10.0, 10.0, 10.0, 10.0], dtype=x.dtype, device=x.device).view(1, 3, 1, 1, 2)
        g1 = torch.ones((1, 3, 1, 1, 2), dtype=x.dtype, device=x.device)
        g2 = torch.ones((1, 3, 1, 1, 2), dtype=x.dtype, device=x.device)
        red = x[:, 0:1]
        seg = torch.cat([(red <= 0.5).to(x.dtype), (red > 0.5).to(x.dtype)], dim=1)
        ll = x[:, 1:2].clone()
        return [[p0, p1, p2], [g0, g1, g2]], seg, ll


SYNTHETIC_BOX_CANVAS = (80.0, 72.0, 120.0, 96.0)  # xyxy of the synthetic module's detection, canvas pixels


def build_synthetic_checkpoint(path: Path) -> tuple[str, int]:
    torch.jit.script(SyntheticYolopLike()).save(str(path))
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


@pytest.fixture(scope="session")
def synthetic_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, int]:
    root = tmp_path_factory.mktemp("synthetic_model_root")
    path = root / "yolopv2.pt"
    sha, size = build_synthetic_checkpoint(path)
    return path, sha, size


class SyntheticRfDetrRuntime:
    """A *synthetic test* stand-in for the constructed RF-DETR (app.rfdetr_detector.DetectorRuntime). NOT a model.

    It returns fixed boxes (fractions of the frame) with raw COCO category ids, like rfdetr's predict() does for
    COCO-pretrained checkpoints: person 0.90, traffic light 0.80, truck 0.70 placed exactly on the synthetic YOLOPv2
    vehicle box (class refinement), car 0.95 and airplane 0.90 (both disabled, must be dropped), stop sign 0.05
    (below the default threshold) and a person box fully outside the frame (must be dropped). Its class list is
    rfdetr's own COCO_CLASSES. Real-weight behaviour is covered by tests/model_smoke.
    """

    precision = "fp32"

    def __init__(self, path: Path) -> None:
        assert path.is_file()
        self.calls = 0

    def coco_category_names(self) -> dict[int, str]:
        from rfdetr.assets.coco_classes import COCO_CLASSES

        return dict(COCO_CLASSES)

    def predict(self, rgb: Any, threshold: float) -> tuple[Any, Any, Any]:
        import numpy as np

        from app.postprocess import invert_boxes
        from app.preprocess import compute_geometry

        self.calls += 1
        h, w = rgb.shape[:2]
        vehicle = invert_boxes(np.array([SYNTHETIC_BOX_CANVAS]), compute_geometry(w, h))[0]
        rows = [
            ((0.1 * w, 0.2 * h, 0.2 * w, 0.6 * h), 0.90, 1),
            ((0.7 * w, 0.05 * h, 0.75 * w, 0.2 * h), 0.80, 10),
            (tuple(vehicle), 0.70, 8),
            ((0.3 * w, 0.3 * h, 0.5 * w, 0.5 * h), 0.95, 3),
            ((0.5 * w, 0.1 * h, 0.6 * w, 0.2 * h), 0.90, 5),
            ((0.4 * w, 0.6 * h, 0.45 * w, 0.65 * h), 0.05, 13),
            ((w + 10.0, h + 10.0, w + 50.0, h + 50.0), 0.85, 1),
        ]
        xyxy = np.array([r[0] for r in rows], dtype=np.float64)
        scores = np.array([r[1] for r in rows], dtype=np.float64)
        cls = np.array([r[2] for r in rows], dtype=np.int64)
        keep = scores > threshold
        return xyxy[keep], scores[keep], cls[keep]


def synthetic_rfdetr_builder(
    path: Path, device: str, precision: str, verified_md5: str | None = None
) -> SyntheticRfDetrRuntime:
    return SyntheticRfDetrRuntime(path)


@pytest.fixture(scope="session")
def synthetic_rfdetr_checkpoint(synthetic_checkpoint: tuple[Path, str, int]) -> tuple[Path, str, int]:
    """Placeholder bytes verified like a checkpoint; the synthetic builder never deserialises them."""
    path = synthetic_checkpoint[0].parent / "rf-detr-nano.pth"
    data = b"synthetic rf-detr test stand-in, not a model. " * 64
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest(), len(data)


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., object]:
    from app.settings import Settings

    def _make(**overrides: object) -> Settings:
        values: dict[str, object] = {
            "model_root": tmp_path / "models",
            "model_manifest_path": REPO_MANIFEST,
            "inference_device": "cpu",
            "inference_precision": "fp32",
            "inference_warmup_runs": 1,
        }
        values.update(overrides)
        return Settings(**values)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def synthetic_manifest(
    tmp_path: Path, synthetic_checkpoint: tuple[Path, str, int], synthetic_rfdetr_checkpoint: tuple[Path, str, int]
) -> Path:
    """Manifest naming both synthetic files (verified by their real SHA256/size)."""
    path, sha, size = synthetic_checkpoint
    rf_path, rf_sha, rf_size = synthetic_rfdetr_checkpoint
    return write_manifest(
        tmp_path / "manifest.yaml",
        file_name=path.name,
        sha256=sha,
        size=size,
        rfdetr=rfdetr_entry(file_name=rf_path.name, sha256=rf_sha, size=rf_size),
    )


@pytest.fixture
def synthetic_ready_client(
    synthetic_checkpoint: tuple[Path, str, int],
    synthetic_manifest: Path,
    make_settings: Callable[..., object],
) -> Iterator[object]:
    """TestClient over an engine that loaded both synthetic test stand-ins (verified by their own manifest)."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.model import YolopV2Engine

    settings = make_settings(
        model_root=synthetic_checkpoint[0].parent,
        model_manifest_path=synthetic_manifest,
        inference_max_waiters=0,
        inference_lock_timeout_s=10.0,
    )
    engine = YolopV2Engine(settings, rfdetr_builder=synthetic_rfdetr_builder)  # type: ignore[arg-type]
    with TestClient(create_app(settings, engine)) as client:  # type: ignore[arg-type]
        assert engine.wait_settled(60)
        assert engine.ready, engine.status()
        yield client
