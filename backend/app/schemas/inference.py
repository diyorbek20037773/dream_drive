"""Wire models for the internal inference service (see docs/IMPLEMENTATION_SPEC.md §2).

The inference service (inference/app) defines the same shapes independently; contract tests compare them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelIdentity(BaseModel):
    model_id: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    device: Literal["cpu", "cuda"]
    precision: Literal["fp32", "fp16"]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    ready: bool
    reason: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    repository_revision: str | None = None
    class_map: dict[str, str] = Field(default_factory=dict)
    device: str | None = None
    precision: str | None = None
    torch_version: str | None = None
    input_policy: str | None = None


class ImageSize(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class LetterboxGeometry(BaseModel):
    canvas_width: int
    canvas_height: int
    scale: float = Field(gt=0)
    pad_left: int = Field(ge=0)
    pad_top: int = Field(ge=0)
    pad_right: int = Field(ge=0)
    pad_bottom: int = Field(ge=0)


class RawDetection(BaseModel):
    class_id: int = Field(ge=0)
    class_name: str
    score: float = Field(ge=0, le=1)
    xyxy_pixels: list[float] = Field(min_length=4, max_length=4)


class MaskStats(BaseModel):
    drivable_fraction: float = Field(ge=0, le=1)
    lane_fraction: float = Field(ge=0, le=1)


class InferenceTiming(BaseModel):
    decode: float
    preprocess: float
    inference: float
    postprocess: float
    total: float


class InferenceResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal["1.0"] = "1.0"
    frame_id: int = Field(ge=0)
    model: ModelIdentity
    image: ImageSize
    geometry: LetterboxGeometry
    detections: list[RawDetection]
    drivable_mask_png: str | None
    lane_mask_png: str | None
    mask_stats: MaskStats
    timing_ms: InferenceTiming
