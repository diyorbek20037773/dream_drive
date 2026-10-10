"""Prometheus metrics of the inference service (names: docs/OBSERVABILITY.md). Labels are bounded enums only."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

# Model forward pass + decode/NMS/mask inversion under the model lock. Excludes lock waiting, image decode,
# PNG encoding and HTTP transfer (those are in the per-request timing_ms breakdown).
INFERENCE_DURATION = Histogram(
    "dreamdrive_inference_duration_seconds",
    "Duration of one model inference (forward + decode + mask inversion), excluding queueing",
    ["model", "device", "outcome"],
    buckets=(0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
    registry=REGISTRY,
)

MODEL_READY = Gauge(
    "dreamdrive_model_ready",
    "1 when the verified model is loaded and warmed up, else 0",
    ["model", "device"],
    registry=REGISTRY,
)

MODEL_LOAD_SECONDS = Gauge(
    "dreamdrive_model_load_seconds",
    "Cold-start seconds of the last successful load (verify + torch.jit.load + warm-up)",
    ["model", "device"],
    registry=REGISTRY,
)

INFERENCE_REQUESTS = Counter(
    "dreamdrive_inference_requests_total",
    "POST /v1/infer requests by outcome",
    ["model", "device", "outcome"],
    registry=REGISTRY,
)
