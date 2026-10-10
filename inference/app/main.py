"""DreamDrive internal inference service (FastAPI). Internal only: never routed by the public gateway.

Endpoints (docs/IMPLEMENTATION_SPEC.md section 2): /health/live, /health/ready, /v1/model, /v1/infer, /metrics.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

import anyio
import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import metrics
from app.ensemble import ENSEMBLE_ID
from app.logging_setup import configure_logging
from app.model import InferenceBusy, ModelNotReady, YolopV2Engine
from app.postprocess import encode_mask_png
from app.preprocess import ImageInvalid, ImageTooLarge, canvas_to_input, compute_geometry, decode_image, letterbox
from app.schemas import ErrorBody, InferenceResult, ModelInfo
from app.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


def _error(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "message": message}, headers=headers)


async def _read_body_limited(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise ApiError(413, "image_too_large", f"request body exceeds {limit} bytes")
    buf = bytearray()
    async for chunk in request.stream():
        buf.extend(chunk)
        if len(buf) > limit:
            raise ApiError(413, "image_too_large", f"request body exceeds {limit} bytes")
    return bytes(buf)


def create_app(settings: Settings | None = None, engine: YolopV2Engine | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = engine or YolopV2Engine(settings)
    model_label, device_label = settings.model_id, settings.inference_device

    def count(outcome: str) -> None:
        metrics.INFERENCE_REQUESTS.labels(model_label, device_label, outcome).inc()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "inference service starting",
            extra={
                "device": settings.inference_device,
                "precision": settings.inference_precision,
                "model": settings.model_id,
                "git_commit": settings.git_commit,
                "build_version": settings.build_version,
            },
        )
        engine.ensure_load_started()  # background: liveness answers while the checkpoint verifies/loads
        yield

    app = FastAPI(
        title="DreamDrive inference (internal)",
        version=settings.build_version,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.engine = engine
    app.state.settings = settings

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        if request.url.path == "/v1/infer":
            count("request_invalid")
        fields = sorted({".".join(str(p) for p in err.get("loc", ())) for err in exc.errors()})
        return _error(400, "request_invalid", f"invalid request parameters: {', '.join(fields)}")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail))

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", responses={503: {"description": "model not ready"}})
    async def health_ready() -> JSONResponse:
        return JSONResponse(status_code=200 if engine.ready else 503, content=engine.status())

    @app.get("/v1/model", response_model=ModelInfo)
    async def model_info() -> dict[str, Any]:
        return engine.model_info()

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    def run_pipeline(
        body: bytes,
        content_type: str | None,
        frame_id: int,
        conf_threshold: float,
        iou_threshold: float,
        include_masks: bool,
        t_request: float,
    ) -> InferenceResult:
        t0 = time.perf_counter()
        image = decode_image(body, content_type, settings.inference_max_image_side, settings.inference_min_image_side)
        t1 = time.perf_counter()
        if not engine.ready:
            raise ModelNotReady(engine.reason or "loading")
        height, width = image.shape[:2]
        geometry = compute_geometry(width, height)
        tensor = canvas_to_input(letterbox(image, geometry))
        rgb = np.ascontiguousarray(image[:, :, ::-1])  # RF-DETR input: the same decoded frame, RGB uint8
        t2 = time.perf_counter()
        out = engine.infer(tensor, geometry, conf_threshold, iou_threshold, rgb)
        t3 = time.perf_counter()
        drivable_png = encode_mask_png(out.drivable) if include_masks else None
        lane_png = encode_mask_png(out.lane) if include_masks else None
        stats = {"drivable_fraction": float(out.drivable.mean()), "lane_fraction": float(out.lane.mean())}
        t4 = time.perf_counter()
        return InferenceResult.model_validate(
            {
                "frame_id": frame_id,
                "model": engine.identity(),
                "models": engine.identities(),
                "ensemble_id": ENSEMBLE_ID,
                "ensemble_sha256": engine.ensemble_sha256(),
                "image": {"width": width, "height": height},
                "geometry": geometry.to_wire(),
                "detections": [d.to_wire() for d in out.detections],
                "drivable_mask_png": drivable_png,
                "lane_mask_png": lane_png,
                "mask_stats": stats,
                "timing_ms": {
                    "decode": round((t1 - t0) * 1000.0, 3),
                    "preprocess": round((t2 - t1) * 1000.0, 3),
                    "inference": round(out.inference_ms, 3),
                    "postprocess": round(out.postprocess_ms + (t4 - t3) * 1000.0, 3),
                    "inference_by_model": out.inference_ms_by_model,
                    # Wall time inside the service from request arrival, incl. body upload and lock waiting.
                    "total": round((t4 - t_request) * 1000.0, 3),
                },
            }
        )

    @app.post(
        "/v1/infer",
        response_model=InferenceResult,
        responses={
            400: {"model": ErrorBody, "description": "image_invalid | request_invalid"},
            413: {"model": ErrorBody, "description": "image_too_large"},
            503: {"model": ErrorBody, "description": "model_not_ready | inference_busy"},
        },
    )
    async def infer(
        request: Request,
        frame_id: int = Query(ge=0, le=2**31 - 1),
        conf_threshold: float = Query(default=0.1, ge=0.001, le=1.0),
        iou_threshold: float = Query(default=0.45, gt=0.0, le=1.0),
        include_masks: bool = Query(default=True),
    ) -> InferenceResult | JSONResponse:
        t_request = time.perf_counter()
        try:
            body = await _read_body_limited(request, settings.inference_max_image_bytes)
            result = await anyio.to_thread.run_sync(
                partial(
                    run_pipeline,
                    body,
                    request.headers.get("content-type"),
                    frame_id,
                    conf_threshold,
                    iou_threshold,
                    include_masks,
                    t_request,
                )
            )
        except ApiError as exc:
            count(exc.code)
            return _error(exc.status, exc.code, exc.message, exc.headers)
        except ImageTooLarge as exc:
            count("image_too_large")
            return _error(413, "image_too_large", str(exc))
        except ImageInvalid as exc:
            count("image_invalid")
            return _error(400, "image_invalid", str(exc))
        except ModelNotReady as exc:
            count("model_not_ready")
            return _error(503, "model_not_ready", f"model not ready: {exc.reason}", {"Retry-After": "5"})
        except InferenceBusy as exc:
            count("inference_busy")
            return _error(503, "inference_busy", str(exc), {"Retry-After": "1"})
        except Exception:
            count("inference_failed")
            logger.exception("inference failed", extra={"code": "inference_failed", "frame_id": frame_id})
            return _error(500, "inference_failed", "inference failed; see service logs")
        count("ok")
        logger.debug(
            "inference ok",
            extra={
                "frame_id": frame_id,
                "detections": len(result.detections),
                "timing_ms": result.timing_ms.model_dump(),
            },
        )
        return result

    return app


_default_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    """``uvicorn app.main:app`` resolves ``app`` lazily so importing this module (tests) has no side effects."""
    global _default_app
    if name == "app":
        if _default_app is None:
            settings = get_settings()
            configure_logging(settings.log_level)
            _default_app = create_app(settings)
        return _default_app
    raise AttributeError(name)
