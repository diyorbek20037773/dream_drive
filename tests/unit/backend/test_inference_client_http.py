"""InferenceClient error mapping, retry/backoff and response validation (httpx MockTransport, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.perception.client import (
    InferenceBusy,
    InferenceClient,
    InferenceClientError,
    InferenceError,
    InferenceTimeout,
    InferenceUnavailable,
)

SHA = "f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c"


def _result(frame_id: int = 7, **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "1.0",
        "frame_id": frame_id,
        "model": {"model_id": "yolopv2", "sha256": SHA, "device": "cpu", "precision": "fp32"},
        "image": {"width": 1280, "height": 720},
        "geometry": {
            "canvas_width": 640,
            "canvas_height": 384,
            "scale": 0.5,
            "pad_left": 0,
            "pad_top": 12,
            "pad_right": 0,
            "pad_bottom": 12,
        },
        "detections": [{"class_id": 3, "class_name": "vehicle", "score": 0.9, "xyxy_pixels": [10, 20, 110, 90]}],
        "drivable_mask_png": "iVBORw0KGgo=",
        "lane_mask_png": "iVBORw0KGgo=",
        "mask_stats": {"drivable_fraction": 0.2, "lane_fraction": 0.03},
        "timing_ms": {"decode": 1, "preprocess": 2, "inference": 400, "postprocess": 20, "total": 430},
    }
    body.update(over)
    return body


def _client(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> tuple[InferenceClient, list[float]]:
    sleeps: list[float] = []
    client = InferenceClient(
        "http://inference:8001/", 5.0, transport=httpx.MockTransport(handler), sleep=sleeps.append, **kw
    )
    return client, sleeps


def _err(status: int, code: str, message: str = "") -> httpx.Response:
    return httpx.Response(status, json={"code": code, "message": message})


def test_infer_success_sends_contract_request() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        seen["ctype"] = request.headers["content-type"]
        seen["body"] = request.content
        return httpx.Response(200, json=_result(7))

    client, _ = _client(handler)
    result = client.infer(b"\xff\xd8jpegbytes", frame_id=7, conf_threshold=0.1)
    assert seen == {
        "path": "/v1/infer",
        "params": {"frame_id": "7", "conf_threshold": "0.1", "include_masks": "true"},
        "ctype": "image/jpeg",
        "body": b"\xff\xd8jpegbytes",
    }
    assert result.frame_id == 7 and result.model.sha256 == SHA
    assert result.detections[0].class_name == "vehicle"


def test_model_not_ready_maps_to_unavailable() -> None:
    client, sleeps = _client(lambda r: _err(503, "model_not_ready", "model not ready: model_missing"))
    with pytest.raises(InferenceUnavailable) as exc:
        client.infer(b"x", 1)
    assert exc.value.code == "model_not_ready" and exc.value.status_code == 503
    assert sleeps == []  # not retried: a missing model is not a transient queue condition


def test_busy_is_retried_with_bounded_backoff_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _err(503, "inference_busy") if calls["n"] <= 2 else httpx.Response(200, json=_result(3))

    client, sleeps = _client(handler)
    assert client.infer(b"x", 3).frame_id == 3
    assert calls["n"] == 3
    assert sleeps == [0.25, 0.5]


def test_busy_gives_up_after_retries() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _err(503, "inference_busy", "waiters exceeded")

    client, sleeps = _client(handler, busy_retries=3)
    with pytest.raises(InferenceBusy):
        client.infer(b"x", 1)
    assert calls["n"] == 4
    assert sleeps == [0.25, 0.5, 1.0]


def test_timeout_maps_to_inference_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client, _ = _client(handler)
    with pytest.raises(InferenceTimeout) as exc:
        client.infer(b"x", 1)
    assert isinstance(exc.value, InferenceClientError)


def test_connection_error_maps_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client, _ = _client(handler)
    with pytest.raises(InferenceUnavailable) as exc:
        client.infer(b"x", 1)
    assert exc.value.code == "inference_unreachable"
    with pytest.raises(InferenceUnavailable):
        client.model_info()
    assert client.ready() is False


@pytest.mark.parametrize(
    ("status", "code"), [(400, "image_invalid"), (413, "image_too_large"), (500, "inference_failed")]
)
def test_other_errors_map_to_inference_error(status: int, code: str) -> None:
    client, _ = _client(lambda r: _err(status, code))
    with pytest.raises(InferenceError) as exc:
        client.infer(b"x", 1)
    assert exc.value.code == code and exc.value.status_code == status


def test_fastapi_detail_body_is_understood() -> None:
    client, _ = _client(lambda r: httpx.Response(503, json={"detail": {"code": "model_not_ready"}}))
    with pytest.raises(InferenceUnavailable):
        client.infer(b"x", 1)
    client2, _ = _client(lambda r: httpx.Response(422, json={"detail": "frame_id must be int"}))
    with pytest.raises(InferenceError):
        client2.infer(b"x", 1)


def test_invalid_json_and_schema_mismatch_are_rejected() -> None:
    client, _ = _client(lambda r: httpx.Response(200, content=b"<html>proxy error</html>"))
    with pytest.raises(InferenceError) as exc:
        client.infer(b"x", 1)
    assert exc.value.code == "invalid_response"
    bad = _result(1)
    bad["model"]["sha256"] = "not-a-sha"
    client2, _ = _client(lambda r: httpx.Response(200, json=bad))
    with pytest.raises(InferenceError):
        client2.infer(b"x", 1)


def test_frame_id_echo_is_verified() -> None:
    client, _ = _client(lambda r: httpx.Response(200, json=_result(99)))
    with pytest.raises(InferenceError, match="frame_id mismatch"):
        client.infer(b"x", 1)


def test_missing_masks_when_requested_is_an_error() -> None:
    client, _ = _client(lambda r: httpx.Response(200, json=_result(1, lane_mask_png=None)))
    with pytest.raises(InferenceError):
        client.infer(b"x", 1)
    ok, _ = _client(lambda r: httpx.Response(200, json=_result(1, lane_mask_png=None, drivable_mask_png=None)))
    assert ok.infer(b"x", 1, include_masks=False).lane_mask_png is None


def test_out_of_image_box_is_rejected() -> None:
    body = _result(1)
    body["detections"][0]["xyxy_pixels"] = [10, 20, 1500, 90]
    client, _ = _client(lambda r: httpx.Response(200, json=body))
    with pytest.raises(InferenceError):
        client.infer(b"x", 1)


def test_ready_and_model_info() -> None:
    model = {
        "model_id": "yolopv2",
        "ready": True,
        "reason": None,
        "sha256": SHA,
        "size_bytes": 156380200,
        "repository_revision": "9388c6aa55f3bf5a878679a1338bb090cff6a256",
        "class_map": {"3": "vehicle"},
        "device": "cpu",
        "precision": "fp32",
        "torch_version": "2.14.1+cpu",
        "input_policy": "letterbox640_stride32",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"ready": True, "reason": None})
        if request.url.path == "/v1/model":
            return httpx.Response(200, json=model)
        return httpx.Response(404)

    client, _ = _client(handler)
    assert client.ready() is True
    info = client.model_info()
    assert info.class_map == {"3": "vehicle"} and info.sha256 == SHA
    not_ready, _ = _client(lambda r: httpx.Response(503, json={"ready": False, "reason": "model_missing"}))
    assert not_ready.ready() is False
    weird, _ = _client(lambda r: httpx.Response(200, content=json.dumps({"ready": "yes"}).encode()))
    assert weird.ready() is False  # only a literal true counts


def test_argument_validation() -> None:
    client, _ = _client(lambda r: httpx.Response(200, json=_result(1)))
    with pytest.raises(ValueError):
        client.infer(b"x", -1)
    with pytest.raises(ValueError):
        client.infer(b"x", 1, content_type="image/gif")
    with pytest.raises(ValueError):
        InferenceClient("http://x", 0)


def test_context_manager_closes() -> None:
    with InferenceClient("http://x", 1.0, transport=httpx.MockTransport(lambda r: httpx.Response(200))) as c:
        assert c.base_url == "http://x"
    assert c._http.is_closed
