"""H.264 output (playable, CFR, faststart, no audio, even size, reproducible) and overlay rendering."""

from __future__ import annotations

import hashlib
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from app.perception.video_decisions import decide
from app.perception.world_state import TrackedDetection
from app.video.annotate import (
    DRIVABLE_ALPHA,
    DRIVABLE_RGB,
    LANE_RGB,
    HudInfo,
    annotate_frame,
    ascii_text,
)
from app.video.encode import H264Writer, encode_jpeg_bytes, even_size, pad_to_even, write_jpeg


def _frames(n: int, w: int, h: int) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        img[: h // 4, : (i * 7) % w] = (255, 0, 0)
        out.append(img)
    return out


def _write(path: Path, frames: list[np.ndarray], fps: float) -> None:
    h, w = frames[0].shape[:2]
    with H264Writer(path, w, h, fps) as writer:
        for f in frames:
            writer.write(f)


def test_output_is_browser_playable_h264(tmp_path: Path) -> None:
    path = tmp_path / "annotated.mp4"
    # 60 frames > x264 lookahead: packets are muxed while frames are still being fed (regression for the
    # stream-time-base rewrite done by the mp4 muxer after the header is written).
    _write(path, _frames(60, 320, 180), fps=5.0)
    data = path.read_bytes()
    assert 0 < data.find(b"moov") < data.find(b"mdat"), "+faststart must place moov before mdat"
    with av.open(str(path)) as container:
        assert "mp4" in container.format.name
        assert len(container.streams.audio) == 0
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.pix_fmt == "yuv420p"
        times = [float(f.time) for f in container.decode(stream)]
        duration_s = container.duration / 1e6
    assert len(times) == 60
    assert times == pytest.approx([i / 5.0 for i in range(60)], abs=1e-3)
    assert duration_s == pytest.approx(12.0, abs=0.01)


def test_odd_size_is_padded_to_even(tmp_path: Path) -> None:
    assert even_size(101, 77) == (102, 78)
    frame = np.zeros((77, 101, 3), dtype=np.uint8)
    frame[-1, :] = 200
    padded = pad_to_even(frame)
    assert padded.shape == (78, 102, 3)
    assert (padded[-1] == padded[-2]).all()  # replicated edge, no rescaling
    path = tmp_path / "odd.mp4"
    _write(path, _frames(4, 101, 77), fps=5.0)
    with av.open(str(path)) as container:
        frame0 = next(container.decode(video=0))
        assert (frame0.width, frame0.height) == (102, 78)


def test_encoding_is_reproducible(tmp_path: Path) -> None:
    frames = _frames(20, 160, 90)
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _write(a, frames, 5.0)
    _write(b, frames, 5.0)
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def test_jpeg_helpers_roundtrip(tmp_path: Path) -> None:
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:, :30] = (255, 0, 0)  # red left half (RGB)
    decoded = cv2.imdecode(np.frombuffer(encode_jpeg_bytes(img, 95), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (40, 60, 3)
    assert decoded[20, 10, 2] > 200 and decoded[20, 10, 0] < 60  # BGR: red channel high -> RGB order preserved
    write_jpeg(tmp_path / "p.jpg", img)
    assert cv2.imread(str(tmp_path / "p.jpg")) is not None


def test_annotation_overlays(tmp_path: Path) -> None:
    h, w = 180, 320
    rng = np.random.default_rng(3)
    image = rng.integers(90, 160, size=(h, w, 3), dtype=np.uint8)
    drivable = np.zeros((h, w), dtype=bool)
    drivable[120:, :] = True
    lane = np.zeros((h, w), dtype=bool)
    lane[110:170, 60:63] = True
    lane[110:170, 257:260] = True
    dets = [
        TrackedDetection(3, "vehicle", 0.91, [130.0, 100.0, 190.0, 150.0], track_id=1),
        TrackedDetection(3, "vehicle", 0.2, [10.0, 90.0, 40.0, 110.0], track_id=None),  # untracked low score
    ]
    decision = decide(image, drivable, lane, dets)
    out = annotate_frame(
        image,
        drivable_mask=drivable,
        lane_mask=lane,
        detections=dets,
        decision=decision,
        hud=HudInfo("YOLOPv2", "learned_model", "CPU"),
        pts_s=1.234,
        sample_index=6,
    )
    assert out.shape == image.shape and out.dtype == np.uint8
    assert not np.shares_memory(out, image)
    # Drivable blend: a pixel away from text/boxes/corridor edge moves towards green by alpha 0.35.
    y, x = 175, 300
    expected = image[y, x] * (1 - DRIVABLE_ALPHA) + np.asarray(DRIVABLE_RGB) * DRIVABLE_ALPHA
    assert np.abs(out[y, x].astype(float) - expected).max() <= 1.0
    # Lane pixels are tinted cyan (high G and B, lower R).
    lane_px = out[140, 61].astype(int)
    assert lane_px[1] > 180 and lane_px[2] > 180 and lane_px[0] < lane_px[1]
    assert LANE_RGB == (0, 255, 255)
    # An untracked low-score box is not drawn: its neighbourhood (no masks there) is pixel-identical.
    # (rows 86..114 lie below the 4-line HUD band, which is 80 px high at this frame size, and above the masks)
    assert np.array_equal(out[86:114, 6:44], image[86:114, 6:44])
    # The tracked box outline is drawn.
    assert not np.array_equal(out[100, 130:190], image[100, 130:190])
    # The HUD band darkens the top rows.
    assert out[:5].mean() < image[:5].mean()


def test_frame_text_is_ascii() -> None:
    assert ascii_text("YOLOPv2 · learned_model · CPU") == "YOLOPv2 ? learned_model ? CPU"
    assert ascii_text("ko‘rsatkich").isascii()
