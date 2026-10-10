"""Synthetic MP4 generators used by the video tests, plus checks that the generated fixtures really have the timing
and metadata the other tests rely on (otherwise a VFR or rotation test could pass vacuously).

Clips are written with PyAV into pytest tmp dirs; nothing is committed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

TIME_BASE = Fraction(1, 90000)
RED = (255, 0, 0)


def textured_background(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Road-like test image: textured grey (not 'blurry'), darker lower half, mid luma (not 'dark')."""
    rng = np.random.default_rng(seed)
    img = rng.integers(70, 190, size=(height, width, 3), dtype=np.uint8)
    img[height // 2 :] = (img[height // 2 :] * 0.8).astype(np.uint8)
    return img


def default_draw(index: int, t: float, img: np.ndarray) -> None:
    """White marker in the top-left corner (orientation probe) and a red 'vehicle' box approaching in the centre."""
    h, w = img.shape[:2]
    img[: max(2, h // 10), : max(2, w // 10)] = 255
    bw = int(w * (0.12 + 0.02 * min(index, 10)))
    bh = int(bw * 0.6)
    cx = w // 2
    y2 = int(h * 0.72) + min(index, 10) * max(1, h // 100)
    img[max(0, y2 - bh) : min(h, y2), max(0, cx - bw // 2) : min(w, cx + bw // 2)] = RED


def make_clip(
    path: Path,
    *,
    times_s: Sequence[float] | None = None,
    n_frames: int = 30,
    fps: float = 30.0,
    size: tuple[int, int] = (320, 180),
    rotation_ccw: int = 0,
    pix_fmt: str = "yuv420p",
    draw: Callable[[int, float, np.ndarray], None] | None = default_draw,
    with_audio: bool = False,
) -> list[float]:
    """Write an H.264 MP4 whose frames have exactly the given presentation times. Returns the times."""
    times = list(times_s) if times_s is not None else [i / fps for i in range(n_frames)]
    width, height = size
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        stream.width, stream.height, stream.pix_fmt = width, height, pix_fmt
        stream.codec_context.time_base = TIME_BASE
        stream.time_base = TIME_BASE
        stream.codec_context.options = {"preset": "ultrafast", "crf": "18", "threads": "1"}
        if rotation_ccw:
            stream.set_display_rotation(rotation_ccw)
        audio = None
        if with_audio:
            audio = container.add_stream("aac", rate=16000)
            audio.layout = "mono"
        for i, t in enumerate(times):
            img = textured_background(width, height, seed=i)
            if draw is not None:
                draw(i, t, img)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = round(t / TIME_BASE)
            frame.time_base = TIME_BASE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        if audio is not None:
            samples = np.zeros((1, 1024), dtype=np.float32)
            for k in range(max(1, int(times[-1] * 16000 / 1024))):
                aframe = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
                aframe.sample_rate = 16000
                aframe.pts = k * 1024
                for packet in audio.encode(aframe):
                    container.mux(packet)
            for packet in audio.encode(None):
                container.mux(packet)
    return times


def decoded_times(path: Path) -> list[float]:
    with av.open(str(path)) as container:
        return [float(f.time) for f in container.decode(video=0)]


# ---------------------------------------------------------------------------------------------------------------
# Fixture self-checks
# ---------------------------------------------------------------------------------------------------------------
def test_cfr_clip_has_exact_frame_times(tmp_path: Path) -> None:
    path = tmp_path / "cfr.mp4"
    expected = make_clip(path, n_frames=12, fps=30.0)
    got = decoded_times(path)
    assert len(got) == 12
    assert np.allclose(got, expected, atol=1e-4)


def test_vfr_clip_keeps_irregular_gaps(tmp_path: Path) -> None:
    path = tmp_path / "vfr.mp4"
    expected = [0.0, 0.033, 0.066, 0.4, 0.433, 1.2, 1.233]
    make_clip(path, times_s=expected)
    got = decoded_times(path)
    assert np.allclose(got, expected, atol=1e-4)
    gaps = np.diff(got)
    assert gaps.max() > 10 * gaps.min(), "fixture must be genuinely variable-frame-rate"


def test_rotated_clip_carries_display_matrix(tmp_path: Path) -> None:
    path = tmp_path / "rot.mp4"
    make_clip(path, n_frames=3, rotation_ccw=90)
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
        assert frame.rotation == 90
        assert (frame.width, frame.height) == (320, 180)  # coded size is unrotated
