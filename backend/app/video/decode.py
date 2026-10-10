"""PyAV decoding with real presentation timestamps and PTS-grid sampling (spec section 5, step 2).

Sampling rule: for grid times g_k = k / analysis_fps (seconds since the stream start), the sample for g_k is the
first decoded frame whose PTS >= g_k. After taking a frame at time t the grid advances to the first g_k > t, so a VFR
gap (or a source slower than analysis_fps) yields fewer samples instead of synthesised ones, and sampled timestamps
always come from the stream - never from `frame_index / fps`.

Timestamps: `frame.time` (pts * time_base). The reported `pts_s` is relative to the stream start (start_time, or the
first decoded frame when the container has none) so it starts near 0 like a player's clock. Frames whose PTS is
missing get `previous + 1/fps_avg` and are flagged `pts_estimated`; non-increasing PTS frames are dropped and counted.

Rotation: the clockwise display rotation from the probe is applied with np.rot90 (k = -rotation/90).

Watchdog: cumulative wall time spent inside demux/decode/convert is bounded (`max_decode_wall_s`), as are the number
of decoded frames and the decoded size; violations raise VideoError. Time spent by the consumer between frames
(inference) is not charged to the decode watchdog.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from app.video import errors
from app.video.errors import VideoError
from app.video.probe import VideoProbe

RgbImage = NDArray[np.uint8]


@dataclass(frozen=True)
class DecodeLimits:
    max_decode_wall_s: float = 300.0
    max_decoded_frames: int = 120 * 240  # 120 s at 240 fps
    max_width: int = 1920
    max_height: int = 1080


@dataclass(frozen=True)
class SampledFrame:
    sample_index: int  # 0-based index among analyzed samples
    source_frame_index: int  # 0-based index among decoded frames
    pts_s: float  # seconds since the stream start, from the stream's PTS
    pts_estimated: bool
    image: RgbImage  # H x W x 3 RGB, rotation applied, C-contiguous


@dataclass
class DecodeStats:
    decoded_frames: int = 0
    sampled_frames: int = 0
    estimated_pts_frames: int = 0
    dropped_non_monotonic: int = 0
    resized_frames: int = 0
    stopped_at_decode_error: bool = False  # decoding ended early at a corrupt packet after valid frames
    decode_wall_s: float = 0.0
    start_time_s: float | None = None
    last_pts_s: float | None = None
    output_size: tuple[int, int] | None = None  # (width, height) after rotation
    sample_pts: list[float] = field(default_factory=list)


def rotate_display(image: RgbImage, rotation_cw: int) -> RgbImage:
    """Apply a clockwise display rotation (0/90/180/270)."""
    k = (-(rotation_cw // 90)) % 4
    if k == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, k=k))


def _stream_start_s(stream: av.video.stream.VideoStream) -> float | None:
    if stream.start_time is None or stream.time_base is None:
        return None
    return float(stream.start_time * stream.time_base)


def iter_sampled_frames(
    path: Path,
    probe: VideoProbe,
    analysis_fps: float,
    limits: DecodeLimits,
    stats: DecodeStats,
) -> Iterator[SampledFrame]:
    """Yield PTS-grid samples. `stats` is filled in place (valid also after an exception or early close)."""
    if analysis_fps <= 0:
        raise ValueError("analysis_fps must be positive")
    try:
        container = av.open(str(path), mode="r")
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise VideoError(errors.VIDEO_INVALID, "file is not a readable video container") from exc
    with container:
        try:
            stream = _stream_by_index(container, probe)
        except IndexError as exc:
            raise VideoError(errors.VIDEO_NO_STREAM, "no video stream found") from exc
        stream.thread_type = "AUTO"
        start_s = _stream_start_s(stream)
        fallback_step = 1.0 / probe.fps_avg if probe.fps_avg else 1.0 / 30.0
        next_k = 0
        prev_t: float | None = None
        frame_iter = container.decode(stream)
        while True:
            t0 = time.perf_counter()
            try:
                frame = next(frame_iter)
            except StopIteration:
                stats.decode_wall_s += time.perf_counter() - t0
                break
            except (av.FFmpegError, ValueError) as exc:
                stats.decode_wall_s += time.perf_counter() - t0
                if stats.decoded_frames == 0:
                    raise VideoError(errors.VIDEO_DECODE_FAILED, "video could not be decoded") from exc
                # A corrupt tail after valid frames ends decoding; the pipeline reports this as a limitation.
                stats.stopped_at_decode_error = True
                break
            stats.decoded_frames += 1
            if stats.decoded_frames > limits.max_decoded_frames:
                raise VideoError(errors.VIDEO_TOO_MANY_FRAMES, f"more than {limits.max_decoded_frames} frames")

            raw_t = frame.time
            estimated = raw_t is None
            if raw_t is None:
                raw_t = (prev_t + (start_s or 0.0) + fallback_step) if prev_t is not None else (start_s or 0.0)
                stats.estimated_pts_frames += 1
            if start_s is None:
                start_s = float(raw_t)
            stats.start_time_s = start_s
            t = max(0.0, float(raw_t) - start_s)
            if prev_t is not None and t <= prev_t:
                stats.dropped_non_monotonic += 1
                stats.decode_wall_s += time.perf_counter() - t0
                _check_watchdog(stats, limits)
                continue
            prev_t = t
            stats.last_pts_s = t

            take = t + 1e-9 >= next_k / analysis_fps
            image: RgbImage | None = None
            if take:
                image = _to_rgb(frame, probe.rotation, limits, stats)
                # Skip every grid point covered by this frame (VFR gaps produce no synthetic samples).
                next_k = int(np.floor(t * analysis_fps + 1e-9)) + 1
            stats.decode_wall_s += time.perf_counter() - t0
            _check_watchdog(stats, limits)
            if image is not None:
                sample = SampledFrame(
                    sample_index=stats.sampled_frames,
                    source_frame_index=stats.decoded_frames - 1,
                    pts_s=t,
                    pts_estimated=estimated,
                    image=image,
                )
                stats.sampled_frames += 1
                stats.sample_pts.append(t)
                yield sample


def _stream_by_index(container: av.container.InputContainer, probe: VideoProbe) -> av.video.stream.VideoStream:
    for s in container.streams.video:
        if s.index == probe.stream_index:
            return s
    return container.streams.video[0]


def _check_watchdog(stats: DecodeStats, limits: DecodeLimits) -> None:
    if stats.decode_wall_s > limits.max_decode_wall_s:
        raise VideoError(errors.VIDEO_DECODE_TIMEOUT, f"decoding exceeded {limits.max_decode_wall_s:g} s")


def _to_rgb(frame: av.VideoFrame, rotation_cw: int, limits: DecodeLimits, stats: DecodeStats) -> RgbImage:
    image = rotate_display(np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8), rotation_cw)
    height, width = image.shape[:2]
    if width > limits.max_width or height > limits.max_height:
        raise VideoError(errors.VIDEO_TOO_LARGE_DIMENSIONS, f"decoded frame {width}x{height} exceeds limits")
    if stats.output_size is None:
        stats.output_size = (width, height)
    elif stats.output_size != (width, height):
        # Mid-stream resolution change: keep one analysis geometry (annotation/encoding need a fixed size).
        image = np.asarray(cv2.resize(image, stats.output_size, interpolation=cv2.INTER_AREA), dtype=np.uint8)
        stats.resized_frames += 1
    return np.ascontiguousarray(image, dtype=np.uint8)
