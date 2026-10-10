"""Browser-playable H.264 output via PyAV's bundled libx264 (spec section 5, step 4).

Output: MP4, H.264 (libx264, yuv420p, `profile=high`), constant frame rate = the analysis rate, `+faststart` so
the moov atom precedes media data (progressive playback over HTTP range requests), no audio stream. Odd frame
sizes are padded by one replicated row/column (yuv420p needs even dimensions); pixel coordinates are unchanged.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import TracebackType

import av
import cv2
import numpy as np
from numpy.typing import NDArray

RgbImage = NDArray[np.uint8]


def even_size(width: int, height: int) -> tuple[int, int]:
    return width + (width % 2), height + (height % 2)


def pad_to_even(image: RgbImage) -> RgbImage:
    height, width = image.shape[:2]
    pad_x, pad_y = width % 2, height % 2
    if not pad_x and not pad_y:
        return image
    return np.asarray(cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE), dtype=np.uint8)


class H264Writer:
    """Write RGB frames as a CFR H.264 MP4. Use as a context manager; `close()` flushes the encoder."""

    def __init__(
        self, path: Path, width: int, height: int, fps: float, crf: int = 23, preset: str = "veryfast"
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.path = Path(path)
        self.width, self.height = even_size(width, height)
        self.rate = Fraction(fps).limit_denominator(1001)
        self.frames_written = 0
        self._container = av.open(str(self.path), mode="w", format="mp4", options={"movflags": "+faststart"})
        stream = self._container.add_stream("libx264", rate=self.rate)
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1) / self.rate
        stream.codec_context.time_base = Fraction(1, 1) / self.rate
        # Single-threaded x264 keeps the output byte-for-byte reproducible for identical input frames.
        stream.codec_context.options = {"crf": str(crf), "preset": preset, "profile": "high", "threads": "1"}
        self._stream = stream
        self._closed = False

    def write(self, image: RgbImage) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        frame_rgb = pad_to_even(np.ascontiguousarray(image, dtype=np.uint8))
        if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
            frame_rgb = np.asarray(
                cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_AREA), dtype=np.uint8
            )
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        frame.pts = self.frames_written
        # The codec time base is fixed; the mp4 muxer rewrites the *stream* time base (e.g. 1/10240) once the header
        # is written, so frame timestamps must never be expressed in the stream time base.
        frame.time_base = self._stream.codec_context.time_base
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for packet in self._stream.encode(None):
                self._container.mux(packet)
        finally:
            self._container.close()

    def abort(self) -> None:
        """Close without guaranteeing a valid file (used on cancellation/error before the file is discarded)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._container.close()
        except (av.FFmpegError, OSError, ValueError):
            pass

    def __enter__(self) -> H264Writer:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write_jpeg(path: Path, image_rgb: RgbImage, quality: int = 90) -> None:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    Path(path).write_bytes(buf.tobytes())


def encode_jpeg_bytes(image_rgb: RgbImage, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()
