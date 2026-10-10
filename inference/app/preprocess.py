"""Image validation/decoding and YOLOPv2 letterbox preprocessing with invertible geometry.

Geometry policy (docs/IMPLEMENTATION_SPEC.md section 2): scale the longest side to 640, pad to the minimal
stride-32 multiple (upstream ``letterbox(auto=True)``), pad colour 114, BGR->RGB, float32 / 255.
The padding split and resize sizes are bit-for-bit the upstream ``letterbox`` computation (tested), but the
geometry is kept in integers so boxes and masks can be inverted with the padding that was actually applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

TARGET_SIZE = 640
STRIDE = 32
PAD_VALUE = 114
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
ALLOWED_CONTENT_TYPES: dict[str, Literal["jpeg", "png"]] = {"image/jpeg": "jpeg", "image/png": "png"}

# JPEG start-of-frame markers that carry the image dimensions (excludes DHT C4, JPG C8, DAC CC).
_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_STANDALONE_MARKERS = {0x01, 0xD8} | set(range(0xD0, 0xD8))


class ImageInvalid(ValueError):
    """Bytes are not a decodable JPEG/PNG of the declared type."""


class ImageTooLarge(ValueError):
    """Declared or decoded dimensions exceed the configured bound."""


@dataclass(frozen=True)
class LetterboxGeometry:
    image_width: int
    image_height: int
    canvas_width: int
    canvas_height: int
    scale: float  # letterbox ratio r (same for both axes, as upstream)
    content_width: int  # resized image size inside the canvas: round(W*r), round(H*r)
    content_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    def to_wire(self) -> dict[str, float | int]:
        return {
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "scale": self.scale,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
        }


def compute_geometry(width: int, height: int, target: int = TARGET_SIZE, stride: int = STRIDE) -> LetterboxGeometry:
    """Mirror upstream ``letterbox(img, 640, auto=True, scaleup=True, stride=32)`` sizes in integer form."""
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    r = min(target / height, target / width)
    content_w, content_h = int(round(width * r)), int(round(height * r))
    if content_w < 1 or content_h < 1:
        raise ValueError("image aspect ratio too extreme for the letterbox")
    pad_w = (target - content_w) % stride
    pad_h = (target - content_h) % stride
    # Upstream: left = round(dw/2 - 0.1), right = round(dw/2 + 0.1) -> for integer dw this is floor/ceil.
    pad_left, pad_top = pad_w // 2, pad_h // 2
    return LetterboxGeometry(
        image_width=width,
        image_height=height,
        canvas_width=content_w + pad_w,
        canvas_height=content_h + pad_h,
        scale=r,
        content_width=content_w,
        content_height=content_h,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_w - pad_left,
        pad_bottom=pad_h - pad_top,
    )


def letterbox(image_bgr: np.ndarray, geometry: LetterboxGeometry) -> np.ndarray:
    """Resize with INTER_LINEAR and pad with colour 114; returns a uint8 BGR canvas."""
    h, w = image_bgr.shape[:2]
    if (w, h) != (geometry.image_width, geometry.image_height):
        raise ValueError("geometry does not belong to this image")
    resized = image_bgr
    if (w, h) != (geometry.content_width, geometry.content_height):
        resized = cv2.resize(
            image_bgr, (geometry.content_width, geometry.content_height), interpolation=cv2.INTER_LINEAR
        )
    return cv2.copyMakeBorder(
        resized,
        geometry.pad_top,
        geometry.pad_bottom,
        geometry.pad_left,
        geometry.pad_right,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )


def canvas_to_input(canvas_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 -> contiguous float32 1x3xHxW RGB in [0, 1] (upstream demo normalisation)."""
    rgb_chw = canvas_bgr[:, :, ::-1].transpose(2, 0, 1)
    out = np.ascontiguousarray(rgb_chw, dtype=np.float32)
    out /= 255.0
    return out[None]


def _jpeg_size(data: bytes) -> tuple[int, int]:
    i = 2
    n = len(data)
    while i < n:
        if data[i] != 0xFF:
            raise ImageInvalid("corrupt JPEG marker stream")
        while i < n and data[i] == 0xFF:  # fill bytes
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1
        if marker in _STANDALONE_MARKERS:
            continue
        if marker == 0xD9:  # EOI before SOF
            break
        if i + 2 > n:
            break
        seg_len = int.from_bytes(data[i : i + 2], "big")
        if seg_len < 2:
            raise ImageInvalid("corrupt JPEG segment length")
        if marker in _SOF_MARKERS:
            if i + 7 > n:
                break
            height = int.from_bytes(data[i + 3 : i + 5], "big")
            width = int.from_bytes(data[i + 5 : i + 7], "big")
            return width, height
        i += seg_len
    raise ImageInvalid("JPEG has no frame header")


def _png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[12:16] != b"IHDR":
        raise ImageInvalid("PNG has no IHDR header")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def sniff_image(data: bytes, content_type: str | None) -> tuple[Literal["jpeg", "png"], int, int]:
    """Validate declared type against magic bytes and read dimensions from the header (no pixel decode)."""
    declared = (content_type or "").split(";", 1)[0].strip().lower()
    kind = ALLOWED_CONTENT_TYPES.get(declared)
    if kind is None:
        raise ImageInvalid("Content-Type must be image/jpeg or image/png")
    if not data:
        raise ImageInvalid("empty request body")
    if kind == "jpeg":
        if not data.startswith(JPEG_MAGIC):
            raise ImageInvalid("body is not a JPEG")
        width, height = _jpeg_size(data)
    else:
        if not data.startswith(PNG_MAGIC):
            raise ImageInvalid("body is not a PNG")
        width, height = _png_size(data)
    if width < 1 or height < 1:
        raise ImageInvalid("image header has zero dimensions")
    return kind, width, height


def decode_image(data: bytes, content_type: str | None, max_side: int, min_side: int) -> np.ndarray:
    """Return a BGR uint8 HxWx3 image or raise ImageInvalid / ImageTooLarge.

    Dimensions are checked from the header before decoding (decompression-bomb guard) and again afterwards.
    EXIF orientation is ignored on purpose: callers send frames already in display orientation, and the
    reported image size must equal the pixel grid that was analysed.
    """
    _, width, height = sniff_image(data, content_type)
    if width > max_side or height > max_side:
        raise ImageTooLarge(f"image {width}x{height} exceeds {max_side}x{max_side}")
    if width < min_side or height < min_side:
        raise ImageInvalid(f"image {width}x{height} is smaller than {min_side}x{min_side}")
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    # IMREAD_COLOR (without IMREAD_ANYDEPTH) always yields 8-bit, 3-channel BGR, also for 16-bit/alpha PNGs.
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ImageInvalid("image bytes could not be decoded")
    if image.shape[0] != height or image.shape[1] != width:
        raise ImageInvalid("decoded size differs from header size")
    return image
