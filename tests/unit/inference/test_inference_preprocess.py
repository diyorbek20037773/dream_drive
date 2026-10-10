"""Letterbox geometry and image validation (no weights needed)."""

from __future__ import annotations

import struct
import zlib

import cv2
import numpy as np
import pytest

from app.preprocess import (
    ImageInvalid,
    ImageTooLarge,
    canvas_to_input,
    compute_geometry,
    decode_image,
    letterbox,
    sniff_image,
)
from app.vendored import yolopv2_utils as upstream

SIZES = [
    (1280, 720),
    (1920, 1080),
    (720, 1280),
    (640, 480),
    (1280, 730),
    (1000, 563),
    (300, 200),
    (4096, 64),
    (17, 4000),
    (641, 641),
]


def test_geometry_for_1280x720_matches_spec_example() -> None:
    g = compute_geometry(1280, 720)
    assert (g.canvas_width, g.canvas_height) == (640, 384)
    assert g.scale == 0.5
    assert (g.pad_left, g.pad_top, g.pad_right, g.pad_bottom) == (0, 12, 0, 12)
    assert g.to_wire() == {
        "canvas_width": 640,
        "canvas_height": 384,
        "scale": 0.5,
        "pad_left": 0,
        "pad_top": 12,
        "pad_right": 0,
        "pad_bottom": 12,
    }


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_letterbox_is_pixel_identical_to_upstream(width: int, height: int) -> None:
    rng = np.random.default_rng(width * 7919 + height)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    g = compute_geometry(width, height)
    ours = letterbox(image, g)
    theirs, ratio, (dw, dh) = upstream.letterbox(image, 640, stride=32)
    assert ours.shape == theirs.shape == (g.canvas_height, g.canvas_width, 3)
    assert np.array_equal(ours, theirs)
    assert ratio == (g.scale, g.scale)
    # Integer pads equal upstream's round(d - 0.1) / round(d + 0.1) split.
    assert g.pad_left == int(round(dw - 0.1)) and g.pad_right == int(round(dw + 0.1))
    assert g.pad_top == int(round(dh - 0.1)) and g.pad_bottom == int(round(dh + 0.1))
    assert g.canvas_width % 32 == 0 and g.canvas_height % 32 == 0
    assert max(g.content_width, g.content_height) == 640


def test_odd_padding_is_split_floor_ceil() -> None:
    g = compute_geometry(1280, 730)  # content 640x365, 19 px padding -> 9 top / 10 bottom
    assert (g.content_width, g.content_height) == (640, 365)
    assert (g.pad_top, g.pad_bottom) == (9, 10)
    assert g.canvas_height == 384


def test_pad_colour_and_tensor_layout() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[..., 0] = 10  # B
    image[..., 1] = 20  # G
    image[..., 2] = 250  # R
    g = compute_geometry(1280, 720)
    canvas = letterbox(image, g)
    assert (canvas[: g.pad_top] == 114).all() and (canvas[-g.pad_bottom :] == 114).all()
    x = canvas_to_input(canvas)
    assert x.shape == (1, 3, 384, 640) and x.dtype == np.float32 and x.flags.c_contiguous
    centre = x[0, :, 200, 320]
    assert np.allclose(centre, [250 / 255, 20 / 255, 10 / 255])  # RGB order after BGR->RGB
    assert np.allclose(x[0, :, 0, 0], 114 / 255)


def _png_bytes(width: int, height: int) -> bytes:
    """A real (tiny-file) PNG whose header declares width x height; compresses to a few KB."""
    raw = b"".join(b"\x00" + b"\x00" * (width * 3) for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def test_decode_valid_jpeg_and_png_roundtrip_sizes() -> None:
    img = np.full((48, 64, 3), 77, dtype=np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    assert ok
    assert sniff_image(jpg.tobytes(), "image/jpeg") == ("jpeg", 64, 48)
    out = decode_image(jpg.tobytes(), "image/jpeg; charset=binary", 4096, 16)
    assert out.shape == (48, 64, 3) and out.dtype == np.uint8
    ok, png = cv2.imencode(".png", img)
    assert sniff_image(png.tobytes(), "image/png") == ("png", 64, 48)
    assert decode_image(png.tobytes(), "image/png", 4096, 16).shape == (48, 64, 3)


def test_progressive_jpeg_header_is_parsed() -> None:
    img = np.random.default_rng(0).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
    assert ok
    assert sniff_image(jpg.tobytes(), "image/jpeg")[1:] == (50, 40)


@pytest.mark.parametrize(
    ("data", "ctype"),
    [
        (b"", "image/jpeg"),
        (b"not an image at all", "image/jpeg"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 64, "image/jpeg"),  # JPEG magic, no frame header
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image/png"),  # PNG magic, truncated IHDR
        (b"GIF89a....", "image/gif"),
    ],
)
def test_invalid_bytes_are_rejected(data: bytes, ctype: str) -> None:
    with pytest.raises(ImageInvalid):
        decode_image(data, ctype, 4096, 16)


def test_content_type_must_match_magic() -> None:
    ok, png = cv2.imencode(".png", np.zeros((32, 32, 3), np.uint8))
    with pytest.raises(ImageInvalid):
        decode_image(png.tobytes(), "image/jpeg", 4096, 16)
    with pytest.raises(ImageInvalid):
        decode_image(png.tobytes(), None, 4096, 16)


def test_truncated_jpeg_payload_is_invalid() -> None:
    img = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    data = jpg.tobytes()
    # Header intact (dimensions readable) but the entropy-coded data is cut off before decoding can succeed.
    cut = data[: data.index(b"\xff\xda") + 4]
    with pytest.raises(ImageInvalid):
        decode_image(cut, "image/jpeg", 4096, 16)


def test_oversized_dimensions_rejected_before_decode() -> None:
    data = _png_bytes(4097, 16)
    with pytest.raises(ImageTooLarge):
        decode_image(data, "image/png", 4096, 16)
    assert decode_image(_png_bytes(4096, 16), "image/png", 4096, 16).shape == (16, 4096, 3)


def test_too_small_image_rejected() -> None:
    ok, png = cv2.imencode(".png", np.zeros((8, 64, 3), np.uint8))
    with pytest.raises(ImageInvalid):
        decode_image(png.tobytes(), "image/png", 4096, 16)
