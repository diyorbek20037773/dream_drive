"""Inverse letterbox transforms for masks and boxes (no weights needed).

(a) mask inversion of a 1280x720 frame equals upstream driving_area_mask / lane_line_mask on the same tensors;
(b) boxes on padded canvases map back to the exact original pixels for landscape, portrait and 4:3 inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.postprocess import (
    class_name_for,
    decode_detections,
    decode_mask_png,
    encode_mask_png,
    invert_boxes,
    invert_drivable,
    invert_lane,
    to_detections,
)
from app.preprocess import compute_geometry, letterbox
from app.vendored import yolopv2_utils as upstream


# ----------------------------------------------------------------------------------------------- (a) masks
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_mask_inversion_equals_upstream_for_1280x720(seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    g = compute_geometry(1280, 720)
    seg = torch.rand((1, 2, g.canvas_height, g.canvas_width), generator=gen)
    ll = torch.rand((1, 1, g.canvas_height, g.canvas_width), generator=gen)

    ours_da = invert_drivable(seg, g)
    ours_ll = invert_lane(ll, g)
    up_da = upstream.driving_area_mask(seg.clone())
    up_ll = upstream.lane_line_mask(ll.clone())

    assert ours_da.shape == up_da.shape == (720, 1280)
    assert ours_ll.shape == up_ll.shape == (720, 1280)
    assert np.array_equal(ours_da, up_da == 1)
    assert np.array_equal(ours_ll, up_ll == 1)
    # Non-trivial masks: the comparison is not vacuous.
    assert 0.3 < ours_da.mean() < 0.7 and 0.2 < ours_ll.mean() < 0.8


def test_mask_inversion_equals_upstream_on_structured_probabilities() -> None:
    """Smooth, saturated probability fields (like real outputs) including exact 0.5 ties."""
    g = compute_geometry(1280, 720)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, g.canvas_height), torch.linspace(0, 1, g.canvas_width), indexing="ij")
    p1 = torch.sigmoid((yy - 0.6) * 30 + torch.sin(xx * 12))
    seg = torch.stack([1 - p1, p1])[None]
    ll = torch.clamp(torch.cos(xx * 40) * 0.6 + 0.5, 0, 1)[None, None]
    ll[..., 100:110, :] = 0.5  # exact ties: upstream rounds half to even -> not lane
    assert np.array_equal(invert_drivable(seg, g), upstream.driving_area_mask(seg) == 1)
    assert np.array_equal(invert_lane(ll, g), upstream.lane_line_mask(ll) == 1)


@pytest.mark.parametrize(("width", "height"), [(720, 1280), (640, 480), (1920, 1080), (1280, 730)])
def test_mask_rectangle_maps_back_to_original_location(width: int, height: int) -> None:
    """A drivable rectangle placed in canvas content coordinates lands at the scaled original location,
    and nothing from the padding leaks into the original-size mask."""
    g = compute_geometry(width, height)
    seg = torch.zeros((1, 2, g.canvas_height, g.canvas_width))
    seg[:, 0] = 1.0
    # Padding rows/cols claim "drivable" to prove they are cropped away.
    seg[:, 1, : g.pad_top, :] = 2.0
    seg[:, 1, g.canvas_height - g.pad_bottom :, :] = 2.0
    seg[:, 1, :, : g.pad_left] = 2.0
    seg[:, 1, :, g.canvas_width - g.pad_right :] = 2.0
    cx1, cy1, cx2, cy2 = 40, 64, 200, 160  # content-relative canvas rectangle
    seg[:, 1, g.pad_top + cy1 : g.pad_top + cy2, g.pad_left + cx1 : g.pad_left + cx2] = 2.0
    mask = invert_drivable(seg, g)
    assert mask.shape == (height, width)
    ys, xs = np.nonzero(mask)
    sx, sy = width / g.content_width, height / g.content_height
    tol_x, tol_y = sx + 1, sy + 1  # bilinear edge falls within one canvas pixel
    assert abs(xs.min() - cx1 * sx) <= tol_x and abs(xs.max() + 1 - cx2 * sx) <= tol_x
    assert abs(ys.min() - cy1 * sy) <= tol_y and abs(ys.max() + 1 - cy2 * sy) <= tol_y


def test_mask_png_roundtrip_is_uint8_0_255() -> None:
    mask = np.zeros((37, 53), dtype=bool)
    mask[5:9, 7:30] = True
    png = decode_mask_png(encode_mask_png(mask))
    assert png.dtype == np.uint8 and png.shape == (37, 53)
    assert set(np.unique(png).tolist()) == {0, 255}
    assert np.array_equal(png == 255, mask)


# ----------------------------------------------------------------------------------------------- (b) boxes
def _canvas_bbox_of_white(canvas: np.ndarray) -> tuple[int, int, int, int]:
    """Edge-coordinate xyxy of pixels brighter than mid-grey (excludes the 114 padding)."""
    ys, xs = np.nonzero(canvas.min(axis=2) > 180)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@pytest.mark.parametrize(
    ("width", "height", "rect"),
    [
        # landscape 1920x1080: r = 1/3, 12 px top/bottom padding; coordinates multiples of 3
        (1920, 1080, (300, 210, 600, 750)),
        # portrait 720x1280: r = 1/2, 12 px left/right padding; even coordinates
        (720, 1280, (100, 300, 500, 1100)),
        # 4:3 640x480: r = 1, canvas 640x480, no padding
        (640, 480, (13, 27, 401, 333)),
        # 1280x730: odd padding split (9 top / 10 bottom)
        (1280, 730, (200, 100, 900, 600)),
    ],
)
def test_box_drawn_on_original_maps_back_exactly(width: int, height: int, rect: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = rect
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[y1:y2, x1:x2] = 255
    g = compute_geometry(width, height)
    canvas = letterbox(image, g)  # the exact canvas the model sees
    found = _canvas_bbox_of_white(canvas)
    # Sanity: the rectangle sits inside the content area, offset by the applied padding.
    assert found[0] >= g.pad_left and found[1] >= g.pad_top
    back = invert_boxes(np.array([found], dtype=np.float64), g)[0]
    assert back.tolist() == pytest.approx([x1, y1, x2, y2], abs=1e-9)


@pytest.mark.parametrize(("width", "height"), [(1920, 1080), (720, 1280), (640, 480), (1000, 563), (333, 1999)])
def test_box_inverse_is_exact_algebraically(width: int, height: int) -> None:
    rng = np.random.default_rng(width + height)
    g = compute_geometry(width, height)
    xy1 = rng.uniform(0, [width * 0.5, height * 0.5], size=(50, 2))
    xy2 = xy1 + rng.uniform(1, [width * 0.5, height * 0.5], size=(50, 2))
    orig = np.concatenate([xy1, xy2], axis=1)
    sx, sy = g.content_width / width, g.content_height / height
    canvas = orig.copy()
    canvas[:, [0, 2]] = orig[:, [0, 2]] * sx + g.pad_left
    canvas[:, [1, 3]] = orig[:, [1, 3]] * sy + g.pad_top
    assert np.allclose(invert_boxes(canvas, g), orig, atol=1e-6)


def test_box_inverse_matches_upstream_scale_coords_for_1280x720() -> None:
    g = compute_geometry(1280, 720)
    boxes = torch.tensor([[0.0, 12.0, 640.0, 372.0], [100.5, 50.25, 300.0, 200.0], [-5.0, 0.0, 700.0, 390.0]])
    up = upstream.scale_coords((g.canvas_height, g.canvas_width), boxes.clone(), (720, 1280)).numpy()
    assert np.allclose(invert_boxes(boxes.numpy(), g), up, atol=1e-4)


def test_boxes_are_clipped_to_image_and_padding_only_boxes_dropped() -> None:
    g = compute_geometry(1280, 720)
    det = torch.tensor(
        [
            [-10.0, 0.0, 50.0, 30.0, 0.9, 3.0],  # partly in top padding and left of the image
            [10.0, 0.0, 60.0, 11.0, 0.8, 3.0],  # entirely inside the 12 px top padding
            [600.0, 350.0, 700.0, 400.0, 0.7, 0.0],  # extends past right/bottom; class 0 not verified
        ]
    )
    dets = to_detections(det, g, {3: "vehicle"})
    assert len(dets) == 2
    assert dets[0].xyxy == pytest.approx((0.0, 0.0, 100.0, 36.0))
    assert dets[0].class_name == "vehicle"
    assert dets[1].xyxy == pytest.approx((1200.0, 676.0, 1280.0, 720.0))
    assert dets[1].class_name == "unverified_class_0"
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        assert 0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720


def test_class_names_never_guessed() -> None:
    assert class_name_for(3, {3: "vehicle"}) == "vehicle"
    assert class_name_for(0, {3: "vehicle"}) == "unverified_class_0"
    assert class_name_for(79, {3: "vehicle"}) == "unverified_class_79"


# ----------------------------------------------------------------------------------------------- head decoding
def test_decode_heads_recovers_known_anchor_box() -> None:
    """One strong anchor in the stride-8 head decodes to the expected canvas box via the vendored code."""
    h, w = 384, 640
    pred = [torch.full((1, 255, h // s, w // s), -20.0) for s in (8, 16, 32)]
    anchors = [torch.tensor([[12.0, 16.0], [5.0, 5.0], [5.0, 5.0]]).view(1, 3, 1, 1, 2) for _ in range(3)]
    gy, gx = 7, 20
    a = 0
    pred[0][0, a * 85 + 0 : a * 85 + 4, gy, gx] = 0.0
    pred[0][0, a * 85 + 4, gy, gx] = 10.0
    pred[0][0, a * 85 + 5 + 3, gy, gx] = 10.0
    det = decode_detections(pred, anchors, conf_thres=0.1, iou_thres=0.45)
    assert det.shape == (1, 6)
    cx, cy = (gx + 0.5) * 8, (gy + 0.5) * 8
    assert det[0, :4].tolist() == pytest.approx([cx - 6, cy - 8, cx + 6, cy + 8], abs=1e-4)
    assert det[0, 4].item() == pytest.approx(torch.sigmoid(torch.tensor(10.0)).item() ** 2, rel=1e-5)
    assert det[0, 5].item() == 3.0
    # Input list is not mutated by the vendored decode (modification 2).
    assert pred[0].shape == (1, 255, h // 8, w // 8)


def test_decode_heads_threshold_filters_everything() -> None:
    pred = [torch.full((1, 255, 384 // s, 640 // s), -20.0) for s in (8, 16, 32)]
    anchors = [torch.ones((1, 3, 1, 1, 2)) for _ in range(3)]
    assert decode_detections(pred, anchors, 0.1, 0.45).shape == (0, 6)
