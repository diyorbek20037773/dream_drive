"""PyAV decode: real PTS, PTS-grid sampling (never index/fps), rotation, limits and watchdog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_video_synth import make_clip

from app.video import errors
from app.video.decode import DecodeLimits, DecodeStats, iter_sampled_frames, rotate_display
from app.video.errors import VideoError
from app.video.probe import probe_with_pyav


def _sample(path: Path, fps: float, limits: DecodeLimits | None = None) -> tuple[list, DecodeStats]:
    stats = DecodeStats()
    samples = list(iter_sampled_frames(path, probe_with_pyav(path), fps, limits or DecodeLimits(), stats))
    return samples, stats


def test_cfr_sampling_follows_the_pts_grid(tmp_path: Path) -> None:
    path = tmp_path / "cfr.mp4"
    make_clip(path, n_frames=60, fps=30.0)  # 2.0 s
    samples, stats = _sample(path, 5.0)
    assert stats.decoded_frames == 60
    assert [s.sample_index for s in samples] == list(range(10))
    assert [s.pts_s for s in samples] == pytest.approx([k / 5.0 for k in range(10)], abs=1e-4)
    assert [s.source_frame_index for s in samples] == [6 * k for k in range(10)]
    assert not any(s.pts_estimated for s in samples)


def test_vfr_sampling_preserves_gaps_instead_of_index_over_fps(tmp_path: Path) -> None:
    path = tmp_path / "vfr.mp4"
    make_clip(path, times_s=[0.0, 0.033, 0.066, 0.4, 0.433, 1.2, 1.233])
    samples, stats = _sample(path, 5.0)
    # Grid 0, 0.2, 0.4, ...: 0.0 -> k=0; first frame >= 0.2 is 0.4; first frame >= 0.6 is 1.2 (gap preserved).
    assert [s.pts_s for s in samples] == pytest.approx([0.0, 0.4, 1.2], abs=1e-4)
    assert [s.source_frame_index for s in samples] == [0, 3, 5]
    # index/fps sampling would have reported 0.0, 0.2, 0.4 for the same frames: make sure we did not.
    assert stats.sample_pts == pytest.approx([0.0, 0.4, 1.2], abs=1e-4)


def test_source_slower_than_analysis_rate_is_not_upsampled(tmp_path: Path) -> None:
    path = tmp_path / "slow.mp4"
    make_clip(path, n_frames=6, fps=2.0)  # frames at 0, 0.5, ..., 2.5
    samples, _ = _sample(path, 5.0)
    assert [s.pts_s for s in samples] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], abs=1e-4)


def test_rotation_is_applied(tmp_path: Path) -> None:
    path = tmp_path / "rot.mp4"
    make_clip(path, n_frames=3, size=(320, 180), rotation_ccw=90)
    samples, stats = _sample(path, 5.0)
    image = samples[0].image
    assert image.shape == (320, 180, 3)
    assert stats.output_size == (180, 320)
    # The white marker drawn in the coded top-left corner ends up bottom-left after a 90 deg CCW display rotation.
    h, w = image.shape[:2]
    assert image[h - 5, 5].mean() > 230
    assert image[5, 5].mean() < 230


def test_rotate_display_directions() -> None:
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[0, 0] = 255  # top-left
    assert rotate_display(img, 0) is img
    cw90 = rotate_display(img, 90)
    assert cw90.shape == (3, 2, 3) and cw90[0, 1, 0] == 255  # top-left -> top-right when rotated clockwise
    ccw = rotate_display(img, 270)
    assert ccw.shape == (3, 2, 3) and ccw[2, 0, 0] == 255  # top-left -> bottom-left when rotated counter-clockwise
    r180 = rotate_display(img, 180)
    assert r180[1, 2, 0] == 255


def test_odd_dimensions_decode(tmp_path: Path) -> None:
    path = tmp_path / "odd.mp4"
    make_clip(path, n_frames=10, size=(101, 77), pix_fmt="yuv444p")
    samples, _ = _sample(path, 5.0)
    assert samples[0].image.shape == (77, 101, 3)
    assert samples[0].image.dtype == np.uint8 and samples[0].image.flags["C_CONTIGUOUS"]


def test_frame_limit(tmp_path: Path) -> None:
    path = tmp_path / "many.mp4"
    make_clip(path, n_frames=20, fps=30.0)
    with pytest.raises(VideoError) as exc:
        _sample(path, 5.0, DecodeLimits(max_decoded_frames=10))
    assert exc.value.code == errors.VIDEO_TOO_MANY_FRAMES


def test_decode_watchdog(tmp_path: Path) -> None:
    path = tmp_path / "a.mp4"
    make_clip(path, n_frames=10)
    with pytest.raises(VideoError) as exc:
        _sample(path, 5.0, DecodeLimits(max_decode_wall_s=0.0))
    assert exc.value.code == errors.VIDEO_DECODE_TIMEOUT


def test_decoded_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "a.mp4"
    make_clip(path, n_frames=3, size=(320, 180))
    with pytest.raises(VideoError) as exc:
        _sample(path, 5.0, DecodeLimits(max_width=200, max_height=200))
    assert exc.value.code == errors.VIDEO_TOO_LARGE_DIMENSIONS


def _zero_mdat_payload(src: Path, dst: Path) -> None:
    """Keep the container (moov) intact but zero every coded video byte inside the mdat box."""
    data = bytearray(src.read_bytes())
    mdat = data.find(b"mdat")
    size = int.from_bytes(data[mdat - 4 : mdat], "big")
    data[mdat + 4 : mdat - 4 + size] = bytes(size - 8)
    dst.write_bytes(bytes(data))


def test_corrupt_payload_fails_decode(tmp_path: Path) -> None:
    good = tmp_path / "good.mp4"
    make_clip(good, n_frames=10)
    bad = tmp_path / "bad.mp4"
    _zero_mdat_payload(good, bad)
    stats = DecodeStats()
    with pytest.raises(VideoError) as exc:
        list(iter_sampled_frames(bad, probe_with_pyav(good), 5.0, DecodeLimits(), stats))
    assert exc.value.code == errors.VIDEO_DECODE_FAILED
    assert stats.decoded_frames == 0
    with pytest.raises(VideoError) as exc:
        probe_with_pyav(bad)
    assert exc.value.code == errors.VIDEO_DECODE_FAILED


def _corrupt_tail(src: Path, dst: Path, keep_fraction: float) -> None:
    """Overwrite the tail of the mdat payload with 0xFF garbage, keeping the container (moov) intact."""
    data = bytearray(src.read_bytes())
    mdat = data.find(b"mdat")
    size = int.from_bytes(data[mdat - 4 : mdat], "big")
    start = mdat + 4 + int((size - 8) * keep_fraction)
    data[start : mdat - 4 + size] = b"\xff" * (mdat - 4 + size - start)
    dst.write_bytes(bytes(data))


def test_corrupt_tail_stops_decoding_and_is_flagged(tmp_path: Path) -> None:
    good = tmp_path / "good.mp4"
    make_clip(good, n_frames=30)
    bad = tmp_path / "tail.mp4"
    _corrupt_tail(good, bad, 0.5)
    stats = DecodeStats()
    samples = list(iter_sampled_frames(bad, probe_with_pyav(good), 30.0, DecodeLimits(), stats))
    assert 0 < stats.decoded_frames < 30
    assert len(samples) == stats.decoded_frames
    assert stats.stopped_at_decode_error is True
    clean = DecodeStats()
    list(iter_sampled_frames(good, probe_with_pyav(good), 30.0, DecodeLimits(), clean))
    assert clean.stopped_at_decode_error is False and clean.decoded_frames == 30
