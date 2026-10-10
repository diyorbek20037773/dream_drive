"""Probe parsing and upload validation (spec section 5 step 1)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_video_synth import make_clip

from app.video import errors
from app.video.errors import VideoError
from app.video.probe import (
    ProbeLimits,
    VideoProbe,
    parse_ffprobe_json,
    probe_video,
    probe_with_pyav,
    rotation_from_ffprobe_stream,
    run_ffprobe,
    validate_probe,
)


def _ffprobe_json(**stream_overrides: Any) -> dict[str, Any]:
    stream: dict[str, Any] = {
        "index": 0,
        "codec_name": "h264",
        "codec_type": "video",
        "width": 1280,
        "height": 720,
        "pix_fmt": "yuv420p",
        "avg_frame_rate": "30000/1001",
        "r_frame_rate": "30000/1001",
        "duration": "10.010000",
        "nb_frames": "300",
        "disposition": {"attached_pic": 0},
    }
    stream.update(stream_overrides)
    return {
        "streams": [stream, {"index": 1, "codec_type": "audio", "codec_name": "aac"}],
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "10.010000", "bit_rate": "4000000"},
    }


def _require_ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if exe is None:
        if os.environ.get("DD_REQUIRE_LIVE") == "1":
            pytest.fail("ffprobe binary is required when DD_REQUIRE_LIVE=1 (backend image must ship ffmpeg)")
        pytest.skip("no ffprobe binary on this host (Windows dev host); this test runs in the Linux backend image")
    return exe


class TestParseFfprobe:
    def test_basic_fields(self) -> None:
        probe = parse_ffprobe_json(_ffprobe_json())
        assert probe.codec == "h264"
        assert (probe.coded_width, probe.coded_height) == (1280, 720)
        assert probe.duration_s == pytest.approx(10.01)
        assert probe.fps_avg == pytest.approx(29.97, abs=0.01)
        assert probe.nb_frames == 300
        assert probe.has_audio is True
        assert probe.rotation == 0
        assert probe.probe_tool == "ffprobe"

    @pytest.mark.parametrize(
        ("side_rotation", "expected_cw"), [(-90, 90), (90, 270), (180, 180), (-180, 180), (0, 0), (-89.6, 90)]
    )
    def test_display_matrix_rotation_is_converted_to_clockwise(self, side_rotation: float, expected_cw: int) -> None:
        stream = {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": side_rotation}]}
        assert rotation_from_ffprobe_stream(stream) == expected_cw

    def test_legacy_rotate_tag(self) -> None:
        assert rotation_from_ffprobe_stream({"tags": {"rotate": "90"}}) == 90
        assert rotation_from_ffprobe_stream({"tags": {"rotate": "270"}}) == 270

    def test_side_data_wins_over_tag(self) -> None:
        stream = {"tags": {"rotate": "180"}, "side_data_list": [{"rotation": -90}]}
        assert rotation_from_ffprobe_stream(stream) == 90

    def test_rotated_display_dimensions(self) -> None:
        probe = parse_ffprobe_json(_ffprobe_json(side_data_list=[{"rotation": -90}]))
        assert (probe.display_width, probe.display_height) == (720, 1280)
        assert probe.as_job_video()["width"] == 720
        assert probe.as_job_video()["rotation"] == 90

    def test_no_video_stream(self) -> None:
        data = _ffprobe_json()
        data["streams"] = [s for s in data["streams"] if s["codec_type"] != "video"]
        with pytest.raises(VideoError) as exc:
            parse_ffprobe_json(data)
        assert exc.value.code == errors.VIDEO_NO_STREAM

    def test_attached_picture_is_not_a_video_stream(self) -> None:
        data = _ffprobe_json(disposition={"attached_pic": 1})
        with pytest.raises(VideoError) as exc:
            parse_ffprobe_json(data)
        assert exc.value.code == errors.VIDEO_NO_STREAM

    def test_missing_duration_is_invalid(self) -> None:
        data = _ffprobe_json(duration=None)
        data["format"].pop("duration")
        with pytest.raises(VideoError) as exc:
            parse_ffprobe_json(data)
        assert exc.value.code == errors.VIDEO_INVALID


class TestValidate:
    def _probe(self, **kw: Any) -> VideoProbe:
        base = parse_ffprobe_json(_ffprobe_json())
        return VideoProbe(**{**base.__dict__, **kw})

    def test_valid(self) -> None:
        probe = self._probe()
        assert validate_probe(probe, ProbeLimits()) is probe

    def test_too_long(self) -> None:
        with pytest.raises(VideoError) as exc:
            validate_probe(self._probe(duration_s=121.0), ProbeLimits(max_seconds=120))
        assert exc.value.code == errors.VIDEO_TOO_LONG

    def test_too_large_after_rotation(self) -> None:
        # 1920x1080 landscape is allowed, the same stream rotated to portrait is 1080x1920 and exceeds 1920x1080.
        validate_probe(self._probe(coded_width=1920, coded_height=1080), ProbeLimits())
        with pytest.raises(VideoError) as exc:
            validate_probe(self._probe(coded_width=1920, coded_height=1080, rotation=90), ProbeLimits())
        assert exc.value.code == errors.VIDEO_TOO_LARGE_DIMENSIONS

    def test_non_mp4_container_rejected(self) -> None:
        with pytest.raises(VideoError) as exc:
            validate_probe(self._probe(format_name="matroska,webm"), ProbeLimits())
        assert exc.value.code == errors.VIDEO_INVALID

    def test_undecodable_codec(self) -> None:
        with pytest.raises(VideoError) as exc:
            validate_probe(self._probe(codec="no_such_codec_xyz"), ProbeLimits())
        assert exc.value.code == errors.VIDEO_DECODE_FAILED


class TestPyavFallback:
    def test_cfr_clip(self, tmp_path: Path) -> None:
        path = tmp_path / "a.mp4"
        make_clip(path, n_frames=30, fps=30.0, size=(320, 180))
        probe = probe_with_pyav(path)
        assert probe.probe_tool == "pyav"
        assert probe.codec == "h264"
        assert (probe.coded_width, probe.coded_height) == (320, 180)
        assert probe.duration_s == pytest.approx(1.0, abs=0.05)
        assert probe.rotation == 0
        assert probe.has_audio is False
        validate_probe(probe, ProbeLimits())

    def test_rotation_from_display_matrix(self, tmp_path: Path) -> None:
        path = tmp_path / "rot.mp4"
        make_clip(path, n_frames=5, rotation_ccw=90, size=(320, 180))
        probe = probe_with_pyav(path)
        assert probe.rotation == 270  # 90 deg counter-clockwise == 270 deg clockwise
        assert (probe.display_width, probe.display_height) == (180, 320)

    def test_audio_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "audio.mp4"
        make_clip(path, n_frames=30, with_audio=True)
        assert probe_with_pyav(path).has_audio is True

    def test_garbage_bytes_are_invalid(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.mp4"
        path.write_bytes(os.urandom(4096))
        with pytest.raises(VideoError) as exc:
            probe_with_pyav(path)
        assert exc.value.code == errors.VIDEO_INVALID

    def test_production_path_never_silently_falls_back(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "a.mp4"
        make_clip(path, n_frames=5)
        monkeypatch.setattr("app.video.probe.shutil.which", lambda _name: None)
        with pytest.raises(VideoError) as exc:
            probe_video(path, ProbeLimits())
        assert exc.value.code == errors.PROBE_UNAVAILABLE
        assert probe_video(path, ProbeLimits(), allow_pyav_fallback=True).probe_tool == "pyav"


class TestLiveFfprobe:
    """Production probe path. Skipped only on hosts without ffprobe; fails under DD_REQUIRE_LIVE=1."""

    def test_ffprobe_matches_pyav_metadata(self, tmp_path: Path) -> None:
        exe = _require_ffprobe()
        path = tmp_path / "rot.mp4"
        make_clip(path, n_frames=15, fps=15.0, rotation_ccw=90, size=(320, 180), with_audio=True)
        probe = parse_ffprobe_json(run_ffprobe(path, ffprobe=exe))
        ref = probe_with_pyav(path)
        assert probe.probe_tool == "ffprobe"
        assert probe.codec == ref.codec == "h264"
        assert (probe.coded_width, probe.coded_height) == (ref.coded_width, ref.coded_height)
        assert probe.rotation == ref.rotation == 270
        assert probe.has_audio is True
        assert probe.duration_s == pytest.approx(ref.duration_s, abs=0.1)
        validate_probe(probe, ProbeLimits())

    def test_ffprobe_rejects_garbage(self, tmp_path: Path) -> None:
        exe = _require_ffprobe()
        path = tmp_path / "bad.mp4"
        path.write_bytes(b"\x00" * 1024 + os.urandom(2048))
        with pytest.raises(VideoError) as exc:
            parse_ffprobe_json(run_ffprobe(path, ffprobe=exe))
        assert exc.value.code in (errors.VIDEO_INVALID, errors.VIDEO_NO_STREAM)
