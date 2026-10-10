"""Container/stream probing and upload validation (docs/IMPLEMENTATION_SPEC.md section 5, step 1).

Production path: the `ffprobe` binary (present in the backend Linux image), invoked with an argument array (never a
shell), a 15 s timeout and `-protocol_whitelist file` so a crafted container cannot make it open network or other
protocols. `probe_video()` uses it whenever `ffprobe` is on PATH.

Fallback: `probe_with_pyav()` reads the same metadata through PyAV (libavformat). It exists for unit tests on the
Windows developer host, which has no ffprobe binary. It is used only when the caller passes `allow_pyav_fallback=True`
(tests) - production code calls `probe_video()` with the default `False` and gets `probe_unavailable` if ffprobe is
missing, so a broken image cannot silently change the probing implementation.

Rotation convention: `rotation` is the clockwise rotation in degrees (0/90/180/270) that must be applied to decoded
frames for display - the meaning of the legacy `rotate` tag. A display matrix reports the counter-clockwise angle
(ffprobe `side_data_list[].rotation`, PyAV `VideoFrame.rotation`), so clockwise = -ccw mod 360.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av

from app.video import errors
from app.video.errors import VideoError

PROBE_TIMEOUT_S = 15.0
# ffprobe/libavformat names of the ISO-BMFF family (mp4, mov, m4v, 3gp...).
_MOV_FAMILY = "mov,mp4,m4a,3gp,3g2,mj2"
_ALLOWED_FORMAT_TOKENS = frozenset(_MOV_FAMILY.split(","))


@dataclass(frozen=True)
class VideoProbe:
    format_name: str
    duration_s: float
    codec: str
    coded_width: int
    coded_height: int
    rotation: int  # clockwise degrees to apply for display: 0, 90, 180, 270
    fps_avg: float | None  # avg_frame_rate (may differ from the real PTS cadence for VFR)
    nb_frames: int | None
    stream_index: int
    pix_fmt: str | None
    has_audio: bool
    bit_rate: int | None
    probe_tool: str  # "ffprobe" or "pyav"

    @property
    def display_width(self) -> int:
        return self.coded_height if self.rotation in (90, 270) else self.coded_width

    @property
    def display_height(self) -> int:
        return self.coded_width if self.rotation in (90, 270) else self.coded_height

    def as_job_video(self) -> dict[str, Any]:
        """`Job.video` shape of docs/IMPLEMENTATION_SPEC.md section 3.4 (display dimensions after rotation)."""
        return {
            "duration_s": round(self.duration_s, 4),
            "width": self.display_width,
            "height": self.display_height,
            "fps_avg": None if self.fps_avg is None else round(self.fps_avg, 4),
            "rotation": self.rotation,
            "codec": self.codec,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["display_width"] = self.display_width
        data["display_height"] = self.display_height
        return data


@dataclass(frozen=True)
class ProbeLimits:
    max_seconds: float = 120.0
    max_width: int = 1920
    max_height: int = 1080


def normalize_rotation_cw(value: float) -> int:
    """Snap an angle to the nearest multiple of 90 in [0, 360)."""
    return int(round(value / 90.0)) * 90 % 360


def _parse_fraction(text: str | None) -> float | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        value = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 and math.isfinite(value) else None


def _as_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def rotation_from_ffprobe_stream(stream: dict[str, Any]) -> int:
    """Clockwise display rotation from a ffprobe stream: display matrix side data first, then the `rotate` tag."""
    for side in stream.get("side_data_list") or []:
        if not isinstance(side, dict):
            continue
        if "rotation" in side and (ccw := _as_float(side.get("rotation"))) is not None:
            return normalize_rotation_cw(-ccw)
    tag = (stream.get("tags") or {}).get("rotate")
    if (cw := _as_float(tag)) is not None:
        return normalize_rotation_cw(cw)
    return 0


def parse_ffprobe_json(data: dict[str, Any]) -> VideoProbe:
    """Convert `ffprobe -show_format -show_streams` JSON into a VideoProbe (no limit checks)."""
    fmt = data.get("format") or {}
    streams = [s for s in data.get("streams") or [] if isinstance(s, dict)]
    video = [s for s in streams if s.get("codec_type") == "video" and not _is_attached_pic(s)]
    if not video:
        raise VideoError(errors.VIDEO_NO_STREAM, "no video stream found")
    vs = video[0]
    width, height = _as_int(vs.get("width")) or 0, _as_int(vs.get("height")) or 0
    duration = _as_float(vs.get("duration")) or _as_float(fmt.get("duration"))
    if duration is None or duration <= 0:
        raise VideoError(errors.VIDEO_INVALID, "video duration is unknown")
    return VideoProbe(
        format_name=str(fmt.get("format_name") or ""),
        duration_s=duration,
        codec=str(vs.get("codec_name") or "unknown"),
        coded_width=width,
        coded_height=height,
        rotation=rotation_from_ffprobe_stream(vs),
        fps_avg=_parse_fraction(vs.get("avg_frame_rate")) or _parse_fraction(vs.get("r_frame_rate")),
        nb_frames=_as_int(vs.get("nb_frames")),
        stream_index=_as_int(vs.get("index")) or 0,
        pix_fmt=vs.get("pix_fmt"),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        bit_rate=_as_int(fmt.get("bit_rate")),
        probe_tool="ffprobe",
    )


def _is_attached_pic(stream: dict[str, Any]) -> bool:
    return bool((stream.get("disposition") or {}).get("attached_pic"))


def run_ffprobe(path: Path, timeout_s: float = PROBE_TIMEOUT_S, ffprobe: str | None = None) -> dict[str, Any]:
    exe = ffprobe or shutil.which("ffprobe")
    if exe is None:
        raise VideoError(errors.PROBE_UNAVAILABLE, "ffprobe binary not found")
    args = [
        exe,
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        f"file:{Path(path).resolve()}",
    ]
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout_s, check=False)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        raise VideoError(errors.VIDEO_INVALID, "probe timed out") from exc
    if proc.returncode != 0:
        raise VideoError(errors.VIDEO_INVALID, "file is not a readable video container")
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise VideoError(errors.VIDEO_INVALID, "probe output unreadable") from exc
    if not isinstance(data, dict):
        raise VideoError(errors.VIDEO_INVALID, "probe output unreadable")
    return data


def probe_with_pyav(path: Path) -> VideoProbe:
    """Metadata via PyAV. Test fallback for hosts without ffprobe (see module docstring)."""
    try:
        container = av.open(str(path), mode="r")
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise VideoError(errors.VIDEO_INVALID, "file is not a readable video container") from exc
    with container:
        video = [s for s in container.streams.video if not (s.disposition & av.stream.Disposition.attached_pic)]
        if not video:
            raise VideoError(errors.VIDEO_NO_STREAM, "no video stream found")
        vs = video[0]
        duration: float | None = None
        if vs.duration is not None and vs.time_base is not None:
            duration = float(vs.duration * vs.time_base)
        if (duration is None or duration <= 0) and container.duration is not None:
            duration = container.duration / av.time_base
        if duration is None or duration <= 0:
            raise VideoError(errors.VIDEO_INVALID, "video duration is unknown")
        rotation = 0
        tag = vs.metadata.get("rotate")
        if tag is not None and (cw := _as_float(tag)) is not None:
            rotation = normalize_rotation_cw(cw)
        else:
            # Stream-level display matrices are exposed by PyAV on decoded frames only.
            try:
                for frame in container.decode(vs):
                    rotation = normalize_rotation_cw(-float(frame.rotation))
                    break
            except av.FFmpegError as exc:
                raise VideoError(errors.VIDEO_DECODE_FAILED, "first frame could not be decoded") from exc
        rate = vs.average_rate or vs.guessed_rate
        return VideoProbe(
            format_name=container.format.name,
            duration_s=duration,
            codec=vs.codec_context.name,
            coded_width=vs.codec_context.width,
            coded_height=vs.codec_context.height,
            rotation=rotation,
            fps_avg=float(rate) if rate else None,
            nb_frames=vs.frames or None,
            stream_index=vs.index,
            pix_fmt=vs.codec_context.pix_fmt,
            has_audio=len(container.streams.audio) > 0,
            bit_rate=container.bit_rate or None,
            probe_tool="pyav",
        )


def validate_probe(probe: VideoProbe, limits: ProbeLimits) -> VideoProbe:
    """Apply the upload policy. Raises VideoError with the spec codes; returns the probe unchanged when valid."""
    tokens = set(probe.format_name.split(","))
    if not tokens & _ALLOWED_FORMAT_TOKENS:
        raise VideoError(errors.VIDEO_INVALID, "container must be MP4/MOV")
    if probe.coded_width <= 0 or probe.coded_height <= 0:
        raise VideoError(errors.VIDEO_INVALID, "video dimensions are missing")
    if probe.duration_s > limits.max_seconds:
        raise VideoError(errors.VIDEO_TOO_LONG, f"duration {probe.duration_s:.1f} s exceeds {limits.max_seconds:g} s")
    if probe.display_width > limits.max_width or probe.display_height > limits.max_height:
        raise VideoError(
            errors.VIDEO_TOO_LARGE_DIMENSIONS,
            f"display size {probe.display_width}x{probe.display_height} exceeds {limits.max_width}x{limits.max_height}",
        )
    try:
        av.Codec(probe.codec, "r")
    except (av.FFmpegError, ValueError) as exc:
        raise VideoError(errors.VIDEO_DECODE_FAILED, f"codec {probe.codec!r} is not decodable") from exc
    return probe


def probe_video(
    path: Path, limits: ProbeLimits, *, allow_pyav_fallback: bool = False, timeout_s: float = PROBE_TIMEOUT_S
) -> VideoProbe:
    """Probe and validate an uploaded file. Uses ffprobe; PyAV only when explicitly allowed and ffprobe is absent."""
    if shutil.which("ffprobe") is None:
        if not allow_pyav_fallback:
            raise VideoError(errors.PROBE_UNAVAILABLE, "ffprobe binary not found")
        probe = probe_with_pyav(path)
    else:
        probe = parse_ffprobe_json(run_ffprobe(path, timeout_s))
    return validate_probe(probe, limits)
