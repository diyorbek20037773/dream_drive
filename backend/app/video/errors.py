"""Error types raised by the uploaded-video analysis pipeline.

`VideoError.code` values are stable machine codes; the API maps them to localized messages. Messages never contain
filesystem paths or stack traces because they may be shown to the uploader.
"""

from __future__ import annotations

from typing import Final

# Validation / decode (docs/IMPLEMENTATION_SPEC.md section 5)
VIDEO_INVALID: Final = "video_invalid"
VIDEO_TOO_LONG: Final = "video_too_long"
VIDEO_TOO_LARGE_DIMENSIONS: Final = "video_too_large_dimensions"
VIDEO_NO_STREAM: Final = "video_no_stream"
VIDEO_DECODE_FAILED: Final = "video_decode_failed"
# Additional pipeline failure codes (documented in the component report / docs/API.md error table)
VIDEO_DECODE_TIMEOUT: Final = "video_decode_timeout"
VIDEO_TOO_MANY_FRAMES: Final = "video_too_many_frames"
PROBE_UNAVAILABLE: Final = "probe_unavailable"
INFERENCE_UNAVAILABLE: Final = "inference_unavailable"
INFERENCE_FAILED: Final = "inference_failed"
MODEL_CHANGED: Final = "model_changed"
BUILD_IDENTITY_MISSING: Final = "build_identity_missing"

ALL_CODES: Final = frozenset(
    {
        VIDEO_INVALID,
        VIDEO_TOO_LONG,
        VIDEO_TOO_LARGE_DIMENSIONS,
        VIDEO_NO_STREAM,
        VIDEO_DECODE_FAILED,
        VIDEO_DECODE_TIMEOUT,
        VIDEO_TOO_MANY_FRAMES,
        PROBE_UNAVAILABLE,
        INFERENCE_UNAVAILABLE,
        INFERENCE_FAILED,
        MODEL_CHANGED,
        BUILD_IDENTITY_MISSING,
    }
)


class VideoError(Exception):
    """A user-visible, coded failure of a video analysis job."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message or code


class JobCancelled(Exception):  # noqa: N818 - name fixed by docs/IMPLEMENTATION_SPEC.md
    """Raised when `cancel_check()` reports a cancellation request between frames."""
