# Ported from ByteTrack yolox/tracker/basetrack.py @ d1bf0191adff59bc8fcfeaa0b33d3d1642552a99
# (MIT, (c) 2021 Yifu Zhang).
# DreamDrive modifications (see NOTICE.md): per-instance IdCounter instead of the global class counter; type hints;
# timestamps of the first/last observation; unused multi-camera/feature fields removed.
from __future__ import annotations


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class IdCounter:
    """Monotonic track-ID source owned by one tracker run (upstream used a process-global counter)."""

    def __init__(self) -> None:
        self._count = 0

    def next_id(self) -> int:
        self._count += 1
        return self._count

    @property
    def issued(self) -> int:
        return self._count


class BaseTrack:
    def __init__(self) -> None:
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.score = 0.0
        self.start_frame = 0
        self.frame_id = 0
        self.start_time_s = 0.0
        self.last_time_s = 0.0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed
