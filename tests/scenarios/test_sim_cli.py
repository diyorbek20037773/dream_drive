"""Headless CLI: summary, labelled H.264 camera video, report only with an explicit valid commit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.timeout(300)


def test_cli_writes_summary_video_and_report(tmp_path: Path) -> None:
    import av

    from app.agents.cli import main

    out = tmp_path / "run"
    commit = "89abcdef0123456789abcdef0123456789abcdef"  # explicit test input
    assert main(["--scenario", "camera_blackout", "--seed", "1", "--out", str(out), "--video", "--commit", commit]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["outcome"] == "stopped_camera_invalid"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["code_commit"] == commit and report["status"] == "stopped"
    with av.open(str(out / "camera.mp4")) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == (640, 360)
        assert container.metadata.get("title", "").startswith("DreamDrive LiteRoad synthetic render")
        frames = sum(1 for _ in container.decode(video=0))
    assert frames == summary["ticks"] + 1, "one video frame per consumed camera frame (start + every tick)"


def test_cli_rejects_unreachable_destination(tmp_path: Path) -> None:
    from app.agents.cli import main

    assert main(["--start", "A", "--dest", "Q", "--out", str(tmp_path / "x")]) == 2
