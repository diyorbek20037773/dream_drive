"""Headless LiteRoad run from the command line.

    python -m app.agents.cli --scenario normal --seed 1 --start A --dest C --out runs/demo [--video] [--overlay]

Writes ``summary.json`` (measured run statistics), ``report.json`` (RunReport + details; only when a 40-64 hex
commit is known from ``--commit``, ``GIT_COMMIT`` or the HEAD of a clean git checkout) and, with ``--video``,
``camera.mp4``: H.264 (libx264, yuv420p, +faststart, no audio) at 1/dt fps of exactly the frames perception
consumed. Every video frame carries a visible "LiteRoad synthetic render" label so it can never be mistaken for
real camera footage.
Exit code: 0 when the run reached a terminal state (any outcome), 2 for invalid arguments / unreachable routes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.agents.runner import RouteUnreachable, SessionConfig, SimulationSession, UnsupportedMode
from app.simulators.lite.scenarios import SCENARIOS

_COMMIT = re.compile(r"^[a-f0-9]{40,64}$")


def _git_commit(explicit: str | None) -> str | None:
    """Commit for report.json: --commit, GIT_COMMIT, or HEAD of a *clean* checkout (a dirty tree is not HEAD)."""
    for cand in (explicit, os.environ.get("GIT_COMMIT")):
        if cand and _COMMIT.match(cand):
            return cand
    repo = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, cwd=repo, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, cwd=repo, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = head.stdout.strip()
    if head.returncode != 0 or status.returncode != 0 or status.stdout.strip() or not _COMMIT.match(sha):
        return None
    return sha


def _draw_overlay(img: np.ndarray, live: dict[str, Any], draw_perception: bool) -> np.ndarray:
    out = np.ascontiguousarray(img.copy())
    if draw_perception:
        perc = live["perception"]
        pts = np.asarray(perc["lane_points"], dtype=np.float32)
        if len(pts) >= 2:
            cv2.polylines(out, [np.round(pts).astype(np.int32)], False, (0, 220, 255), 2, cv2.LINE_AA)
        for x1, y1, x2, y2 in perc["obstacle_boxes"]:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
    cmd = live.get("command") or {}
    lines = [
        "LiteRoad synthetic render | synthetic_rgb_cv",
        f"frame {live['frame_id']}  t={live['sim_time_s']:.2f}s  v={live['ego']['speed_mps']:.2f} m/s",
        f"{live['decision']['state']}  steer={cmd.get('steer', 0):+.2f} thr={cmd.get('throttle', 0):.2f} "
        f"brk={cmd.get('brake', 0):.2f}",
    ]
    cv2.rectangle(out, (0, 0), (430, 62), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (8, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class _Mp4Writer:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        import av  # PyAV wheel ships libx264

        self.container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        self.container.metadata["title"] = "DreamDrive LiteRoad synthetic render (not real camera footage)"
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "20", "preset": "veryfast"}
        self._av = av

    def write(self, rgb: np.ndarray) -> None:
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m app.agents.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenario", default="normal", choices=sorted(SCENARIOS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", default=None, help="start node id (default: the scenario's suite route)")
    ap.add_argument("--dest", default=None, help="destination node id (default: the scenario's suite route)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--video", action="store_true", help="write camera.mp4 (H.264) of the consumed frames")
    ap.add_argument("--overlay", action="store_true", help="draw perceived lane points / obstacle boxes on the video")
    ap.add_argument("--realtime", action="store_true", help="pace simulated time to wall time")
    ap.add_argument("--max-ticks", type=int, default=2400)
    ap.add_argument("--commit", default=None)
    args = ap.parse_args(argv)

    spec = SCENARIOS[args.scenario]
    start = args.start or spec.suite_route[0]
    dest = args.dest or spec.suite_route[1]
    cfg = SessionConfig(
        run_id=f"cli-{args.scenario}-{args.seed}",
        start_node=start,
        destination_node=dest,
        scenario=args.scenario,
        seed=args.seed,
        max_ticks=args.max_ticks,
        encode_jpeg=False,
    )
    try:
        session = SimulationSession(cfg)
    except (RouteUnreachable, UnsupportedMode) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    writer: _Mp4Writer | None = None
    t0 = time.monotonic()
    result = session.start()
    if args.video:
        img = session.last_frame_image
        assert img is not None
        writer = _Mp4Writer(args.out / "camera.mp4", img.shape[1], img.shape[0], int(round(1.0 / cfg.dt)))
    try:
        while True:
            if writer is not None and session.last_frame_image is not None:
                writer.write(_draw_overlay(session.last_frame_image, result.live, args.overlay))
            if result.terminal:
                break
            result = session.tick()
            session.note_wall_time(time.monotonic() - t0)
            if args.realtime:
                ahead = result.sim_time_s - (time.monotonic() - t0)
                if ahead > 0:
                    time.sleep(ahead)
    finally:
        if writer is not None:
            writer.close()
    wall = time.monotonic() - t0
    summary = session.summary(wall)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    commit = _git_commit(args.commit)
    if commit:
        (args.out / "report.json").write_text(json.dumps(session.report(commit, wall), indent=2), encoding="utf-8")
    else:
        print(
            "note: no clean git commit known; report.json not written (summary.json has the measurements)",
            file=sys.stderr,
        )
    print(
        json.dumps(
            {
                "outcome": summary["outcome"],
                "ticks": summary["ticks"],
                "sim_time_s": summary["sim_time_s"],
                "wall_time_s": summary["wall_time_s"],
                "collisions": summary["evaluator"]["collisions"],
                "out": str(args.out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
