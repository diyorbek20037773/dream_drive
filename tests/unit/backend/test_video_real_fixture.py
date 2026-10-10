"""Checks on the licensed real-road fixture (CC0 Wikimedia Commons dash-cam clip, see docs/DATA_SOURCES.md).

The clip is downloaded by scripts/fetch_fixtures.py into tests/fixtures/cache/ (git-ignored). These tests skip when
it is absent on a developer host and FAIL when DD_REQUIRE_LIVE=1 (QA gate), so a missing fixture never counts as a pass.
No inference here: real-model runs belong to the model-smoke/integration gates.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import pytest

from app.perception.video_decisions import frame_quality
from app.video.decode import DecodeLimits, DecodeStats, iter_sampled_frames
from app.video.probe import ProbeLimits, probe_video

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests" / "fixtures" / "manifest.json"
CACHE = ROOT / "tests" / "fixtures" / "cache"


def _fixture() -> tuple[Path, dict]:
    entry = json.loads(MANIFEST.read_text(encoding="utf-8"))["fixtures"][0]
    path = CACHE / entry["derived"]["file_name"]
    if not path.exists():
        if os.environ.get("DD_REQUIRE_LIVE") == "1":
            pytest.fail(f"fixture {path.name} missing: run scripts/fetch_fixtures.py (required when DD_REQUIRE_LIVE=1)")
        pytest.skip("real-road fixture not downloaded (python scripts/fetch_fixtures.py)")
    return path, entry


def test_manifest_records_license_and_hashes() -> None:
    entry = json.loads(MANIFEST.read_text(encoding="utf-8"))["fixtures"][0]
    src = entry["source"]
    assert src["license"] == "CC0 1.0" and src["license_template"] == "{{self|cc-zero}}"
    assert src["page_url"].startswith("https://commons.wikimedia.org/wiki/File:")
    assert len(src["sha256"]) == 64 and len(src["sha1"]) == 40
    assert len(entry["derived"]["sha256"]) == 64
    assert entry["committed"] is False
    assert entry["derived"]["duration_s"] <= 12


def test_fixture_hash_probe_and_sampling() -> None:
    path, entry = _fixture()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["derived"]["sha256"]
    probe = probe_video(path, ProbeLimits(), allow_pyav_fallback=True)
    assert probe.codec == "h264" and (probe.display_width, probe.display_height) == (1280, 720)
    assert probe.rotation == 0 and not probe.has_audio
    assert probe.duration_s == pytest.approx(10.0, abs=0.05)
    stats = DecodeStats()
    samples = list(iter_sampled_frames(path, probe, 5.0, DecodeLimits(), stats))
    assert stats.decoded_frames == 300
    assert len(samples) == 50
    assert [s.pts_s for s in samples] == pytest.approx([k / 5.0 for k in range(50)], abs=1e-3)


def test_quality_thresholds_on_real_daylight_frames() -> None:
    path, _ = _fixture()
    probe = probe_video(path, ProbeLimits(), allow_pyav_fallback=True)
    frames = [s.image for s in iter_sampled_frames(path, probe, 1.0, DecodeLimits(), DecodeStats())]
    assert len(frames) == 10
    for img in frames:
        q = frame_quality(img)
        assert not q.dark and not q.blurry, q
        assert frame_quality(cv2.GaussianBlur(img, (0, 0), 6)).blurry
        assert frame_quality((img * 0.3).astype(img.dtype)).dark


# ---------------------------------------------------------------------------------------------------------------
# Second real fixture: CC BY 3.0 downtown street driving (traffic lights, pedestrians), docs/DATA_SOURCES.md
# ---------------------------------------------------------------------------------------------------------------
STREET_ID = "commons_memphis_front_st_ccby3"


def _street_entry() -> dict:
    entries = [e for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["fixtures"] if e["id"] == STREET_ID]
    assert len(entries) == 1
    return entries[0]


def test_street_fixture_manifest_records_license_attribution_and_hashes() -> None:
    entry = _street_entry()
    src = entry["source"]
    assert src["license"] == "CC BY 3.0" and src["license_template"] == "{{self|GFDL|cc-by-3.0}}"
    assert "Thomas R Machnitzki" in src["author"] and "Attribution" in src["license_note"]
    assert src["page_url"].startswith("https://commons.wikimedia.org/wiki/File:")
    assert len(src["sha256"]) == 64 and len(src["sha1"]) == 40
    assert len(entry["derived"]["sha256"]) == 64 and entry["derived"]["duration_s"] <= 12
    assert entry["committed"] is False


def test_street_fixture_hash_probe_and_sampling() -> None:
    entry = _street_entry()
    path = CACHE / entry["derived"]["file_name"]
    if not path.exists():
        if os.environ.get("DD_REQUIRE_LIVE") == "1":
            pytest.fail(f"fixture {path.name} missing: run scripts/fetch_fixtures.py (required when DD_REQUIRE_LIVE=1)")
        pytest.skip("street fixture not downloaded (python scripts/fetch_fixtures.py)")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["derived"]["sha256"]
    probe = probe_video(path, ProbeLimits(), allow_pyav_fallback=True)
    assert probe.codec == "h264" and (probe.display_width, probe.display_height) == (640, 480)
    assert not probe.has_audio and probe.duration_s <= 12.0
    stats = DecodeStats()
    samples = list(iter_sampled_frames(path, probe, 5.0, DecodeLimits(), stats))
    assert stats.decoded_frames == entry["derived"]["frames"] == 358
    assert len(samples) == 60
    assert [s.pts_s for s in samples] == pytest.approx([k / 5.0 for k in range(60)], abs=2e-3)
