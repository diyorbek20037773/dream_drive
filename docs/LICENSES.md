# Licenses and third-party notices

This page separates **code licenses**, **model-weight terms** and **data rights**; one never implies another.
It records evidence found in the repository and upstream sources; it is not legal advice and has not been reviewed
by a lawyer.

## DreamDrive source

No project license file has been selected yet (owner decision). Until one is added, the repository's own code is not
offered under an open-source license.

## Vendored source code

| Component | Upstream / pinned commit | License | Where | Modifications |
|---|---|---|---|---|
| ByteTrack tracker (`basetrack`, `kalman_filter`, `matching`, `byte_tracker`) | https://github.com/FoundationVision/ByteTrack @ `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` | MIT, (c) 2021 Yifu Zhang | `backend/app/perception/vendored/bytetrack/` (`LICENSE` byte-identical to upstream, `NOTICE.md` with upstream file SHA256s) | listed in `NOTICE.md`: scipy assignment instead of `lap`, numpy IoU instead of `cython_bbox`, per-run ID counter, timestamp-aware prediction/expiry, pixel-space N x 5 input, detection back-reference |
| YOLOPv2 helper functions (`utils/utils.py` subset) | https://github.com/CAIC-AD/YOLOPv2 @ `9388c6aa55f3bf5a878679a1338bb090cff6a256` | MIT, (c) 2022 CAIC-AD | `inference/app/vendored/` (`NOTICE`, module docstring) | listed in the module docstring |

## Model weights

| Model | Source | Evidence | Distribution |
|---|---|---|---|
| YOLOPv2 `yolopv2.pt` (SHA256 `f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c`) | GitHub release V0.0.1 of CAIC-AD/YOLOPv2 | The repository code is MIT; **no separate weight license is published**. Upstream states training on BDD100K, whose dataset terms apply to the training data. See `config/models.resolved.yaml` (`weight_license_evidence`). | Not committed and not baked into images: `model-init` downloads from the upstream URL and verifies SHA256. |

## Test data

| Fixture | License | Evidence |
|---|---|---|
| Wikimedia Commons "Dashcam recording example - Panorama II BlackBox Dashcam daylight.mp4.ogv" by Fernost | CC0 1.0 | Commons API extmetadata + `{{self|cc-zero}}` page template, verified 2026-10-10; details and hashes in `docs/DATA_SOURCES.md` and `tests/fixtures/manifest.json`. Not committed. |

## Key runtime dependencies (backend)

Versions are pinned by `backend/uv.lock`; license fields below were read from the installed package metadata
(`importlib.metadata`) in the backend environment on 2026-10-10.

| Package | Version | Declared license |
|---|---|---|
| av (PyAV) | 18.1.0 | BSD-3-Clause (PyAV itself) |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| opencv-python-headless | 5.0.0.93 | Apache 2.0 |
| scipy | 1.17.1 | BSD License |
| httpx | 0.28.1 | BSD-3-Clause |
| pydantic / pydantic-settings | 2.14.0 / 2.15.0 | MIT |
| fastapi | 0.143.0 | MIT |
| jsonschema | 4.26.0 | MIT |
| pyyaml | 6.0.3 | MIT |
| redis / rq | 8.1.0 / 2.12.0 | MIT / BSD-2-Clause |

### FFmpeg, x264 and x265 inside the PyAV wheel

The PyAV 18.1.0 binary wheel bundles FFmpeg 8.1.2 shared libraries. On the Windows wheel inspected,
`av._core.library_meta["libavcodec"]` reports the license string "LGPL version 3 or later" and the configure line
contains `--enable-version3`, `--enable-libx264`, `--enable-libx265`, `--enable-libvpx`, `--enable-libsvtav1` and
`--enable-libdav1d`; the wheel ships `libx264` and `libx265` DLLs. Upstream x264 and x265 are licensed
GPL-2.0-or-later. DreamDrive uses libx264 (through PyAV) to encode `annotated.mp4` and the derived test clip.
Publishing the backend image therefore redistributes these binaries; whoever publishes images must meet the
corresponding source-availability obligations (upstream sources: https://github.com/PyAV-Org/PyAV,
https://ffmpeg.org, https://code.videolan.org/videolan/x264, https://bitbucket.org/multicoreware/x265_git).
The production probe path also needs an `ffprobe` binary in the Linux backend image (for example from the Debian
`ffmpeg` package); its license follows that package's metadata.

Inference-service dependencies (PyTorch etc.) are pinned in `inference/uv.lock`; their licenses are recorded by the
inference component and are not repeated here.
