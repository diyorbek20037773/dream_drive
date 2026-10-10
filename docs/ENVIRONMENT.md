# Developer environment audit

Audited 2026-10-10 11:06 UTC (16:06 Asia/Tashkent) by Claude Code. Sanitized facts only; no secrets.

## Host
| Item | Observed value | Note |
|---|---|---|
| OS | Windows 11 Pro 10.0.26300 | Claude Code runs on Windows (Git Bash + PowerShell) |
| CPU | 13th Gen Intel Core i5-13420H, 8 cores / 12 logical | |
| RAM | 15.7 GiB host; Docker Desktop VM reports 7.6 GiB | WSL VM ~7 GiB |
| Disk free | C: 29.1 GB, D: 74.2 GB (project on D:) | |
| GPU | **NVIDIA GeForce RTX 3050 6GB Laptop GPU**, 6144 MiB, driver 576.52, CUDA 12.9 (nvidia-smi) | Owner brief said RTX 4050 6 GB; `nvidia-smi` reports RTX 3050 6 GB. VRAM matches. |
| WSL | Ubuntu 24.04.3 LTS (WSL2), python3 3.12.3, node, docker CLI | no ffmpeg/redis/uv inside WSL |
| Docker | Docker Desktop 4.88.0, Engine 29.7.2, Compose v5.4.0, runtimes: runc, nvidia | linux/amd64 engine |
| Host tools | Python 3.11.9/3.12/3.13.5, uv 0.12.9, Node 22.17.0, npm 11.12.1, git 2.53, gh 2.96.0, GNU make | ffmpeg not on host PATH (used inside containers) |

## Accounts and repository
- `gh auth status`: logged in to github.com as `diyorbek20037773` (keyring), scopes `repo`, `workflow`, `gist`, `read:org`.
- `git ls-remote https://github.com/diyorbek20037773/dream_drive.git`: no refs. `gh repo view`: public, `isEmpty: true`.
- Local folder was not a git repository; only the handoff kit existed. `git init -b main` performed; kit files copied to root (identical CLAUDE.md).
- Docker credential store `desktop` has an entry for `https://index.docker.io/v1/` (account name not printed). Push permission verified only at publish time.

## Model spike
Date 2026-10-10. Container `python:3.11-slim-bookworm` + `uv sync --frozen` (inference/uv.lock): torch 2.14.1+cpu, torchvision 0.29.1+cpu, numpy 2.4.6, opencv-python-headless 5.0.0.93.

- Download: `https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt` → 156,380,200 bytes, SHA256 `f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c`.
- Upstream source revision `9388c6aa55f3bf5a878679a1338bb090cff6a256` (2023-03-24), MIT.
- `torch.jit.load` OK (1.5 s, FutureWarning that `torch.jit.load` is deprecated). 38,955,000 parameters, float32.
- Outputs for 1×3×384×640: detection heads 1×255×{48×80, 24×40, 12×20} + anchor grids 1×3×1×1×2; seg 1×2×384×640; ll 1×1×384×640, both already in probability range 0..1.
- 14 upstream demo images (1280×720): CPU inference 386–631 ms/frame in Docker (torch 6 threads), 0–20 NMS detections/frame, drivable fraction 0.08–0.37, lane fraction 0.003–0.042.
- Class audit: all 913 anchors with objectness > 0.3 have argmax class 3; all detections class 3. Class map resolved to `{3: vehicle}` (see config/models.resolved.yaml for evidence and limits).
- Graph accepted 384×640, 480×640 and 640×640 canvases.

These are spike observations, not release benchmarks. Release benchmarks are produced by `make model-smoke` and stored under `artifacts/qa/<commit>/`.

## Deadline
The source brief states 2026-10-11 13:00 Asia/Tashkent. At audit time about 21 hours remained.
