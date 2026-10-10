# Model card: YOLOPv2 in DreamDrive

Status: verified checkpoint, real inference smoke passed on CPU (fp32) and CUDA (fp16) on 2026-10-10.
Nothing in this card is an accuracy claim. No labelled DreamDrive evaluation set exists yet, so precision, recall,
mAP and mask IoU against ground truth are **not evaluated**.

## Identity
| Field | Value |
|---|---|
| Model id (DreamDrive) | `yolopv2` |
| Upstream repository | https://github.com/CAIC-AD/YOLOPv2 |
| Upstream source revision | `9388c6aa55f3bf5a878679a1338bb090cff6a256` (2023-03-24) |
| Checkpoint URL | https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt |
| Checkpoint SHA256 | `f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c` |
| Checkpoint size | 156,380,200 bytes |
| Format | TorchScript archive loaded with `torch.jit.load` (38,955,000 float32 parameters, measured in the spike) |
| Manifest | `config/models.resolved.yaml` (the only source of the expected hash/size/class map) |
| Runtime (CPU image) | Python 3.11, torch 2.14.1+cpu, torchvision 0.29.1+cpu (`inference/uv.lock`) |
| Runtime (GPU image) | Python 3.11, torch 2.14.1+cu126, torchvision 0.29.1+cu126 (`inference/gpu/uv.lock`) |

## Provenance and licence evidence
- Code: MIT, Copyright (c) 2022 CAIC-AD (upstream `LICENSE`). The decode/NMS/letterbox helpers DreamDrive uses are
  vendored in `inference/app/vendored/yolopv2_utils.py` with the MIT notice, the pinned commit and a list of
  modifications; see `inference/app/vendored/NOTICE`.
- Weights: no separate weight licence is published. The checkpoint is a GitHub release asset (V0.0.1) of the
  MIT-licensed repository; the upstream README states it was trained on BDD100K, whose dataset terms apply to the
  training data. DreamDrive therefore does **not** redistribute the weights in any image or in git. The one-shot
  `python -m app.model_init` service downloads them from the upstream URL into the `model-cache` volume and verifies
  size and SHA256 before an atomic rename.
- Test fixture: `tests/fixtures/images/yolopv2_example.jpg` is the upstream `data/example.jpg` at the pinned commit
  (MIT repository; provenance and hash in `tests/fixtures/images/README.md`).

## Outputs (verified on the real graph)
For a 1x3xHxW input the graph returns `[pred, anchor_grid], seg, ll`:
- `pred`: 3 raw heads `[1,255,H/8,W/8]`, `[1,255,H/16,W/16]`, `[1,255,H/32,W/32]` (3 anchors x 85 = 4 box +
  1 objectness + 80 class logits); `anchor_grid`: 3 tensors `[1,3,1,1,2]`. Decoded with upstream
  `split_for_trace_model` + `non_max_suppression` (vendored).
- `seg` `[1,2,H,W]` per-pixel 2-channel probabilities; drivable = argmax channel == 1.
- `ll` `[1,1,H,W]` per-pixel probability; lane = probability > 0.5 (identical to upstream `torch.round`, which rounds
  0.5 down to 0).
The service validates this structure at warm-up and refuses to become ready if it differs.

## Classes
| Class index | Name | Status |
|---|---|---|
| 3 | `vehicle` | Verified as the only class this checkpoint emits (spike: all 913 anchors with objectness > 0.3 on 14 upstream demo frames had argmax class 3; every NMS detection was class 3). The name follows the upstream README's single "traffic object detection" task; no multi-class list is published. |
| any other | `unverified_class_<id>` | Reported verbatim, never renamed, and excluded from decisions. None were observed in the smoke runs. |

**Not supported** by this checkpoint (must not be claimed in UI, reports or decisions): person/pedestrian, rider,
bicycle, motorcycle as a separate class, traffic light, traffic sign, animals, road debris and any other category.
There is no depth, distance, speed or time-to-collision output.

## Input policy (`letterbox640_stride32`)
- Accepted request bodies: JPEG or PNG (declared `Content-Type` must match the magic bytes), at most 8 MiB, decoded
  size 16x16 to 4096x4096 (checked from the header before decoding). EXIF orientation is ignored: callers send frames
  already in display orientation.
- Letterbox: scale the longest side to 640 (upscaling allowed), pad to the minimal multiple of 32 with colour 114,
  BGR -> RGB, float32 / 255. A 1280x720 frame becomes a 640x384 canvas with 12 px padding top and bottom.
  The canvas is pixel-identical to upstream `letterbox(auto=True)` (unit-tested for 10 sizes).
- Difference from the upstream demo: upstream `LoadImages` first resizes *every* image to 1280x720 (stretching
  non-16:9 inputs). DreamDrive letterboxes the original instead so boxes and masks can be mapped back to the original
  pixels exactly. Inputs that are not 16:9 therefore produce canvases (e.g. 384x640 portrait, 640x480 for 4:3) that
  the network did not see during training; quality there is unknown.
- Inverse transform: boxes subtract the applied integer padding and divide by the per-axis content scale, then clip to
  the image; masks crop the padding, resize the probabilities bilinearly to the original size, then threshold.
  Unit tests prove equality with upstream `driving_area_mask`/`lane_line_mask` for 1280x720 and exact box recovery
  for 1920x1080, 720x1280, 640x480 and an odd-padding 1280x730 input.
- Defaults: `conf_threshold=0.1` (low so ByteTrack's second association sees low-score boxes; upstream demo uses
  0.3), `iou_threshold=0.45` (upstream default). Scores are objectness x class probability from the detector; they
  are not calibrated probabilities of a real vehicle being present.

## Measured performance (this hardware, not a general claim)
Input: the 1280x720 fixture frame (canvas 640x384), batch 1, 2 warm-up + 10 timed runs, percentiles by
`numpy.percentile` (linear). Host: Intel Core i5-13420H laptop, Docker Desktop WSL2 VM (12 vCPU, 7.6 GiB visible),
shared with concurrent image builds during the runs. Evidence JSON: `artifacts/qa/spike/model_smoke_<device>.json`
(local, gitignored; release evidence goes to `artifacts/qa/<commit>/`).

| Path | Forward p50 / p95 | Engine total p50 / p95 (pre + forward + decode/NMS + mask inversion) | Service `timing_ms.total` p50 / p95 (incl. JPEG decode, PNG masks) | Cold start (verify + jit.load + 2 warm-ups) | Peak memory |
|---|---|---|---|---|---|
| CPU fp32, torch 2.14.1+cpu, 6 threads | 442.0 / 484.6 ms | 453.0 / 495.3 ms | 496.9 / 759.8 ms | 3.2 s | not measured |
| CUDA fp16, RTX 3050 6GB Laptop, driver 576.52, torch 2.14.1+cu126 | 20.5 / 20.9 ms | 25.6 / 28.6 ms | 45.3 / 48.3 ms | 4.4 s | `max_memory_allocated` 138.2 MiB, reserved 224.0 MiB |

An earlier CPU run the same day measured forward p50 356.6 ms / p95 404.9 ms; CPU latency varies with host load.
At the measured CPU rate the service analyses roughly 2-3 frames per second, so offline video analysis on CPU is
slower than real time at 5 analysed fps. The upstream V100 FPS figures do not apply to this hardware. CUDA fp32 was
not benchmarked. Cold-start "verify" (1.5-1.8 s here) includes reading and hashing 156 MB from a Windows bind mount.

Example-frame outputs (sanity only, no labels): 23 detections, all class 3; drivable fraction 0.226; lane fraction
0.035 (CPU and GPU agree to 4 decimal places).

### CUDA fp16 vs CPU fp32 agreement
On 3 frames (example, horizontally flipped, 50 % darkened): all 74 CPU detections matched a GPU detection at
IoU >= 0.5 (mean matched IoU 0.997-0.999), drivable pixel agreement >= 0.99996 and lane pixel agreement >= 0.99996.
This shows the fp16 GPU path is numerically consistent with fp32 CPU; it is not an accuracy measurement. The GPU
image (`dreamdrive-inference-gpu`, ~11 GB because the cu126 wheels bundle the CUDA libraries) is supported only on
linux/amd64 hosts with an NVIDIA driver supporting CUDA >= 12.6 and the NVIDIA container runtime.

## Known limitations and domain
- Training domain (per upstream): BDD100K dashcam video, US roads, day and night, 1280x720 forward-facing views.
  Performance on other countries' roads, other camera placements, heavy weather, fisheye or vertical video is unknown.
- Not validated on LiteRoad synthetic renders. The simulator's default perception is `synthetic_rgb_cv`; using
  YOLOPv2 in simulation stays gated until measured scenario results exist.
- Orientation sensitive: the fixture frame rotated by 90 degrees (portrait) produced 1 detection and **zero** drivable
  and lane pixels on both CPU and GPU, while a 4:3 crop and a 640x360 downscale behaved like the original
  (29 / 23 detections, drivable ~0.23-0.26). Callers must send upright frames.
- One vehicle class only (see Classes). Missing detections are not evidence that a road is clear.
- Masks are per-pixel thresholded probabilities resized from a 640-wide canvas; thin lane markings far from the
  camera may be lost, and lane masks are not lane instances or lane geometry.
- `torch.jit.load` emits a deprecation `FutureWarning` in torch 2.14 (suppressed at the single load call). The upstream
  checkpoint exists only as TorchScript, so a future torch release that removes TorchScript loading would require a
  re-export and re-verification.
- No ARM image has been built or tested; only linux/amd64 is claimed.

## Service behaviour (inference/app)
- Startup: manifest -> device/precision check -> read the allowlisted `MODEL_ROOT/<file_name>` once while hashing ->
  `torch.jit.load` from those exact bytes -> warm-up with output-structure validation. Any failure leaves
  `ready=false` with reason `manifest_invalid | model_missing | hash_mismatch | load_failed | device_unavailable`
  (`loading` while in progress); `/v1/infer` then answers 503 `model_not_ready`. There is no mock or fallback output.
- fp16 is accepted only with `INFERENCE_DEVICE=cuda`; `fp16` on CPU is refused (`device_unavailable`).
- One forward pass at a time (single lock); more than `INFERENCE_MAX_WAITERS` (default 4) queued requests, or a wait
  longer than `INFERENCE_LOCK_TIMEOUT_S` (default 30 s), returns 503 `inference_busy`.
- Every result carries the verified model identity (`model_id`, `sha256`, `device`, `precision`) so downstream
  perception records can be tagged `learned_model` with the exact checkpoint.

## Reproduce
```bash
docker build -f inference/Dockerfile --target test -t dreamdrive-inference:test .
# Git Bash: prefix with MSYS_NO_PATHCONV=1
docker run --rm -v <models dir with yolopv2.pt>:/models:ro -e DD_EVIDENCE_DIR=/evidence -v <out dir>:/evidence \
  dreamdrive-inference:test pytest /app/tests/model_smoke -q
docker build -f inference/Dockerfile.gpu --target test -t dreamdrive-inference-gpu:test .
docker run --rm --gpus all -e INFERENCE_DEVICE=cuda -e INFERENCE_PRECISION=fp16 -v <models dir>:/models:ro \
  -e DD_EVIDENCE_DIR=/evidence -v <out dir>:/evidence dreamdrive-inference-gpu:test pytest /app/tests/model_smoke -q
```
Without weights the model smoke fails (it never passes by skipping).
