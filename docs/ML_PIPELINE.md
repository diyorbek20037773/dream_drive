# Learned video pipeline

## Gate 0 checkpoint and compatibility
The official YOLOPv2 demo uses `torch.jit.load` and custom prediction decoding with drivable/lane output helpers. Follow the upstream graph and preprocessing; it is not an Ultralytics model. Pin the source commit, preserve license notices and record changes to vendored helper code. Confirm runtime Python/Torch/NumPy compatibility empirically in the dedicated image.

The upstream README at observed source commit `9388c6aa55f3bf5a878679a1338bb090cff6a256` points to `https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt`. This source link is included in the candidate manifest. Check asset availability, download and hash the actual bytes; this kit does not fabricate a checkpoint checksum or multiclass list. Resolve `config/models.yaml` into `config/models.resolved.yaml` after obtaining bytes. Save repository revision, checkpoint SHA256/size/source, graph outputs, input shape policy, supported classes, code/weights license evidence and runtime locks. Download model assets through a timeout/checksum-verifying bootstrap, not opportunistically inside a user request. Only allowlisted verified files are loaded. Keep weights out of git; cache in a persistent volume or a legally redistributable versioned image layer. A successful download alone is not proof of inference compatibility.

On RTX 4050 6 GB start batch=1, one process and bounded input. Compare CPU FP32 and CUDA FP16 outputs/latency; FP16 only where the actual graph supports it. Use documented input geometry rather than arbitrarily forcing a size the graph cannot decode. Measure peak VRAM under model inference separately from browser/GPU display usage. Never advertise the upstream V100 FPS as this laptop's FPS.

## Frame pipeline
1. Probe supported stream, duration, dimensions, rotations and timestamps.
2. Decode frames with actual presentation timestamps. Handle variable frame rate; `frame_index / advertised_fps` is not always the original time.
3. Letterbox/RGB normalize with metadata sufficient to invert transforms. Validate channel ordering and tensor shapes.
4. Run the true model, decode heads/NMS with a pinned implementation, unpad/resize masks using correct interpolation and recover boxes in original pixels.
5. Feed detections into an audited ByteTrack adapter with one tracker per run. Avoid filtering away all low-score boxes before the algorithm's second association stage. Preserve source timestamp gaps when sampling frames.
6. Construct world state and uncertainty/status signals. Do not turn an opaque detector score into a calibrated safety confidence.
7. Produce deterministic analytical decision reason codes. Uploaded video speed/depth may be unknown; no physical braking action is performed.
8. Encode annotated browser-playable MP4 (supported codec, faststart, valid duration) plus timeline JSON/report. Report audio preserved/removed explicitly.

Optional analyzed-frame rate is configurable; annotation and decisions include actual timestamps and inference coverage. Do not overlay stale detections on unanalyzed frames without a visibly documented hold/interpolation policy. Metrics distinguish decoding/queue/inference/postprocessing/encoding/total wall time, warm vs cold start and effective analyzed FPS.

## Model alternatives
RF-DETR Nano is a conditional extra detector if verified YOLOPv2 classes omit a required category such as person. Core Nano is Apache-designated in the official repository; verify exact weight license/revision. Do not replace lane/drivable segmentation with a generic detector. Running both models is optional and must pass VRAM/latency benchmarks. If YOLOPv2 cannot be made compatible after a documented short attempt, use a tested equivalent panoptic model only with an explicit decision record and actual lane/drivable outputs; don't silently substitute fake masks. A bounded classical lane fallback is labeled separately and does not inherit learned-model accuracy claims.

Depth Anything/VLM/LLM are P2 only. Relative monocular depth does not justify metric distance or TTC. Image expansion may provide an approximate image-space collision cue; report its assumptions/invalid states. MVP may set `ttc_seconds=null`. Explanations use state/reason templates; no model hallucination may override safety.

## Evaluation
Provide a manifest of authorized test clips and a small manually reviewed labeled frame subset covering supported classes and segmentation where possible. Separate train/validation/test by clip/session, not random adjacent frames. Precision/recall/mAP/IoU/IDF1 require corresponding labels; emit null plus explanation/sample count if unavailable. Synthetic metrics and learned real-video metrics belong in separate sections. Include at least one owner/newly uploaded clip not used to tune rules. Missing domain coverage is reported, not guessed.
