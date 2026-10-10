# Drivable-area segmentation accuracy (YOLOPv2 on KITTI Road, training split)

Status: measured once, locally, on 2026-10-10 (evidence is gitignored; see "Evidence files"). This is the first and
only measured accuracy number for YOLOPv2's drivable-area output in DreamDrive. It is a **cross-dataset, perspective-space,
fixed-threshold** measurement and is **not** a KITTI Road benchmark (leaderboard) result. Lane-line accuracy is still
not evaluated.

Code: `backend/app/evaluation/segmentation_accuracy.py`. Tests: `tests/unit/backend/test_segmentation_accuracy.py`
(synthetic masks and GT colours, no network, no KITTI data). Manifest: `datasets/manifests/kitti_road_training_v1.json`.

## 1. Data

| Item | Value |
| --- | --- |
| Dataset | KITTI Road benchmark (Fritsch, Kuehnl, Geiger, ITSC 2013), **training split** - the only split with public labels |
| Archive | `data_road.zip` from `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip` |
| Archive size / SHA256 | 470,992,343 bytes / `45508c51ba09020235eb714e71c7d3ef15b4551be06d996772ecc8a20fc34a34` |
| Local path (gitignored) | `datasets/raw/kitti_road/data_road.zip` |
| Licence | "All datasets and benchmarks on this page are copyright by us and published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License" (https://www.cvlibs.net/datasets/kitti/) - **non-commercial** |
| Attribution | J. Fritsch, T. Kuehnl, A. Geiger, "A New Performance Measure and Evaluation Benchmark for Road Detection Algorithms", ITSC 2013 |
| Images | 289 left colour images `data_road/training/image_2/<cat>_<idx>.png` (lossless PNG): um 95, umm 96, uu 98 |
| Image sizes | 1242x375 (202), 1241x376 (35), 1226x370 (49), 1238x374 (3) - aspect about 3.3:1 |
| Ground truth used | `data_road/training/gt_image_2/<cat>_road_<idx>.png` (road area = all lanes), one per image |
| Ground truth not used | `um_lane_<idx>.png` (95 files): the ego-lane **area**, not lane markings; hashed in the manifest, not scored |
| Committed manifest | `datasets/manifests/kitti_road_training_v1.json`: per image the zip member names, category, size, SHA256 of image and GT members, GT pixel counts and GT colour histogram, plus archive SHA256, licence and attribution. No images or labels are in git. |
| Devkit (read, not vendored) | `https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_road.zip`, 22,283 bytes, SHA256 `6f7cba10eeeb82d26cb0fa64ea4771ae92ff818be50098b162a21fcab8e06156` |

### Ground-truth encoding (verified, not assumed)
The zip has no devkit, so the official devkit was downloaded separately and read. `devkit_road/readme.txt` (update
February 2015): "first/R plane contains valid evaluation area ... third/B plane contains road area ground truth";
`python/helper.py getGroundTruth()`: `roadArea = full_gt[:,:,0] > 0` (OpenCV BGR, so B) and
`validArea = full_gt[:,:,2] > 0` (R); `evalExp` counts positives/negatives only inside `validArea`.

The manifest builder counted every colour in all 289 road GT PNGs (exact RGB histogram). Only four colours exist:

| RGB | Meaning (devkit rule) | Pixels (all 289) | um | umm | uu |
| --- | --- | ---: | ---: | ---: | ---: |
| `#ff00ff` | road, valid | 23,734,702 | 7,151,370 | 10,433,021 | 6,150,311 |
| `#ff0000` | non-road, valid | 108,528,852 | 36,268,476 | 33,321,084 | 38,939,292 |
| `#000000` | don't care (R = 0) | 1,765,447 | 718,957 | 876,736 | 169,754 |
| `#0000ff` | road but R = 0 -> outside valid area, ignored | 475 | 9 | 439 | 27 |

So: road = B > 0, valid = R > 0; pixels with R = 0 (black and the 475 blue pixels) are excluded from every count.

## 2. Method

1. Verify `data_road.zip` SHA256 against the manifest; verify each image and GT member's SHA256 as it is read.
2. Decode the PNG to RGB and encode it as JPEG quality 95 with `app.video.encode.encode_jpeg_bytes` - exactly what the
   video pipeline sends - then call the production client `InferenceClient.infer(include_masks=True)` against a real
   `dreamdrive-inference` container (`POST /v1/infer`). No resizing on the client side: the service letterboxes
   (longest side 640, stride-32 minimal padding, pad 114), runs YOLOPv2, crops the padding and resizes the
   segmentation probabilities back to the original frame, then `drivable = argmax(channel) == 1`
   (`config/models.resolved.yaml`).
3. Decode `drivable_mask_png` (uint8 {0,255}, original HxW) strictly: a missing, undecodable or wrongly sized mask is a
   failure, never skipped silently. Prediction = value > 127; non-{0,255} values are counted (integrity check).
4. Per image, inside the valid area only: TP = pred & road, FP = pred & non-road, FN = !pred & road, TN = the rest.
5. **Dataset-level** ratios pool pixels: IoU = sum TP / sum (TP+FP+FN); precision = sum TP / sum (TP+FP);
   recall = sum TP / sum (TP+FN); F1 = 2 sum TP / (2 sum TP + sum FP + sum FN), per category (um, umm, uu) and overall.
   F1 equals KITTI's F-measure at one fixed operating point **in perspective space**; KITTI's official MaxF maximises F
   over a confidence sweep in a bird's-eye-view grid, which the binary service mask cannot reproduce.
6. **Per-image** IoU distribution: mean, p10, p50, p90 over images with a non-empty union.
7. `sample_count` of every ratio metric = number of **images**; pixel totals are in each metric's `note` and in
   `details.totals`. Latency: service `timing_ms.total` and `timing_ms.inference_by_model.yolopv2`, plus client round
   trip, p50/p95.
8. Output: an `Evaluation` JSON (`kind=video_ml_accuracy`, `source=learned_model`, registry-compatible, listed by the
   API) in `artifacts/evaluations/`, and evidence in `artifacts/qa/spike/kitti_road/`: per-image CSV, the 289 binary
   prediction masks, and best/median/worst visual comparisons.

`model_sha256` of the Evaluation is the YOLOPv2 checkpoint, because the drivable mask is produced by YOLOPv2 alone; the
service ran the YOLOPv2 + RF-DETR-N ensemble and its `ensemble_sha256` is recorded in `details`.

## 3. Results

RESULTS_PLACEHOLDER

VISUAL_PLACEHOLDER

## 5. Limitations (also recorded in the Evaluation JSON `limitations`)

- **Cross-dataset, convention mismatch.** YOLOPv2 was trained on BDD100K. BDD100K "drivable area" covers the directly
  drivable lane plus alternative lanes; KITTI "road" is the whole road surface (all lanes). Where the conventions differ
  (opposite carriageway, parking strips, road edges, areas next to and under vehicles, lane markings) part of the error
  is label convention, not model error. No relabelling or class remapping was attempted.
- **Aspect ratio / letterbox.** KITTI frames are about 1242x375 (3.3:1). The production letterbox (longest side 640,
  stride-32 minimal padding) gives a 640x224 canvas with about 193 rows of content; BDD100K frames are 16:9 (the
  1280x720 reference gives 640x384). The model therefore sees an unusually wide, low-resolution canvas. This is the
  production input policy and was deliberately not changed for the benchmark.
- **Not the KITTI benchmark metric.** Perspective-space, fixed binary operating point. KITTI ranks MaxF over a
  confidence sweep in an 800x400 bird's-eye-view grid; perspective space weights near-field pixels much more heavily.
  Do not compare with the KITTI leaderboard. `kitti_bev_maxf` is recorded as `not_evaluated`.
- **Small, narrow domain.** 289 training images, Karlsruhe urban scenes in daylight (um/umm/uu). YOLOPv2 never trained on
  KITTI and nothing was tuned on these images, but the result says nothing about night, rain, highways, rural roads or
  other countries (for example Uzbekistan road scenes).
- **Licence.** CC BY-NC-SA 3.0, non-commercial. Images and labels stay local (`datasets/raw/`, `artifacts/`); only names
  and hashes are committed. A commercial use of DreamDrive must not rely on this dataset.
- **Lane markings not evaluated.** `um_lane` ground truth is the ego-lane area, not lane markings, and YOLOPv2 has no
  ego-lane output; `lane_marking_accuracy` stays `not_evaluated` and `ego_lane_iou` is `unsupported`. Vehicle detection
  is not evaluated here (KITTI Road has no boxes; see docs/EVALUATION.md for the COCO detector evaluation).
- **One JPEG step.** KITTI PNGs are encoded once as JPEG q95 before inference, as in the video pipeline.
- **Don't-care pixels** (R plane = 0; 1,765,922 of 134,029,476 GT pixels including the 475 blue pixels) are excluded,
  following the devkit; predictions there are neither TP nor FP.
- **Shared CPU.** Latency was measured on a CPU container sharing the Docker Desktop VM with two other inference
  containers under load; it is an upper-bound style observation for this host, not a benchmark.
- **One run, no confidence intervals.** The per-image IoU percentiles show the spread; no bootstrap intervals were
  computed.

## 6. Reproduce

All commands from the repository root in Git Bash on Windows (Docker Desktop, linux/amd64). The weights must already be
present and verified (`models/yolopv2/yolopv2.pt`, `models/rfdetr_nano/rf-detr-nano.pth`).

```bash
# 0. Data (once; 471 MB; check the SHA256 before use)
mkdir -p datasets/raw/kitti_road
curl -L -o datasets/raw/kitti_road/data_road.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip
sha256sum datasets/raw/kitti_road/data_road.zip   # 45508c51ba09020235eb714e71c7d3ef15b4551be06d996772ecc8a20fc34a34

# 1. Manifest (deterministic; must reproduce the committed file byte for byte)
cd backend
.venv/Scripts/python -m app.evaluation.segmentation_accuracy build-manifest \
  --zip ../datasets/raw/kitti_road/data_road.zip --out ../datasets/manifests/kitti_road_training_v1.json
git diff --exit-code ../datasets/manifests/kitti_road_training_v1.json
cd ..

# 2. Inference service (CPU image; both weights at the paths the image's manifest expects)
MSYS_NO_PATHCONV=1 docker run -d --rm --name dd-kitti -p 127.0.0.1:18041:8001 \
  -v D:/myprojects/west_hack_dream_drive/models/yolopv2/yolopv2.pt:/models/yolopv2.pt:ro \
  -v D:/myprojects/west_hack_dream_drive/models/rfdetr_nano/rf-detr-nano.pth:/models/rf-detr-nano.pth:ro \
  dreamdrive-inference:dev
until curl -sf http://127.0.0.1:18041/health/ready; do sleep 5; done

# 3. Evaluation (289 images)
cd backend
.venv/Scripts/python -m app.evaluation.segmentation_accuracy run \
  --zip ../datasets/raw/kitti_road/data_road.zip --manifest ../datasets/manifests/kitti_road_training_v1.json \
  --inference-url http://127.0.0.1:18041 --out-dir ../artifacts/evaluations \
  --evidence-dir ../artifacts/qa/spike/kitti_road
cd ..
docker stop dd-kitti

# Unit tests (no network, no KITTI data)
cd backend && .venv/Scripts/python -m pytest ../tests/unit/backend/test_segmentation_accuracy.py -q
```

## 7. Proposed DATA_SOURCES entry

For the lead to merge into `docs/DATA_SOURCES.md` (this file does not edit it):

> ### KITTI Road benchmark - training split (drivable-area evaluation only)
> - Source: `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip` (KITTI Vision Benchmark Suite,
>   https://www.cvlibs.net/datasets/kitti/eval_road.php). 470,992,343 bytes, SHA256
>   `45508c51ba09020235eb714e71c7d3ef15b4551be06d996772ecc8a20fc34a34`. Downloaded 2026-10-10 to
>   `datasets/raw/kitti_road/` (gitignored).
> - Licence: "All datasets and benchmarks on this page are copyright by us and published under the Creative Commons
>   Attribution-NonCommercial-ShareAlike 3.0 License" (https://www.cvlibs.net/datasets/kitti/). Non-commercial; not
>   redistributed; not used for training; used only for a local accuracy measurement.
> - Attribution: J. Fritsch, T. Kuehnl, A. Geiger, "A New Performance Measure and Evaluation Benchmark for Road
>   Detection Algorithms", ITSC 2013.
> - Used: 289 training images (`image_2`) and their `<cat>_road_<idx>.png` ground truth (um 95, umm 96, uu 98).
>   `um_lane` (ego-lane area) is hashed but not scored. Testing split and calibration files are unused.
> - Committed: `datasets/manifests/kitti_road_training_v1.json` (member names, sizes, SHA256, GT colour histograms;
>   no pixels). GT encoding taken from the official devkit (`devkit_road.zip`, SHA256
>   `6f7cba10eeeb82d26cb0fa64ea4771ae92ff818be50098b162a21fcab8e06156`): R plane = valid area, B plane = road.
> - Consumer: `backend/app/evaluation/segmentation_accuracy.py`; results and limitations in
>   `docs/EVALUATION_SEGMENTATION.md`.
