# YOLOPv2 lane-marking accuracy on a Roboflow Universe dataset

Status: measured once on 2026-10-10 (evaluation `9e313020dd6c4f4eb89431e1a6c02840`, 150 images, CPU). A pixel-level, cross-dataset measurement on one small Indian-road dataset; not a benchmark score and not comparable with TuSimple/CULane leaderboards.

This document records (1) the licence/provenance audit of Roboflow Universe lane datasets, (2) the measurement
protocol, (3) the exact numbers of the real run, (4) the visual check, (5) limitations and (6) how to reproduce it.
Everything below was produced on 2026-10-10 with the tools in this repository; nothing is copied from a leaderboard.

## 1. Audit (Task A)

### Bar (not lowered)

A dataset is accepted only if **all** hold:

1. lane **markings** (painted lines), not lane areas, drivable areas or curbs;
2. labelled as polygons / masks / polylines (bounding boxes are not enough for a pixel metric);
3. at least 100 labelled images and a `test` or `valid` split;
4. licence string from the Roboflow API is CC BY 4.0, CC0, Public Domain or MIT;
5. no evidence that the images are a re-upload of BDD100K, TuSimple, CULane, CurveLanes, LLAMAS, ApolloScape or
   KITTI, and no evidence of third-party footage whose rights are unclear (checked: descriptions, resolutions such as
   TuSimple 1280x720 / CULane 1640x590 / CurveLanes 2560x1440, file names, sample images viewed by eye).

### Method

`tools/roboflow/rf_universe.py` (read-only, GET requests only; API key loaded inside Python from the owner's
`roboflow.login()` store and scrubbed from every output; signed export links are used once in memory and never
stored):

* `search`: 11 queries (`lane line segmentation`, `lane marking segmentation`, `road lane lines`, `lane detection`,
  `road marking segmentation`, `lane markings`, `lane segmentation`, `lane lines`, `road lines segmentation`,
  `lane line`, `tobias price lane`), 12 results per page, 1-3 pages -> 191 unique projects (112 of segmentation type; most of them are road, bike-lane, pothole or unrelated segmentation).
* `info`: project + all version metadata (licence, classes with instance counts, splits per version,
  preprocessing, augmentation, generated exports) for the 29 segmentation projects whose name or classes suggested lane lines or lane markings.
* Sample image check: the project's public cover image (`icon.original`) was downloaded to a scratch directory for 13
  projects and viewed; the public annotation thumbnails of 5 projects were viewed to see what the polygons cover.

All object-detection projects (bounding boxes only) fail criterion 2 and are not listed individually (e.g.
`vicentia-ck8b8/road-marking-detection-qj8ey`, `markdamages/road-marking-zvq3d`, `liu-ruiling/lane-line-o28wc`,
`harish-ramachandran-frcqg/road-lane-qmcnq`, `kroad/lane-to-line-kroad`; `projectjyo/road-lane-zs4cd` even names its
class `CULanes - v1`, i.e. CULane-derived).

### Segmentation candidates

URL pattern: `https://universe.roboflow.com/<workspace>/<project>`. "Imgs" is the project image count from the API;
splits are for the version named in the "Ver" column.

| # | Workspace/project | Ver | Licence (API) | Imgs | Type / classes (instances) | Splits (ver) | Preproc. | Image origin evidence | Decision |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `rtqi/lane-marking-9qzw7` | 2 | CC BY 4.0 | 1937 | inst-seg polygons; LC 1829, LD 1979, MD 4993, RC 2066, RD 1660 (left/middle/right, continuous/dashed markings) | train 1356 / valid 388 / test 193 | stretch 640x640, no augmentation | export file names are frames of the author's own videos (`GH030169_MP4-*`, `GH030156_MP4-*`, `GX010104_MP4-*`, `GH020156_MP4-*`, `GH010126_MP4-*` = GoPro HERO naming; `nextsync_MP4-*`, `Video_mp4-*`, `My-Project_mp4-*`, `Untitled-video-Made-with-Clipchamp_mp4-*`); 4 test images viewed: vehicle-mounted camera on Indian highways/streets (autorickshaws, yellow-black kerbs), one with a PowerDirector editing-app watermark; nothing matches TuSimple/CULane/BDD100K/CurveLanes/LLAMAS/ApolloScape/KITTI naming, resolution or content | **ACCEPTED** (selected) |
| 2 | `projects-4xr9g/road-marking-segmentation-jhhrv` | 5/8 | CC BY 4.0 | 1600 | inst-seg; SWL 3334, WL 1216, YL 377 (lane markings, thumbnail shows thin line polygons) | v5: 1120 / 320 / test 160 | stretch 640x640 | cover 848x478 ego dashcam, multi-lane urban road at dusk; not a listed benchmark size | passes the bar; runner-up, not evaluated (time; download aborted) |
| 3 | `road-damage-and-lane-detection/lane-detection-f9bja` | 10 | Public Domain | 4876 | inst-seg; lane classes BROKEN WHITE LINES 1711, SOLID WHITE LINE 665, MIXED DOUBLE LINES 367 mixed with POTHOLE 7892, VEHICLE 4149, HUMP, ARROWS | 3901 / valid 975 / - | stretch 640x640 | cover 1280x720 | not selected: lane lines are a minority task in a road-damage set; completeness of line labelling not verified |
| 4 | `tobias-price-lane-detection-solid-and-dashed/lane-detection-segmentation-edyqp` | 7 | CC BY 4.0 | 1610 | inst-seg; only Left/Right Boundary (Dashed/Solid) of the ego lane | 1141 / 318 / test 151 | static crop + contrast + stretch 640x360 | cover 1280x720 dashcam with "70mai" overlay and "VIVAVIDEO" app watermark (edited video, origin undocumented) | **REJECTED**: only the two ego-lane boundaries are labelled (other visible markings are unlabelled, so pixel precision is meaningless); pixels altered by contrast preprocessing |
| 5 | `demarcation-road-lane-line/lane-line-segmentation` | 4 | CC BY 4.0 | 3591 | inst-seg; lane-lines 19786 (long continuous polygons drawn through dashed gaps) | 3090 / valid 497 / - | none | cover 704x480 fixed CCTV frame with "aimetis" watermark and caption "Andrew Hall Entrance ..." | **REJECTED**: third-party surveillance-camera footage, rights of the camera operator undocumented (unclear licence); also not a vehicle viewpoint |
| 6 | `luiz3/lane-segmentation-yarlk` | 1 | Public Domain | 6323 | inst-seg; lane_marking 35147 | 7544 / 844 / 207 | stretch 312x312 + augmentation | cover 2560x1440 night urban dashcam, Chinese city (CurveLanes resolution and style) | **REJECTED**: suspected CurveLanes re-upload |
| 7 | `yoloptraining/semantic-segmentation-lane-lines-vbt3n` | 5 | CC BY 4.0 | 5454 | semantic; lane-lines | train only | none | cover 1280x720 (TuSimple size); name suggests YOLOP training data | **REJECTED**: no test/valid split; possible TuSimple/BDD re-upload |
| 8 | `curbseg/lane-line-seg` | 9 | CC BY 4.0 | 543 | inst-seg; lane 1212 | 1371 / 82 / 4 | none | cover 3840x2160 Singapore dashcam | **REJECTED**: annotation thumbnail shows large lane-AREA polygons, not markings (the earlier, incomplete download of this project was deleted) |
| 9 | `dku-jforg/lane-detection-rvovr` | 1 | CC BY 4.0 | 3795 | inst-seg; Line 4912, Dotted 3469, crosswalk 244 | 12755 / 606 / - | stretch 640x360 + aug | cover is an indoor scale-model track | **REJECTED**: not a real road |
| 10 | `capstoneteam/road-lines-segmentation-phoc6` | 2 | CC BY 4.0 | 9970 | inst-seg; solid-yellow-line 7540, car, License_Plate | 24210 / 1400 / 500 | stretch 640x640 + contrast | not checked | **REJECTED**: only one marking type (solid yellow) labelled |
| 11 | `manishme19acc/road-edge-lines` | 8 | CC BY 4.0 | 6743 | inst-seg; lines 11225 | 6069 / 674 / - | stretch 432x432 | cover 1920x1080 French motorway dashcam | not selected: name says road-EDGE lines; label scope not verifiable (no annotation thumbnail) |
| 12 | `zero-kpiaq/lane-segmentation-rpxtt` | 5 | CC BY 4.0 | 1111 | semantic; lane_1 1013, lane_2 1111 | 953 / 90 / 68 | stretch 224x224 | cover 448x300 | **REJECTED**: per-lane area classes (one instance per image), 224 px |
| 13 | `burnemcalories/texas-lane-markings-seg` | 1 | CC BY 4.0 | 279 | inst-seg; lane_marking 2126, edge_line 641 | 195 / 56 / 28 | stretch 640x640 | not checked | not selected: evaluable test/valid splits are 28/56 images (< 100) |
| 14 | `ninas-thesis/lane-segmentation-hajji` | 5 | CC BY 4.0 | 864 | inst-seg; lane 2431 (Line, YOLCIZGISI 0) | 604 / 174 / 86 | fit 640x400 | cover 640x400 | not selected: class semantics (area vs line) not verified |
| 15 | `segmentation-tsxwy/lane-segmentation-wnojc` | 1 | CC BY 4.0 | 853 | inst-seg; Lane 981 | 1791 / 171 / 85 | stretch 240x320 (portrait) + aug | not checked | not selected: portrait 240x320 stretch; semantics not verified |
| 16 | `emines-workspace-fmozu/lane-f7nam` | 2 | CC BY 4.0 | 1879 | inst-seg; lane 4399, pothole 309 | 3965 / 311 / 138 | stretch 432x432 | not checked | not selected: semantics not verified |
| 17 | `trungducks-workspace/lane-segmentation-k3v8t` | 2 | CC BY 4.0 | 2972 | inst-seg; Lane 484 mixed with arrows, cars, STOP | 6240 / 594 / 298 | stretch 640x640 | not checked | not selected: lane is a minority class, semantics not verified |
| 18 | `zineb-haimer-1zukf/road-marking-detection` | 3 | CC BY 4.0 | 2885 | inst-seg; arrows, pedestrian crossings, symbols | - | stretch 640x640 | not checked | **REJECTED**: road-marking symbols, not lane lines |
| 19 | `velotech-road-sign/road-marking-vxdcw` | 3 | Public Domain | 4335 | inst-seg; ligne 1765 with parking/crossing/arrows | 3035 / 867 / 433 | stretch 640x640 | not checked | not selected: lines are a minority class, semantics not verified |
| 20 | `computer-vision-controls-research-group/lane-detection-tjaa0` | 1 | CC BY 4.0 | 135 | inst-seg; Lane Line 339, Lanes 135 | 96 / 26 / 13 | stretch 640x640 | not checked | not selected: test/valid 13/26 images |
| 21 | `lane-marking-aasnt/lane-marking-bosdu` | 4 | CC BY 4.0 | 240 | inst-seg; LC/LD/MD/RC/RD | 168 / 24 / 48 | stretch 640x640 | not checked | not selected: test/valid 48/24 images |
| 22 | `jk-nanu0/lane-marking-u5e9r` | 2 | CC BY 4.0 | 347 | inst-seg; '1' 569 | 354 / 34 / 17 | grayscale + contrast + stretch | not checked | **REJECTED**: grayscale/contrast preprocessing alters pixels; tiny splits |
| 23 | `hahahahaha/lane-detection-qu2ab` | 3 | CC BY 4.0 | 464 | semantic; lane (0 instances reported) | 966 / 93 / 46 | stretch 640x640 | cover 1280x720 (TuSimple size) | **REJECTED**: class count 0 / unclear; possible TuSimple |
| 24 | `gabriel-yw7q6/lane-line-detection-h6auu` | 1 | CC BY 4.0 | 402 | inst-seg; line + arrows | train only | stretch 512 | not checked | **REJECTED**: no test/valid split |
| 25 | `internship-tvkfo/road-lines-segmentation` | 1 | CC BY 4.0 | 200 | inst-seg; Road-Line 525 with Road, Footpath | 480 / 20 / 20 | stretch 640x640 + aug | not checked | not selected: test/valid 20/20 images |
| 26 | `dugheana/road-lane-gyn8x` | 2 | CC BY 4.0 | 742 | inst-seg; lane 703, parking 101 | 1671 / 120 / 65 | stretch 640x640 + aug | not checked | not selected: semantics not verified |
| 27 | `project-teqgp/road-marking-bi06i` | 3 | CC BY 4.0 | 300 | inst-seg; roadmarking 1049 | 630 / 60 / 30 | stretch 640x640 | not checked | not selected: generic road-marking class, small splits |
| 28 | `md-safirur-rashid/otowa-lane-segmentation` | - | CC BY 4.0 | 3968 | inst-seg; '0' 7439 | no version | - | not checked | **REJECTED**: no generated version, nothing exportable |
| 29 | `rc-whs1r/lane-marking-nkjg4` | - | Public Domain (search) | 2764 | inst-seg | - | - | - | **REJECTED**: project API returns HTTP 404 |

"not selected" = not disqualified by a hard rule but weaker than the selected dataset or not verifiable in the time
available; only row 1 was evaluated.

### Decision

**Accepted and evaluated: `rtqi/lane-marking-9qzw7` version 2** (https://universe.roboflow.com/rtqi/lane-marking-9qzw7/dataset/2),
licence string from the API and from the export's `README.dataset.txt`: "CC BY 4.0" ("Provided by a Roboflow user").
It meets every hard criterion: painted lane markings (left/middle/right, continuous/dashed) as instance-segmentation
polygons including individual dashes and far-field lines; 1937 labelled images; a 193-image `test` split (the
export contains 194 files in `test/`: 193 images + `_annotations.coco.json`); CC BY 4.0; frame names and content of
the author's own GoPro/phone videos, not a re-upload of a listed benchmark.

Caveats that remain (recorded as limitations, not hidden): the author applied "Resize to 640x640 (Stretch)" before
export, so every frame is non-uniformly stretched from its camera aspect ratio; the capture provenance (who filmed,
where exactly) is not documented beyond the file names; the export served by Roboflow was generated on
2026-07-30 (README: "exported via roboflow.com on July 30, 2026"), downloaded 2026-10-10, zip SHA256
`8d56f3cfdcaa47b588ef9d238cb43412f828e8ec3073ec335c3c2abb5c2fa310`, 69,153,469 bytes (zip deleted after extracting
the test split).

Runner-up for a second measurement: `projects-4xr9g/road-marking-segmentation-jhhrv` v5 (CC BY 4.0, 160 test
images, also stretched to 640x640).

## 2. Method (Task B)

Code: `backend/app/evaluation/lane_accuracy.py` (follows the KITTI `segmentation_accuracy.py` conventions; unit
tests in `tests/unit/backend/test_lane_accuracy.py`).

1. **Data.** The Roboflow export API only delivers whole versions, so the `coco-segmentation` zip of the selected
   version was downloaded transiently, only the `test` split was extracted to
   `datasets/raw/roboflow/lane-marking-9qzw7-v2/test/` (gitignored), and the zip (with the train/valid splits) was
   deleted. The committed manifest `datasets/manifests/roboflow_lane-marking-9qzw7_v2.json` holds the URL, version,
   licence string, attribution, format, preprocessing, split, counts and per-file SHA256 (no key, no signed URL, no
   images). The run verifies every file hash against the manifest.
2. **Ground truth.** Union of all annotation polygons of an image (all five classes are lane markings), rasterised
   at the original exported resolution with `cv2.fillPoly` (vertices shifted by -0.5 px to map COCO continuous
   coordinates to pixel centres, 4 fractional bits). Polygon boundaries are included, so thin polygons are not lost.
3. **Prediction.** Each original exported image is decoded to RGB and re-encoded as JPEG q95 (as the video pipeline
   does), sent through the production `app.perception.client.InferenceClient.infer` to a locally started
   `dreamdrive-inference:dev` container (CPU, YOLOPv2 + RF-DETR-N ensemble), and `lane_mask_png` (YOLOPv2 lane-line
   probability > 0.5, cropped/unletterboxed by the service to the original H x W) is decoded; a missing, undecodable or
   wrongly sized mask fails the image (no silent skips; > 5 % failures aborts the run).
4. **Metrics.** Pixel TP/FP/FN pooled over all images (dataset level) and per image (p10/p50/p90):
   * exact: binary prediction vs GT pixel for pixel;
   * tolerant: both masks dilated with a disk of radius `k = floor(8 * W / 1280 + 0.5)` px before counting
     (k = 4 px for the 640 px wide images here). This forgives small lateral offsets and the different line widths
     of YOLOPv2 output and hand-drawn polygons, but does not reward a missing or invented line far from any GT line;
   * buffered precision/recall: predicted pixels within k px of a GT pixel / predicted pixels, and GT pixels within
     k px of a predicted pixel / GT pixels (boundary-F-score style).
   Per-image values with a zero denominator are excluded and counted. Latency: service `timing_ms.total`,
   YOLOPv2 forward time and client round trip.
5. **Visual evidence.** Per-image CSV and best/median/worst comparisons by per-image tolerant IoU.

## 3. Results

Evaluation JSON: `artifacts/evaluations/9e313020dd6c4f4eb89431e1a6c02840.json` (generated, gitignored; validates as
`app.schemas.api.Evaluation`, kind `video_ml_accuracy`, source `learned_model`, listed by `app.evaluation.registry`).
Per-image CSV and visuals: `artifacts/qa/spike/lanes/9e313020dd6c4f4eb89431e1a6c02840.*`.

Run identity

| Field | Value |
|---|---|
| Dataset | `rtqi/lane-marking-9qzw7` v2, split `test`, first 150 of 193 images by sorted file name (cap recorded in the JSON) |
| Manifest | `datasets/manifests/roboflow_lane-marking-9qzw7_v2.json`, SHA256 `d64426dd9a3da884c6b77a7d7c0687f69b2e1d4de87ec869ed3e87c4ec918f4a` (rebuilt byte-identical) |
| `_annotations.coco.json` (test) SHA256 | `a0d7b6a3c3247420b7758944ae06f6bf0ff33cf60e8a3215b339da0e59ac8118` |
| Images evaluated / failed | 150 / 0 (all 640x640, letterbox canvas 640x640, scale 1.0, no padding; k = 4 px for all 150) |
| YOLOPv2 SHA256 (`model_sha256`) | `f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c` |
| RF-DETR-N SHA256 (ran in the ensemble, not scored) | `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d` |
| Ensemble SHA256 reported | `e514a67df0620326cb5a57c6a3a3db79d789dae42f43a96b71e7268867d34777` |
| Service | `dreamdrive-inference:dev` (local image 64ccb4436d9a), CPU fp32, torch 2.14.1+cpu, input policy `letterbox640_stride32` |
| Code | HEAD `fa64cd00994dd5215e1a992fdd236765f69ed54b`, working tree dirty (this evaluator not yet committed) |
| Host / time | Windows 11, 12 logical CPUs shared with a concurrently running compose stack; 2026-10-10 16:11:34-16:14:17 UTC |

Dataset-level (pooled over 150 images; 723,819 GT lane pixels, 1,315,707 predicted lane pixels)

| Protocol | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| exact pixels | 0.2948 | 0.3529 | 0.6415 | 0.4553 |
| tolerant (both masks dilated, k = 4 px) | 0.4440 | 0.5076 | 0.7801 | 0.6150 |
| buffered (within k = 4 px of the other mask) | - | 0.6250 | 0.7940 | 0.6994 |

Exact counts: TP 464,308, FP 851,399, FN 259,511. Tolerant counts: TP 1,209,186, FP 1,173,159, FN 340,827.

Per image (150 images, none undefined)

| Series | p10 | p50 | p90 |
|---|---|---|---|
| exact IoU | 0.1488 | 0.3055 | 0.4316 |
| exact F1 | 0.2590 | 0.4680 | 0.6029 |
| tolerant IoU | 0.2516 | 0.4720 | 0.5960 |
| tolerant F1 | 0.4020 | 0.6413 | 0.7468 |

One image (`My-Project_mp4-0032...`) had no predicted lane pixel (IoU 0).

Latency (150 samples, CPU shared with another stack, both models of the ensemble): service total p50 990.8 ms /
p95 1223.2 ms; YOLOPv2 forward p50 753.5 ms / p95 949.3 ms; client round trip p50 995.3 ms / p95 1230.2 ms.

`lane_instance_f1` is recorded as `unsupported` (YOLOPv2 has no lane instances).

Reading the numbers: recall is clearly higher than precision. The visual check shows why: many "false positive"
pixels are YOLOPv2 drawing dashed lines as continuous strokes and slightly wider or offset relative to the
hand-drawn dash polygons, a convention difference rather than invented lines; the buffered precision (0.625) is the
share of predicted pixels within 4 px of some labelled marking.

## 4. Visual check

The three comparisons (input / GT polygons / prediction with TP green, FP red, FN blue) were opened and inspected:

* **best** (`GH020156_MP4-0008`, exact IoU 0.477, tolerant 0.680): two-lane highway with trucks ahead. YOLOPv2 finds
  the left edge line, the dashed centre line and the right edge line. Red FP pixels are mainly the centre line drawn
  continuously through the dash gaps and a slightly wider/offset right edge line; a thin blue FN sliver remains along
  the right kerb line.
* **median** (`GH030156_MP4-0021`, exact IoU 0.246, tolerant 0.472): open highway. The left edge line is a TP; the
  dashed centre line is predicted as one continuous red stroke while the GT only has the painted dashes, and the
  right edge line is predicted a few pixels off its polygon (red/blue side by side), which the tolerant metric
  partly forgives.
* **worst** (`My-Project_mp4-0032`, IoU 0.0): shaded rural road without clear painted lines and a PowerDirector
  watermark; the GT contains a single small faint line polygon at the bottom right and YOLOPv2 predicts no lane
  pixel at all (FN only).

No systematic misalignment between mask and image was visible (the 640x640 input needs no letterbox padding).

## 5. Limitations

* **Stretched inputs.** The dataset author applied "Resize to 640x640 (Stretch)" before export, so camera frames
  (GoPro/phone, most likely 16:9) are vertically stretched by about 1.78x. YOLOPv2 was trained on undistorted 16:9
  BDD100K frames; the numbers describe performance on these distorted frames, not on the original camera frames
  (which are not published).
* **Label convention mismatch.** GT polygons cover the painted dashes only and vary in width; YOLOPv2 (BDD100K lane
  annotations) draws thin continuous strokes, including through dash gaps. This inflates exact FP and lowers exact
  IoU independently of model quality; the tolerant/buffered numbers reduce but do not remove the effect.
* **Pixel metrics only.** No lane instances, no type (solid/dashed), no ego-lane assignment, no threshold sweep
  (fixed operating point: lane probability > 0.5).
* **Small, narrow sample.** 150 of 193 test images (cap for shared CPU, first 150 by sorted file name), from nine
  video-name families of one author on Indian highways and streets in daylight. Consecutive frames of the same video
  are correlated, so the effective sample size is smaller than 150. Nothing about night, rain, other countries or
  other cameras.
* **Provenance.** The licence is the author's declaration on Roboflow Universe (CC BY 4.0); the author does not
  document where or by whom the videos were filmed. Some frames carry consumer video-editor marks (PowerDirector
  watermark, Clipchamp file names), which indicates edited personal recordings but proves nothing further.
* **Compression.** Exported JPEGs are re-encoded once as JPEG q95 before inference (same as the video pipeline).
* **Shared CPU.** Latencies were measured while another compose stack was running and include RF-DETR-N; they are not
  a CPU latency benchmark.
* **Not evaluated:** the remaining 43 test images, the valid split (388 images), and the runner-up dataset.

## 6. Reproduce

Prerequisites: Roboflow login stored by the official SDK (never type a key on a command line or into a file):

```bash
py -3.11 -c "import roboflow; roboflow.login()"     # interactive, stores %USERPROFILE%/roboflow/config.json
```

Audit metadata (read-only):

```bash
py -3.11 -I tools/roboflow/rf_universe.py search --out <scratch>/search.json "lane line segmentation" "lane markings"
py -3.11 -I tools/roboflow/rf_universe.py info --out <scratch>/info.json rtqi/lane-marking-9qzw7
```

Data (downloads the whole version zip; keep only the test split):

```bash
py -3.11 -I tools/roboflow/rf_universe.py export rtqi/lane-marking-9qzw7/2 --format coco-segmentation \
    --dest datasets/raw/roboflow/_downloads/rtqi
py -3.11 -I -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); [z.extract(n, sys.argv[2]) for n in z.namelist() if n.startswith('test/')]" \
    datasets/raw/roboflow/_downloads/rtqi/rtqi__lane-marking-9qzw7__2__coco-segmentation.zip \
    datasets/raw/roboflow/lane-marking-9qzw7-v2
rm -r datasets/raw/roboflow/_downloads/rtqi
```

Manifest (deterministic; must reproduce the committed file byte for byte):

```bash
cd backend
.venv/Scripts/python -m app.evaluation.lane_accuracy build-manifest --dataset rtqi-lane-marking-9qzw7-v2 \
    --split test --split-dir ../datasets/raw/roboflow/lane-marking-9qzw7-v2/test \
    --out ../datasets/manifests/roboflow_lane-marking-9qzw7_v2.json
```

Inference service and run:

```bash
MSYS_NO_PATHCONV=1 docker run -d --rm --name dd-lanes -p 127.0.0.1:18051:8001 \
    -v D:/myprojects/west_hack_dream_drive/models/yolopv2/yolopv2.pt:/models/yolopv2.pt:ro \
    -v D:/myprojects/west_hack_dream_drive/models/rfdetr_nano/rf-detr-nano.pth:/models/rf-detr-nano.pth:ro \
    dreamdrive-inference:dev
curl -s http://127.0.0.1:18051/health/ready        # wait for "ready":true (RF-DETR construction ~30-90 s)
cd backend
.venv/Scripts/python -m app.evaluation.lane_accuracy run \
    --split-dir ../datasets/raw/roboflow/lane-marking-9qzw7-v2/test \
    --manifest ../datasets/manifests/roboflow_lane-marking-9qzw7_v2.json \
    --inference-url http://127.0.0.1:18051 --out-dir ../artifacts/evaluations \
    --evidence-dir ../artifacts/qa/spike/lanes --max-images 150 --timeout-s 60
docker stop dd-lanes
.venv/Scripts/python -m pytest ../tests/unit/backend/test_lane_accuracy.py -q
```

## 7. Proposed DATA_SOURCES entry

```yaml
- id: roboflow_rtqi_lane_marking_v2
  name: "Lane-Marking (RTQI), Roboflow Universe, version 2"
  url: https://universe.roboflow.com/rtqi/lane-marking-9qzw7/dataset/2
  licence: "CC BY 4.0"   # string from the Roboflow API and from the export README.dataset.txt
  licence_url: https://creativecommons.org/licenses/by/4.0/
  attribution: "RTQI, 'Lane-Marking' dataset (version 2), Roboflow Universe, https://universe.roboflow.com/rtqi/lane-marking-9qzw7, licensed CC BY 4.0."
  use: evaluation only (YOLOPv2 lane-marking pixel accuracy); not used for training; not redistributed
  format: coco-segmentation (instance polygons; classes LC, LD, MD, RC, RD)
  split_used: test (193 images; first 150 by sorted file name evaluated on 2026-10-10)
  preprocessing_by_author: auto-orient, stretch to 640x640; no augmentation
  obtained: Roboflow REST export via tools/roboflow/rf_universe.py (owner's roboflow.login(); key never stored); zip SHA256 8d56f3cfdcaa47b588ef9d238cb43412f828e8ec3073ec335c3c2abb5c2fa310, 69153469 bytes (deleted after extracting test/)
  local_path: datasets/raw/roboflow/lane-marking-9qzw7-v2/test/   # gitignored
  manifest: datasets/manifests/roboflow_lane-marking-9qzw7_v2.json  # per-file SHA256, no images
  evaluation: artifacts/evaluations/9e313020dd6c4f4eb89431e1a6c02840.json (docs/EVALUATION_LANES.md)
  provenance_notes: frames of the author's own GoPro/phone videos (file names), Indian roads; capture details undocumented
```
