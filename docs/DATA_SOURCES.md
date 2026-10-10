# Data sources and test fixtures

Only small, licensed fixtures are used. Media files are **never committed**: `scripts/fetch_fixtures.py` downloads them
into `tests/fixtures/cache/` (git-ignored), verifies their hashes and derives short clips. The machine-readable record
is `tests/fixtures/manifest.json`; this page mirrors it for humans.

## Real-road video fixture

| Field | Value |
|---|---|
| Fixture id | `commons_dashcam_daylight_cc0` |
| Title | File:Dashcam recording example - Panorama II BlackBox Dashcam daylight.mp4.ogv |
| Page | https://commons.wikimedia.org/wiki/File:Dashcam_recording_example_-_Panorama_II_BlackBox_Dashcam_daylight.mp4.ogv (page id 29915797, revision 886093428) |
| Direct URL | https://upload.wikimedia.org/wikipedia/commons/e/eb/Dashcam_recording_example_-_Panorama_II_BlackBox_Dashcam_daylight.mp4.ogv |
| Author | Fernost (Wikimedia Commons user `User:Fernost`), "own work" |
| Date / description | 2013-11-29, "Just an example for a dashcam recording" |
| License | **CC0 1.0** (public-domain dedication), http://creativecommons.org/publicdomain/zero/1.0/deed.en |
| License evidence | Verified 2026-10-10 via the Commons API (`action=query&prop=imageinfo&iiprop=url|sha1|size|extmetadata`: `LicenseShortName=CC0`, `License=cc0`, `Credit=Own work`) and the file page wikitext (`{{self|cc-zero}}`). |
| Source file | Ogg Theora video (+ audio stream), 1280x720, 30 fps, 120.03 s, 73,237,816 bytes |
| Source SHA1 | `48b212f58374fdc6881287e4daf1532bd26b279c` (as published by the Commons API; checked on download) |
| Source SHA256 | `3b2fb1a37eb797bdbf4884ad7df9f9118d64240a648208b1fa6ef4a927ac8970` (computed from the downloaded bytes) |

Derived clip used by tests and the real-model end-to-end run:

| Field | Value |
|---|---|
| File | `tests/fixtures/cache/commons_dashcam_daylight_cc0_96s_10s.mp4` |
| Content | 10.0 s excerpt starting at 96 s (multi-lane road, vehicles ahead, painted lane lines); audio dropped |
| Encoding | PyAV 18.1.0 / bundled libx264, yuv420p, CRF 20, preset medium, single-threaded, `+faststart`, 30 fps, 300 frames, 1280x720 |
| SHA256 | `1f61002228dcf46bd810f0a0fb661573bae7ba7e36dc4254f34c6c71223f1293` (re-derived with identical bytes on the Windows host and in a `python:3.11-slim-bookworm` Linux container, both PyAV 18.1.0) |

The dash-cam footage shows public roads, other vehicles and distant pedestrians/cyclists; license plates are small and
were not legible in the frames inspected. It carries a burned-in timestamp/telemetry overlay from the recording device. The clip has **no
labels**: it is used for decode/sampling/quality-threshold tests and as the real-model end-to-end input, never as an
accuracy benchmark. Any accuracy metric remains `not_evaluated`.

Commands:

```bash
python scripts/fetch_fixtures.py          # download, verify SHA1+SHA256, derive the clip, verify its SHA256
python scripts/fetch_fixtures.py --check  # offline verification of the cache (non-zero exit when missing/bad)
```

Tests that need the clip (`tests/unit/backend/test_video_real_fixture.py`) skip on a developer host without the cache
and **fail** when `DD_REQUIRE_LIVE=1` (QA), so a missing fixture never counts as a pass.

## Real street video fixture (traffic lights, pedestrians)

Added for the second detector (RF-DETR-N person / traffic_light / stop_sign, docs/IMPLEMENTATION_SPEC.md section 2.1)
and the traffic-light HSV heuristic. Candidates were searched on Wikimedia Commons (search API, `filetype:video`) and
checked through the API's `extmetadata`; this was the only forward-facing street-driving clip found with visible signal
heads, pedestrians and a free licence that is small enough to fetch in tests.

| Field | Value |
|---|---|
| Fixture id | `commons_memphis_front_st_ccby3` |
| Title | File:Video Downtown Memphis TN 3.theora.ogv |
| Page | https://commons.wikimedia.org/wiki/File:Video_Downtown_Memphis_TN_3.theora.ogv (page id 11606075, revision 528466410) |
| Direct URL | https://upload.wikimedia.org/wikipedia/commons/4/49/Video_Downtown_Memphis_TN_3.theora.ogv |
| Author | Thomas R Machnitzki (Wikimedia Commons uploader `User:DoxTxob`), "own work" |
| Date / description | 2010-09-26, "Memphis, Tennessee. Driving south on Front Street from Madison Avenue to Union Avenue. At the end of the video the memphis Cotton Exchange building is to the left." |
| License | **CC BY 3.0** (the author dual-licenses GFDL and CC BY 3.0; DreamDrive uses CC BY 3.0), https://creativecommons.org/licenses/by/3.0 |
| Attribution (required) | "Video Downtown Memphis TN 3" by Thomas R Machnitzki, CC BY 3.0, via Wikimedia Commons. Changes: 11.9 s excerpt from 11 s, re-encoded to H.264, audio removed. |
| License evidence | Verified 2026-10-10 via the Commons API (`prop=imageinfo&iiprop=extmetadata`: `LicenseShortName=CC BY 3.0`, `License=cc-by-3.0`, `Artist=Thomas R Machnitzki`, `Credit=Own work`, `AttributionRequired=true`) and the file page wikitext (`{{self|GFDL|cc-by-3.0}}`). |
| Source file | Ogg Theora video, 640x480, 30 fps (timebase 1000000/33333), 23.33 s, 14,877,426 bytes |
| Source SHA1 | `b5ab1eeba7139a2c7079d7171738bcb7efbf1140` (as published by the Commons API; checked on download) |
| Source SHA256 | `aa45a85d1a727e77929547715b563c4803bdfc87ad3814efa32a671627245a7c` (computed from the downloaded bytes) |

Derived clip:

| Field | Value |
|---|---|
| File | `tests/fixtures/cache/commons_memphis_front_st_ccby3_11s_12s.mp4` |
| Content | 11.9 s excerpt starting at 11 s: signalised intersections with overhead signal heads, pedestrians on the sidewalks, parked cars and a pickup truck ahead; audio dropped. 11.9 s instead of 12 s because the source's 30.0003 fps timebase would otherwise give a 12.03 s container. |
| Encoding | PyAV 18.1.0 / bundled libx264, yuv420p, CRF 20, preset medium, single-threaded, `+faststart`, 30 fps, 358 frames, 640x480 (source size kept; <= 1280x720) |
| SHA256 | `d9012591b78d06cedb8ec8cde732e3505ad3f2c6b4354fd118658cc3f253a180` (re-derived with identical bytes on the Windows host, PyAV 18.1.0) |

People in the clip are passers-by on public sidewalks at a distance (faces not identifiable at 640x480); licence plates
were not legible in the frames inspected. The clip has **no labels**: it is an end-to-end input for the two-model
pipeline and the HSV light heuristic, never an accuracy benchmark. A CC BY-SA 4.0 Toronto pedestrian-crossover clip
(`File:Roncesvalles Ave crossing at Galley Ave, PXO Type A lights flashing, July 2026.webm`) was also verified but not
adopted: it is filmed from the sidewalk, not from a vehicle, so the ego-corridor decisions would be meaningless.

## Synthetic test media

Unit tests generate their own short MP4 files with PyAV in pytest temporary directories (`tests/unit/backend/
test_video_synth.py`): constant-frame-rate, variable-frame-rate with irregular PTS, odd frame sizes (yuv444p),
display-matrix rotation and an optional silent AAC track. They are created at test time and never stored.

## Model weights

The YOLOPv2 checkpoint is not a test fixture and is not redistributed; see `config/models.resolved.yaml` and
`docs/LICENSES.md`.

## Uploaded videos

User uploads are private job inputs, hashed (SHA256) per job, subject to retention limits, and never used as
fixtures, training data or evaluation data without explicit rights.

## COCO val2017 evaluation subset

Used only to measure detector accuracy (docs/EVALUATION.md). Nothing from COCO is committed except the small manifest
`datasets/manifests/coco_val2017_street_v1.json` (image ids, URLs, sizes, licence ids; no images, no annotation copies).

| Field | Value |
|---|---|
| Dataset | COCO 2017 (Common Objects in Context), split `val2017` |
| Annotation archive | `annotations_trainval2017.zip` from http://images.cocodataset.org/annotations/annotations_trainval2017.zip, 252,907,541 bytes, SHA256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` (local path `datasets/raw/coco/`, git-ignored) |
| Annotation member used | `annotations/instances_val2017.json`, 19,987,840 bytes, SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` |
| Annotation licence | Creative Commons Attribution 4.0 (COCO Consortium terms of use, https://cocodataset.org/#termsofuse) |
| Image licences | The COCO Consortium does not own the images. Each image keeps its Flickr licence; the manifest records per image `license_id`, `license_name`, `license_url` and `flickr_url`. Use must follow the Flickr Terms of Use. |
| Images | Fetched on demand from each manifest `coco_url` (`http://images.cocodataset.org/val2017/<id>.jpg`) into a git-ignored cache (`datasets/raw/coco/val2017_cache/`); the SHA256 of every file is pinned in `cache_index.json` on first download and verified on every later use. Images are never redistributed. |
| Subset rule | Images with at least one non-crowd instance of {traffic light, stop sign, person} AND at least one non-crowd instance of {car, bus, truck, motorcycle, bicycle}; crowd (`iscrowd=1`) annotations never qualify an image; sorted by image id, first 300 taken (656 images qualify). |
| Manifest | `datasets/manifests/coco_val2017_street_v1.json`, regenerated byte-identically by `python -m app.evaluation.detector_accuracy build-manifest` |

Licence mix of the 300 selected images (from the COCO `licenses` table):

| COCO licence id | Licence | URL | Images |
|---|---|---|---|
| 1 | Attribution-NonCommercial-ShareAlike License | http://creativecommons.org/licenses/by-nc-sa/2.0/ | 100 |
| 2 | Attribution-NonCommercial License | http://creativecommons.org/licenses/by-nc/2.0/ | 39 |
| 3 | Attribution-NonCommercial-NoDerivs License | http://creativecommons.org/licenses/by-nc-nd/2.0/ | 68 |
| 4 | Attribution License | http://creativecommons.org/licenses/by/2.0/ | 50 |
| 5 | Attribution-ShareAlike License | http://creativecommons.org/licenses/by-sa/2.0/ | 25 |
| 6 | Attribution-NoDerivs License | http://creativecommons.org/licenses/by-nd/2.0/ | 18 |

Because 207 of the 300 images carry NonCommercial terms, the images are used only for local, non-redistributed
evaluation; they are not fixtures, are not shipped in any image and are not shown in the product UI.

## KITTI Road training split (drivable-area evaluation)

### KITTI Road benchmark - training split (drivable-area evaluation only)
- Source: `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip` (KITTI Vision Benchmark Suite,
  https://www.cvlibs.net/datasets/kitti/eval_road.php). 470,992,343 bytes, SHA256
  `45508c51ba09020235eb714e71c7d3ef15b4551be06d996772ecc8a20fc34a34`. Downloaded 2026-10-10 to
  `datasets/raw/kitti_road/` (gitignored).
- Licence: "All datasets and benchmarks on this page are copyright by us and published under the Creative Commons
  Attribution-NonCommercial-ShareAlike 3.0 License" (https://www.cvlibs.net/datasets/kitti/). Non-commercial; not
  redistributed; not used for training; used only for a local accuracy measurement.
- Attribution: J. Fritsch, T. Kuehnl, A. Geiger, "A New Performance Measure and Evaluation Benchmark for Road
  Detection Algorithms", ITSC 2013.
- Used: 289 training images (`image_2`) and their `<cat>_road_<idx>.png` ground truth (um 95, umm 96, uu 98).
  `um_lane` (ego-lane area) is hashed but not scored. Testing split and calibration files are unused.
- Committed: `datasets/manifests/kitti_road_training_v1.json` (member names, sizes, SHA256, GT colour histograms;
  no pixels). GT encoding taken from the official devkit (`devkit_road.zip`, SHA256
  `6f7cba10eeeb82d26cb0fa64ea4771ae92ff818be50098b162a21fcab8e06156`): R plane = valid area, B plane = road.
- Consumer: `backend/app/evaluation/segmentation_accuracy.py`; results and limitations in
  `docs/EVALUATION_SEGMENTATION.md`.

## Roboflow lane-marking dataset (lane-line evaluation)

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
