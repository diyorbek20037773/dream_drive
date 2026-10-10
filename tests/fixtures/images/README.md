# Image fixtures

| File | Source | License | SHA256 | Used by |
|---|---|---|---|---|
| `yolopv2_example.jpg` | YOLOPv2 repository, `data/example.jpg` at commit `9388c6aa55f3bf5a878679a1338bb090cff6a256` (https://github.com/CAIC-AD/YOLOPv2/blob/9388c6aa55f3bf5a878679a1338bb090cff6a256/data/example.jpg). Copied byte-for-byte, unmodified. 1280x720 baseline JPEG, 55,105 bytes. | MIT, Copyright (c) 2022 CAIC-AD (repository license; full text in `LICENSE.YOLOPv2`). The repository does not state a separate license or origin for its sample images; they are distributed as part of the MIT-licensed repository. | `cdd64de6851f9eca958814858d30d26b6efc68e2aa6af448426817efa2869533` | `tests/model_smoke` (real-weight smoke and CPU/GPU benchmark) |
| `LICENSE.YOLOPv2` | YOLOPv2 repository `LICENSE` at the same commit, unmodified. | MIT | `8d375e3fc358cd07c2e894d0bc79971d24fc27ca9596d2545f4d9e320ee39b57` | license notice for the file above |
| `coco_val2017_000000190923.jpg` | COCO 2017 validation image id 190923, downloaded unmodified from http://images.cocodataset.org/val2017/000000190923.jpg on 2026-10-10. Original: Flickr photo "IMG_0118" by **ttarasiuk**, https://www.flickr.com/photos/tara_siuk/3706332805/ (COCO `flickr_url` http://farm3.staticflickr.com/2669/3706332805_237c47a25d_z.jpg). 375x500 baseline JPEG, 138,770 bytes. Street scene with a vehicle traffic light, pedestrians, a truck and a "ONE WAY" sign. | **CC BY 2.0** (https://creativecommons.org/licenses/by/2.0/): COCO licence id 4 "Attribution License", and the Flickr photo page shows CC BY 2.0 (checked 2026-10-10). Attribution: "IMG_0118" by ttarasiuk, CC BY 2.0. | `3407447c4e8ca2ceb0db4ba1cb97dee1547e63fb3499cb67d6813ac5d4275d3c` | `tests/model_smoke` (RF-DETR class-id mapping check with real weights: person + traffic light) |
| `coco_val2017_000000190923.annotations.json` | All 17 COCO 2017 val instance annotations of image 190923 (`instances_val2017.json` from `annotations_trainval2017.zip`, SHA256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`), bbox values copied unmodified. | Annotations: CC BY 4.0, COCO Consortium (https://cocodataset.org/#termsofuse) | `783afb791bce53f9e843430b15fb3280f6bad098d756edf0e78df09a55d03e64` | ground-truth boxes for the mapping check above |

The example frame is a daytime road scene of the kind YOLOPv2 was trained on (BDD100K-style dashcam view). It has no
ground-truth labels in this repository, so tests that use it check identity, shapes and sanity only; they make no
accuracy claim.

The COCO image is used to prove the RF-DETR class-id mapping with real weights: a returned `person` (101) and
`traffic_light` (110) box must overlap a COCO ground-truth box of the same category (IoU >= 0.5). One image is a
mapping/plumbing check, not an accuracy measurement. COCO val2017 is the standard COCO evaluation split; DreamDrive
has not independently verified which images the publisher used for training.
