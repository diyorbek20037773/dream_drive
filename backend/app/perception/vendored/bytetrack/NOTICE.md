# Vendored ByteTrack tracker

| Item | Value |
|---|---|
| Upstream | https://github.com/FoundationVision/ByteTrack |
| Commit | `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` (2022-12-11, "Merge pull request #296 from SkalskiP/main") |
| Upstream path | `yolox/tracker/` |
| License | MIT, Copyright (c) 2021 Yifu Zhang — full text in `LICENSE` (byte-identical copy, SHA256 `d55e22251df27e2011d431efdd7478e03771db5f5d81dcca45f59072fa0844e8`) |
| Retrieved | 2026-10-10, shallow clone outside the repository, read-only |

Upstream file SHA256 at that commit (the files here are modified ports of these):

| Upstream file | SHA256 |
|---|---|
| `yolox/tracker/basetrack.py` | `305131ea660c3c535a1db101085317d981ab57b9db34a9055225385622c2be40` |
| `yolox/tracker/byte_tracker.py` | `cb018650764db2c7d7d70c8c92fa900fc4fe62518419520b168b281bc71f30d5` |
| `yolox/tracker/kalman_filter.py` | `dc88e3b7c75131616e76ff6e5381239d618c8c07c81692002597416cbc24dce5` |
| `yolox/tracker/matching.py` | `2f32f709c92fa199b22c22b18162249ccde7033a4648485ae4e82a698294f1fc` |

## Modifications (DreamDrive)
1. **No `lap` / `cython_bbox` / `torch` / `cv2` dependency.** `linear_assignment` solves the same extended problem that
   `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)` builds (cost matrix padded with `thresh/2` dummy rows/columns,
   zero dummy-dummy block) with `scipy.optimize.linear_sum_assignment`; ties may resolve differently. `ious` is a numpy
   port of `cython_bbox.bbox_overlaps`, keeping its `+1` pixel area convention.
2. **Per-tracker-instance ID counter.** Upstream `BaseTrack._count` is a process-global class attribute, so IDs leak
   between runs. Here every `BYTETracker` owns (or is given) an `IdCounter`; one job's IDs never affect another's.
3. **Timestamp-aware.** `BYTETracker.update(dets, timestamp_s)` takes the real presentation time of the sampled frame.
   The Kalman motion model uses `dt = (t - t_prev) / nominal_dt` (nominal_dt = 1 / frame_rate), so a regular sample
   step is exactly upstream's `dt = 1`, while VFR gaps or skipped samples predict proportionally further. Process
   noise is scaled linearly with `dt` (identical to upstream at `dt = 1`). Lost tracks are removed when
   `t - t_last_seen > track_buffer_s` (upstream: `frame_id - end_frame > int(frame_rate / 30 * track_buffer)` frames;
   the default `track_buffer_s = 1.0` equals upstream's default 30-frame buffer at 30 fps).
   Expiry is also applied *before* association (to tracked and lost tracks), so after a sampling gap longer than
   the buffer an old track cannot be re-found merely because no intermediate update ran to expire it.
   Timestamps must strictly increase; a non-increasing timestamp raises `ValueError`.
4. **Input format.** `update` takes an `N x 5` array `[x1, y1, x2, y2, score]` already in original image pixels
   (upstream rescales from the detector's input size and supports a 7-column YOLOX output). Thresholds are
   constructor arguments (`track_thresh` 0.5, `low_thresh` 0.1, `match_thresh` 0.8, `mot20` False) instead of an
   argparse namespace; the second-stage (0.5) and unconfirmed (0.7) association thresholds are unchanged.
   Low-score boxes (`low_thresh < score < track_thresh`) are kept for the second association exactly as upstream.
5. **Detection back-reference.** Each `STrack` records `det_index`, the row of the current `update` input that
   updated it (or -1), so the caller can attach track IDs to the detector's own boxes instead of Kalman-smoothed boxes.
6. `removed_stracks` is capped at the most recent 1000 entries so memory stays bounded on long videos (it is only
   used to subtract freshly removed tracks from the lost list).
7. Removed unused code (re-ID embedding/fusion helpers, gating, `merge_matches`, `v_iou_distance`, multi-camera
   fields); added type hints; `np.float` (removed in NumPy 1.24) replaced by `np.float64`.
