# Ported from ByteTrack yolox/tracker/byte_tracker.py @ d1bf0191adff59bc8fcfeaa0b33d3d1642552a99
# (MIT, (c) 2021 Yifu Zhang). Association logic (two-stage high/low score matching, unconfirmed handling, track
# birth/death, duplicate removal) follows upstream line by line.
# DreamDrive modifications (see NOTICE.md): per-instance IdCounter; timestamp-aware prediction and lost-track
# expiry; N x 5 pixel-space input; constructor thresholds; det_index back-reference; no torch dependency.
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import matching
from .basetrack import BaseTrack, IdCounter, TrackState
from .kalman_filter import FloatArray, KalmanFilter


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: FloatArray, score: float, det_index: int = -1) -> None:
        super().__init__()
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter: KalmanFilter | None = None
        self.mean: FloatArray | None = None
        self.covariance: FloatArray | None = None
        self.is_activated = False
        self.score = float(score)
        self.tracklet_len = 0
        self.det_index = det_index

    def predict(self, dt: float = 1.0) -> None:
        assert self.mean is not None and self.covariance is not None and self.kalman_filter is not None
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance, dt)

    @staticmethod
    def multi_predict(stracks: list[STrack], dt: float = 1.0) -> None:
        if len(stracks) > 0:
            means, covs = [], []
            for st in stracks:
                assert st.mean is not None and st.covariance is not None, "pooled tracks are always initiated"
                means.append(st.mean.copy())
                covs.append(st.covariance)
            multi_mean = np.asarray(means)
            multi_covariance = np.asarray(covs)
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance, dt)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance, strict=True)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int, time_s: float, ids: IdCounter) -> None:
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = ids.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.start_time_s = time_s
        self.last_time_s = time_s

    def re_activate(self, new_track: STrack, frame_id: int, time_s: float) -> None:
        assert self.kalman_filter is not None and self.mean is not None and self.covariance is not None
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.last_time_s = time_s
        self.score = new_track.score
        self.det_index = new_track.det_index

    def update(self, new_track: STrack, frame_id: int, time_s: float) -> None:
        """Update a matched track."""
        assert self.kalman_filter is not None and self.mean is not None and self.covariance is not None
        self.frame_id = frame_id
        self.last_time_s = time_s
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.det_index = new_track.det_index

    @property
    def tlwh(self) -> FloatArray:
        """Current box as (top-left x, top-left y, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> FloatArray:
        """Current box as (x1, y1, x2, y2)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: FloatArray) -> FloatArray:
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: FloatArray) -> FloatArray:
        ret = np.asarray(tlbr, dtype=np.float64).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self) -> str:
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        frame_rate: float = 30.0,
        track_buffer_s: float = 1.0,
        mot20: bool = False,
        ids: IdCounter | None = None,
    ) -> None:
        if not 0 <= low_thresh < track_thresh <= 1:
            raise ValueError("require 0 <= low_thresh < track_thresh <= 1")
        if frame_rate <= 0 or track_buffer_s <= 0:
            raise ValueError("frame_rate and track_buffer_s must be positive")
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0
        self.last_time_s: float | None = None
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.mot20 = mot20
        self.det_thresh = track_thresh + 0.1
        self.nominal_dt_s = 1.0 / frame_rate
        self.max_time_lost_s = track_buffer_s
        self.kalman_filter = KalmanFilter()
        self.ids = ids if ids is not None else IdCounter()

    def update(self, output_results: NDArray[np.floating], timestamp_s: float) -> list[STrack]:
        """Associate one frame's detections (N x 5: x1, y1, x2, y2, score in image pixels) at `timestamp_s`."""
        dets_all = np.asarray(output_results, dtype=np.float64).reshape(-1, 5)
        if self.last_time_s is not None and not timestamp_s > self.last_time_s:
            raise ValueError("timestamps must strictly increase within one tracker run")
        dt = 1.0 if self.last_time_s is None else (timestamp_s - self.last_time_s) / self.nominal_dt_s
        self.last_time_s = timestamp_s
        self.frame_id += 1
        for track in self.tracked_stracks + self.lost_stracks:
            track.det_index = -1

        activated_stracks: list[STrack] = []
        refind_stracks: list[STrack] = []
        lost_stracks: list[STrack] = []
        removed_stracks: list[STrack] = []

        # Time-based expiry before association: after a sampling gap longer than the buffer (VFR gap, skipped
        # samples) a track must not be re-found just because no intermediate update ran to expire it.
        expired = [
            t for t in self.tracked_stracks + self.lost_stracks if timestamp_s - t.last_time_s > self.max_time_lost_s
        ]
        if expired:
            for track in expired:
                track.mark_removed()
            self.tracked_stracks = [t for t in self.tracked_stracks if t.state != TrackState.Removed]
            self.lost_stracks = [t for t in self.lost_stracks if t.state != TrackState.Removed]
            removed_stracks.extend(expired)

        scores = dets_all[:, 4]
        bboxes = dets_all[:, :4]
        indices = np.arange(len(dets_all))

        remain_inds = scores > self.track_thresh
        inds_low = scores > self.low_thresh
        inds_high = scores < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        detections = [
            STrack(STrack.tlbr_to_tlwh(tlbr), s, int(i))
            for tlbr, s, i in zip(bboxes[remain_inds], scores[remain_inds], indices[remain_inds], strict=True)
        ]

        # Add newly detected tracklets to tracked_stracks
        unconfirmed: list[STrack] = []
        tracked_stracks: list[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 2: first association, with high score detection boxes
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool, dt)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, timestamp_s)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, timestamp_s)
                refind_stracks.append(track)

        # Step 3: second association, with low score detection boxes
        detections_second = [
            STrack(STrack.tlbr_to_tlwh(tlbr), s, int(i))
            for tlbr, s, i in zip(bboxes[inds_second], scores[inds_second], indices[inds_second], strict=True)
        ]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, timestamp_s)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, timestamp_s)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, timestamp_s)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 4: init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, timestamp_s, self.ids)
            activated_stracks.append(track)

        # Step 5: update state (expiry measured in real seconds since the track was last observed)
        for track in self.lost_stracks:
            if timestamp_s - track.last_time_s > self.max_time_lost_s:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # Bound memory on long videos: removed tracks are only needed for the sub_stracks bookkeeping above.
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]
        return [track for track in self.tracked_stracks if track.is_activated]


def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
    exists: dict[int, int] = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        stracks.pop(t.track_id, None)
    return list(stracks.values())


def remove_duplicate_stracks(stracksa: list[STrack], stracksb: list[STrack]) -> tuple[list[STrack], list[STrack]]:
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs, strict=True):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
