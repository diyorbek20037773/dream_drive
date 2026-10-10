# Ported from ByteTrack yolox/tracker/kalman_filter.py @ d1bf0191adff59bc8fcfeaa0b33d3d1642552a99
# (MIT, (c) 2021 Yifu Zhang; the filter itself originates from deep_sort).
# DreamDrive modifications (see NOTICE.md): `predict`/`multi_predict` take `dt` in nominal-step units (dt=1 is
# upstream behaviour); the transition matrix uses that dt and process noise is scaled linearly with dt.
# Unused gating helpers removed; type hints added.
from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class KalmanFilter:
    """Constant-velocity Kalman filter over (cx, cy, aspect, height, vcx, vcy, vaspect, vheight) in image space."""

    def __init__(self) -> None:
        self._ndim = 4
        self._update_mat = np.eye(self._ndim, 2 * self._ndim)
        # Motion and observation uncertainty relative to the current state estimate (upstream values).
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def _motion_mat(self, dt: float) -> FloatArray:
        mat = np.eye(2 * self._ndim, 2 * self._ndim)
        for i in range(self._ndim):
            mat[i, self._ndim + i] = dt
        return mat

    def initiate(self, measurement: FloatArray) -> tuple[FloatArray, FloatArray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: FloatArray, covariance: FloatArray, dt: float = 1.0) -> tuple[FloatArray, FloatArray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) * dt
        motion_mat = self._motion_mat(dt)
        mean = np.dot(mean, motion_mat.T)
        covariance = np.linalg.multi_dot((motion_mat, covariance, motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: FloatArray, covariance: FloatArray) -> tuple[FloatArray, FloatArray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: FloatArray, covariance: FloatArray, dt: float = 1.0) -> tuple[FloatArray, FloatArray]:
        """Vectorised `predict` for N states (mean N x 8, covariance N x 8 x 8)."""
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T * dt
        motion_cov = np.asarray([np.diag(sqr[i]) for i in range(len(mean))])
        motion_mat = self._motion_mat(dt)
        mean = np.dot(mean, motion_mat.T)
        left = np.dot(motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, motion_mat.T) + motion_cov
        return mean, covariance

    def update(
        self, mean: FloatArray, covariance: FloatArray, measurement: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
