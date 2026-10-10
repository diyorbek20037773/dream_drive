"""The only simulator surface handed to runtime decision code.

``RuntimeView`` wraps an adapter and exposes exactly the four runtime methods of
``app.simulators.base.RuntimeSimulatorView``: camera pixels, ego telemetry, static road topology and static camera
calibration. It deliberately has no ``evaluator``, scene list, ``step`` or ``apply_control`` attribute, so code that
receives it cannot reach simulator truth or actuate the vehicle.
"""

from __future__ import annotations

from app.simulators.base import CameraCalibration, EgoState, RgbFrame, RoadGraph, RuntimeSimulatorView


class RuntimeView:
    __slots__ = ("__adapter",)

    def __init__(self, adapter: RuntimeSimulatorView) -> None:
        self.__adapter = adapter

    def get_rgb_frame(self) -> RgbFrame:
        return self.__adapter.get_rgb_frame()

    def get_ego_telemetry(self) -> EgoState:
        return self.__adapter.get_ego_telemetry()

    def get_route_graph(self) -> RoadGraph:
        return self.__adapter.get_route_graph()

    def get_camera_calibration(self) -> CameraCalibration:
        return self.__adapter.get_camera_calibration()
