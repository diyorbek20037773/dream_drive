"""Optional CARLA adapter placeholder that reports *unavailable* with an honest reason.

CARLA 0.9.16 is the documented optional candidate (docs/SIMULATION.md). This build has no recorded CARLA smoke test
(server/client match, synchronous mode, RGB camera, control, cleanup), the official guide recommends 8 GB+ VRAM while
the developer GPU has 6 GB, and the ``carla`` client wheel is not a backend dependency. The adapter therefore never
claims availability; ``carla`` is only imported lazily to report whether the client package exists.
"""

from __future__ import annotations

import importlib.util

from app.config import Settings, get_settings

CARLA_CANDIDATE_VERSION = "0.9.16"


class CarlaUnavailable(RuntimeError):  # noqa: N818 - mirrors the API reason wording
    """Raised when a CARLA adapter is requested in a build where it is not supported."""


def carla_availability(settings: Settings | None = None) -> tuple[bool, str]:
    """(available, reason). Always unavailable in this build; the reason says which prerequisite is missing."""
    settings = settings or get_settings()
    if not settings.carla_host:
        return False, (
            "CARLA_HOST is not configured. LiteRoad is the default simulator on this hardware; CARLA "
            f"{CARLA_CANDIDATE_VERSION} is an optional, untested adapter."
        )
    if importlib.util.find_spec("carla") is None:
        return False, "The CARLA Python client package is not installed in this image."
    return False, (
        f"No CARLA {CARLA_CANDIDATE_VERSION} smoke test (synchronous RGB + control + cleanup) is recorded "
        "for this build, so the adapter is not enabled."
    )


class CarlaAdapter:
    """Constructing it always raises ``CarlaUnavailable`` with the current reason."""

    def __init__(self, settings: Settings | None = None) -> None:
        available, reason = carla_availability(settings)
        if not available:
            raise CarlaUnavailable(reason)
        raise CarlaUnavailable("CARLA adapter is not implemented beyond availability reporting")  # pragma: no cover
