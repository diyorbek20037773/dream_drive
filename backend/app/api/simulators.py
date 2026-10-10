"""Simulator adapters, perception modes, scenarios and static road graphs."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api import catalog
from app.api.deps import SessionDep, SettingsDep, get_services
from app.errors import error_responses
from app.schemas.api import SimulatorList

router = APIRouter(tags=["simulators"])


@router.get(
    "/simulators",
    response_model=SimulatorList,
    summary="Supported simulators, perception modes, scenarios and road graphs",
    responses=error_responses(401),
)
def list_simulators(request: Request, session: SessionDep, settings: SettingsDep) -> SimulatorList:
    model = get_services(request).model_status.get()
    return SimulatorList(items=catalog.simulators(settings, model))
