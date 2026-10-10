"""Reusable path/query parameter declarations."""

from __future__ import annotations

from typing import Annotated

from fastapi import Path, Query

from app.schemas.api import RESOURCE_ID_PATTERN

JobIdPath = Annotated[str, Path(pattern=RESOURCE_ID_PATTERN, description="Video job id.")]
RunIdPath = Annotated[str, Path(pattern=RESOURCE_ID_PATTERN, description="Simulation run id.")]
EvaluationIdPath = Annotated[str, Path(pattern=RESOURCE_ID_PATTERN, description="Evaluation id.")]
PageLimit = Annotated[int, Query(ge=1, le=100, description="Maximum items per page.")]
PageCursor = Annotated[str | None, Query(max_length=256, description="Opaque cursor from a previous page.")]
