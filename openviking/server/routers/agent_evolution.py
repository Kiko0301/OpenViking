# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Agent Evolution endpoints."""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, model_validator

from openviking.core.uri_validation import validate_request_viking_uri
from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.service.agent_evolution_service import (
    DEFAULT_TRAJECTORY_PAGE_LIMIT,
    MAX_TRAJECTORY_PAGE_LIMIT,
)

router = APIRouter(prefix="/api/v1/agent-evolution", tags=["agent-evolution"])


class LearnExperiencesRequest(BaseModel):
    trajectory_uris: list[str] | None = Field(
        default=None,
        max_length=10_000,
        description="Trigger Trajectories identifying Cases whose inboxes should be consumed; "
        "not a whitelist of files to learn. Existing pending work in those Cases is included.",
    )
    case_names: list[str] | None = Field(
        default=None, max_length=1_000, description="Canonical Cases whose inboxes to consume."
    )
    max_cases: int = Field(default=100, ge=1, le=1_000)
    wait: bool = True
    retry_failed: bool = False
    lease_seconds: int = Field(default=900, ge=30, le=86_400)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def reject_empty_explicit_filters(self) -> "LearnExperiencesRequest":
        if self.trajectory_uris == []:
            raise ValueError("trajectory_uris must be omitted or non-empty")
        if self.case_names == []:
            raise ValueError("case_names must be omitted or non-empty")
        return self


@router.post("/experiences/learn")
async def learn_experiences(
    request: LearnExperiencesRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Process pending Trajectories into at most one Experience update per Case."""
    trajectory_uris = (
        [validate_request_viking_uri(uri, _ctx) for uri in request.trajectory_uris]
        if request.trajectory_uris is not None
        else None
    )
    result = await get_service().agent_evolution.learn_experiences(
        ctx=_ctx,
        trajectory_uris=trajectory_uris,
        case_names=request.case_names,
        max_cases=request.max_cases,
        retry_failed=request.retry_failed,
        lease_seconds=request.lease_seconds,
        wait=request.wait,
    )
    return Response(status="ok", result=result)


@router.get("/experiences/trajectories")
async def list_experience_trajectories(
    experience_uri: str = Query(..., description="Experience file URI"),
    limit: int = Query(
        DEFAULT_TRAJECTORY_PAGE_LIMIT,
        ge=1,
        le=MAX_TRAJECTORY_PAGE_LIMIT,
    ),
    offset: int = Query(0, ge=0),
    start_date: str | None = Query(None, description="UTC start date, inclusive (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="UTC end date, inclusive (YYYY-MM-DD)"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """List trajectories produced by commits that read an Experience."""
    service = get_service()
    experience_uri = validate_request_viking_uri(experience_uri, _ctx)
    result = await service.agent_evolution.list_trajectories_by_experience(
        experience_uri=experience_uri,
        ctx=_ctx,
        limit=limit,
        offset=offset,
        start_date=start_date,
        end_date=end_date,
    )
    return Response(status="ok", result=result)


@router.get("/experiences/outcomes")
async def get_experience_outcome_distribution(
    experience_uri: str = Query(..., description="Experience file URI"),
    start_date: str | None = Query(None, description="UTC start date, inclusive (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="UTC end date, inclusive (YYYY-MM-DD)"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Count application trajectories by outcome for an Experience."""
    service = get_service()
    experience_uri = validate_request_viking_uri(experience_uri, _ctx)
    result = await service.agent_evolution.get_experience_outcome_distribution(
        experience_uri=experience_uri,
        ctx=_ctx,
        start_date=start_date,
        end_date=end_date,
    )
    return Response(status="ok", result=result)
