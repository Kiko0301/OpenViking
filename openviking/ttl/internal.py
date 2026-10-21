# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Internal transport, registered only when a commercial adapter is loaded.

No product logic or Redis access belongs here. The Server owns scope selection;
the adapter owns registration, reset serialization and lock-protected recycling.
"""

from dataclasses import asdict

from fastapi import APIRouter, Depends, Request

from openviking.core.namespace import uri_parts
from openviking.server.auth import get_request_context, require_auth_root
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext, Role
from openviking.server.models import Response
from openviking.server.routers.admin import _check_account_exists, _get_runtime_config_manager
from openviking.ttl.adapter import TTLBackfillRequest, TTLRecycleRequest
from openviking_cli.exceptions import InvalidArgumentError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/admin/ttl", include_in_schema=False)


@router.get("/directories")
@require_auth_root
async def directories(
    request: Request,
    account_id: str,
    uri: str,
    ctx: RequestContext = Depends(get_request_context),
):
    """Enumerate only users/peers for Server scope selection, including history.

    ROOT cannot call tenant data APIs in api_key mode. This narrow admin call
    binds the checked account explicitly and reads no content or abstracts.
    """
    try:
        parts = uri_parts(uri)
    except ValueError as exc:
        raise InvalidArgumentError(str(exc)) from exc
    if parts != ["user"] and not (len(parts) == 3 and parts[0] == "user" and parts[2] == "peers"):
        raise InvalidArgumentError("Only user and peer containers can be enumerated")
    await _check_account_exists(request, account_id)
    account_ctx = RequestContext(user=UserIdentifier(account_id, "ttl-internal"), role=Role.ROOT)
    entries = await get_service().viking_fs.ls(
        uri, ctx=account_ctx, node_limit=None, show_all_hidden=True, include_abstract=False
    )
    return Response(status="ok", result=[entry["uri"] for entry in entries if entry.get("isDir")])


@router.get("/configuration")
@require_auth_root
async def configuration(
    request: Request,
    account_id: str,
    ctx: RequestContext = Depends(get_request_context),
):
    """Return one published policy pair, including startup/cluster merging."""
    await _check_account_exists(request, account_id)
    policies = await _get_runtime_config_manager().resolve_account(
        account_id,
        lambda view: {
            name: config.ttl.model_dump(by_alias=True) if config.ttl else {}
            for name, config in (("cluster", view.cluster), ("account", view.account))
        },
    )
    return Response(status="ok", result=policies)


@router.post("/initial")
@require_auth_root
async def initial(
    body: TTLBackfillRequest,
    request: Request,
    ctx: RequestContext = Depends(get_request_context),
):
    if not body.resource_id.strip() or not body.account_id.strip():
        raise InvalidArgumentError("resource_id and account_id are required")
    await _check_account_exists(request, body.account_id)
    # Adapter verifies target, current policy, roots and persisted state before
    # accepting. Token deduplication belongs to the adapter; an untagged manual
    # reset still requires confirmation before retry after a lost response.
    await request.app.state.ttl_adapter.register_initial_backfill(body)
    logger.info(
        "TTL initial registered request_id=%s scope=%s/%s event=%s",
        getattr(request.state, "request_id", ""),
        body.resource_id,
        body.account_id,
        "reset" if body.reset else "resume",
    )
    return Response(status="ok")


@router.post("/recycle")
@require_auth_root
async def recycle(
    body: TTLRecycleRequest,
    request: Request,
    ctx: RequestContext = Depends(get_request_context),
):
    if not body.resource_id.strip() or not body.account_id.strip() or body.limit <= 0:
        raise InvalidArgumentError("resource_id, account_id and a positive limit are required")
    await _check_account_exists(request, body.account_id)
    result = await request.app.state.ttl_adapter.recycle_due(body)
    return Response(status="ok", result=asdict(result))
