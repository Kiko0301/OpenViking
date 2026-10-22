# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Session file writes must respect native RAGFS lease coverage."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from openviking.server.auth import get_session_request_context
from openviking.server.routers import sessions as sessions_router
from openviking.service.session_service import SessionService
from openviking.session.session import Session
from openviking.storage.viking_fs import VikingFS
from openviking.ttl import wiring
from openviking.ttl.runtime import DirectoryTTL
from openviking.utils.agfs_utils import RagfsBindingConfig, create_agfs_client
from openviking_cli.utils.config.agfs_config import AGFSConfig
from openviking_cli.utils.config.ttl_config import TTLConfig


@pytest.fixture(params=["unconfigured", "account_disabled", "enabled"])
def native_session(request, tmp_path, monkeypatch, request_context):
    fs = VikingFS(
        agfs=create_agfs_client(
            RagfsBindingConfig(agfs=AGFSConfig(path=str(tmp_path), backend="local"))
        )
    )
    monkeypatch.setattr(wiring, "_adapter", None)
    if request.param != "unconfigured":
        adapter = DirectoryTTL(resource_id="session-lease-test")
        adapter.fs = fs
        adapter.policies = AsyncMock(
            return_value=(
                TTLConfig(sessions=7),
                TTLConfig(sessions="disabled") if request.param == "account_disabled" else None,
            )
        )
        adapter.client = AsyncMock(side_effect=AssertionError("request path accessed Redis"))
        wiring.set_ttl_adapter(adapter)
        wiring.install_ttl_hook(fs, adapter)
    return Session(viking_fs=fs, ctx=request_context, session_id="native-lease-test")


async def test_create_session_http_with_native_file_locks(native_session, monkeypatch):
    session = native_session
    service = SimpleNamespace(
        sessions=SessionService(viking_fs=session._viking_fs),
        initialize_user_directories=AsyncMock(),
    )
    monkeypatch.setattr(sessions_router, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(sessions_router.router)
    app.dependency_overrides[get_session_request_context] = lambda: session.ctx
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post("/api/v1/sessions", json={"session_id": session.session_id})
    assert response.status_code == 200, response.text
    await session.load()
    assert session.meta.session_id == session.session_id
    assert (
        await session._viking_fs.read_file(f"{session.uri}/messages.jsonl", ctx=session.ctx) == ""
    )


async def test_commit_metadata_with_native_file_lock(native_session):
    session = native_session
    fs = session._viking_fs
    # Seed directly so this test isolates the Phase 2 write from session creation.
    await fs.write_file(f"{session.uri}/messages.jsonl", "", ctx=session.ctx)
    await fs.write_file(
        f"{session.uri}/.meta.json", json.dumps(session.meta.to_dict()), ctx=session.ctx
    )
    await fs.mkdir(f"{session.uri}/history/archive_001", ctx=session.ctx)

    saved = await session._merge_and_save_commit_meta(1, {"events": 1}, None)

    assert saved is not None
    meta = json.loads(await fs.read_file(f"{session.uri}/.meta.json", ctx=session.ctx))
    assert meta["commit_count"] == 1
    assert meta["memories_extracted"]["events"] == 1
    assert meta["memories_extracted"]["total"] == 1
