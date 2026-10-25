# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Regression tests for transient marker-read failures during reindex."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.core.context import ContextLevel
from openviking.server.identity import RequestContext, Role
from openviking.service import reindex_executor as reindex_module
from openviking.service.reindex_executor import ReindexExecutor, _ReindexCounters
from openviking_cli.exceptions import NotFoundError, PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier


class _FailingMarkerFS:
    """FS surface whose abstract/overview raise the injected error."""

    def __init__(self, error: Exception):
        self._error = error
        self._async_agfs = SimpleNamespace(
            pathlock_acquire_exact=AsyncMock(return_value=object()),
            pathlock_release=AsyncMock(),
        )

    def _uri_to_path(self, uri, ctx=None):
        return f"/local{uri}"

    async def abstract(self, uri, ctx=None):
        raise self._error

    async def overview(self, uri, ctx=None):
        raise self._error


def _ctx() -> RequestContext:
    # Role subclasses str, so Role("root") is the typed Role instance.
    return RequestContext(user=UserIdentifier("acc1", "u1"), role=Role("root"))


def _executor(monkeypatch, fs) -> ReindexExecutor:
    monkeypatch.setattr(reindex_module, "get_viking_fs", lambda: fs)
    return ReindexExecutor()


async def test_marker_reindex_does_not_delete_vector_on_transient_read_error(
    monkeypatch,
):
    fs = _FailingMarkerFS(PermissionDeniedError("agfs hiccup"))
    executor = _executor(monkeypatch, fs)
    delete = AsyncMock()
    monkeypatch.setattr(executor, "delete_uri_level", delete)

    # A transient read failure must fail the reindex (retryable), not be
    # treated as "marker gone" — on main the swallowed error made this path
    # delete the directory's existing L0/L1 vector.
    with pytest.raises(PermissionDeniedError):
        await executor.reindex_directory_marker(
            dir_uri="viking://user/acc1/resources/project",
            level=ContextLevel.ABSTRACT,
            ctx=_ctx(),
        )

    delete.assert_not_awaited()


async def test_marker_reindex_still_deletes_vector_for_missing_directory(
    monkeypatch,
):
    fs = _FailingMarkerFS(
        NotFoundError("viking://user/acc1/resources/project", "directory")
    )
    executor = _executor(monkeypatch, fs)
    delete = AsyncMock()
    monkeypatch.setattr(executor, "delete_uri_level", delete)

    # A genuinely missing directory keeps the delete-on-empty contract.
    await executor.reindex_directory_marker(
        dir_uri="viking://user/acc1/resources/project",
        level=ContextLevel.ABSTRACT,
        ctx=_ctx(),
    )

    delete.assert_awaited_once()


async def test_rebuild_chain_counts_marker_read_failures(monkeypatch):
    fs = _FailingMarkerFS(PermissionDeniedError("agfs hiccup"))
    executor = _executor(monkeypatch, fs)
    upsert = AsyncMock()
    monkeypatch.setattr(executor, "_upsert_context", upsert)
    counters = _ReindexCounters()

    await executor._reindex_memory_directory_chain(
        directory_uris=["viking://user/acc1/memories/project"],
        counters=counters,
        ctx=_ctx(),
    )

    # A transient marker read fails that directory's record like the upsert
    # error handling below it, instead of silently skipping the vector.
    assert counters.failed_records == 1
    assert counters.warnings and "Failed to read directory markers" in counters.warnings[0]
    upsert.assert_not_awaited()
