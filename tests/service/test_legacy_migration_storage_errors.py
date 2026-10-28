# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Migration must not turn storage faults into missing legacy data."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from openviking.pyagfs.exceptions import AGFSTimeoutError
from openviking.service.legacy_migration import LegacyDataMigration
from tests.utils.mock_agfs import MockLocalAGFS

ACCOUNT = "account1"
SOURCE = f"/local/{ACCOUNT}/session/session1"


class FaultingAGFS(MockLocalAGFS):
    def __init__(self, root_path: Path):
        super().__init__(root_path=root_path)
        self.faults: dict[tuple[str, str], Exception] = {}

    def fail_once(self, operation: str, path: str) -> None:
        self.faults[(operation, path)] = AGFSTimeoutError(f"{operation} timed out: {path}")

    def _raise_if_faulted(self, operation: str, path: str) -> None:
        fault = self.faults.pop((operation, path), None)
        if fault is not None:
            raise fault

    def ls(self, path, ctx=None, **kwargs):
        self._raise_if_faulted("ls", path)
        return super().ls(path, ctx=ctx, **kwargs)

    def stat(self, path, ctx=None):
        self._raise_if_faulted("stat", path)
        return super().stat(path, ctx=ctx)

    def read(self, path, ctx=None, **kwargs):
        self._raise_if_faulted("read", path)
        return super().read(path, ctx=ctx, **kwargs)


def _migration(tmp_path: Path, *, metadata: str | None = '{"created_by_user_id":"bob"}'):
    agfs = FaultingAGFS(root_path=tmp_path)
    if metadata is not None:
        agfs.write(f"{SOURCE}/.meta.json", metadata)
    agfs.write(f"{SOURCE}/messages.jsonl", '{"role":"user"}\n')
    manager = SimpleNamespace(
        list_accounts=AsyncMock(return_value=[{"account_id": ACCOUNT}]),
        list_users_page=AsyncMock(return_value={"users": [{"user_id": "alice"}]}),
        has_user=AsyncMock(side_effect=lambda _account, user: user == "alice"),
        register_user=AsyncMock(),
    )
    migration = LegacyDataMigration(
        viking_fs=cast(Any, SimpleNamespace(agfs=agfs)),
        api_key_manager=manager,
        service=SimpleNamespace(initialize_user_directories=AsyncMock()),
    )
    return migration, agfs


@pytest.mark.asyncio
async def test_migration_aborts_instead_of_guessing_owner_after_metadata_timeout(tmp_path):
    migration, agfs = _migration(tmp_path)
    agfs.fail_once("read", f"{SOURCE}/.meta.json")

    with pytest.raises(AGFSTimeoutError, match="read timed out"):
        await migration.run()

    assert not (tmp_path / f"local/{ACCOUNT}/user/alice/sessions/session1").exists()


@pytest.mark.asyncio
async def test_preflight_surfaces_session_listing_timeout(tmp_path):
    migration, agfs = _migration(tmp_path)
    agfs.fail_once("ls", f"/local/{ACCOUNT}/session")

    with pytest.raises(AGFSTimeoutError, match="ls timed out"):
        await migration.preflight()


@pytest.mark.asyncio
async def test_migration_surfaces_source_stat_timeout_instead_of_skipping(tmp_path):
    migration, agfs = _migration(tmp_path)
    agfs.fail_once("stat", SOURCE)

    with pytest.raises(AGFSTimeoutError, match="stat timed out"):
        await migration.run()


@pytest.mark.asyncio
async def test_preflight_uses_metadata_owner_when_read_succeeds(tmp_path):
    migration, _agfs = _migration(tmp_path)

    plan = await migration.preflight()

    assert plan.operations[0].target_path == f"/local/{ACCOUNT}/user/bob/sessions/session1"


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [None, "{invalid-json"])
async def test_preflight_keeps_single_user_fallback_for_missing_or_invalid_metadata(
    tmp_path, metadata
):
    migration, _agfs = _migration(tmp_path, metadata=metadata)

    plan = await migration.preflight()

    assert plan.operations[0].target_path == f"/local/{ACCOUNT}/user/alice/sessions/session1"


@pytest.mark.asyncio
async def test_genuinely_missing_paths_still_return_empty_listing_and_stat(tmp_path):
    migration, _agfs = _migration(tmp_path)

    assert await migration._ls(f"/local/{ACCOUNT}/missing") == []
    assert await migration._stat(f"/local/{ACCOUNT}/missing") is None
