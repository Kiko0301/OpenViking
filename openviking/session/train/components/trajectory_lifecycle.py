# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Trajectory staging paths and recoverable archival after Experience learning."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.memory_updater import MemoryUpdater, MemoryUpdateResult
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.train.components.trajectory_observation import (
    TRAJECTORY_OUTCOME_TAG,
    trajectory_source_session_uri,
)
from openviking.utils.path_safety import normalize_storage_target_uri

TERMINAL_STATES = frozenset({"applied", "no_op"})
_ROOT = r"(?P<root>viking://user/[^/]+/memories/trajectories/[^/]+)"
_NAME = r"(?P<name>[SFU]_[A-Za-z0-9_-]+\.md)"
_INBOX = re.compile(_ROOT + r"/inbox/(?P<date>\d{8})_" + _NAME)
_ARCHIVE = re.compile(_ROOT + r"/archive/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/" + _NAME)


def trajectory_archive_uri(uri: str) -> str:
    """Map a dated inbox filename to its immutable archival address."""
    if normalize_storage_target_uri(uri) != uri:
        raise ValueError(f"Non-canonical Trajectory URI: {uri}")
    match = _INBOX.fullmatch(uri)
    if match:
        date = datetime.strptime(match["date"], "%Y%m%d")
        return f"{match['root']}/archive/{date:%Y/%m/%d}/{match['name']}"
    match = _ARCHIVE.fullmatch(uri)
    if match:
        datetime(int(match["year"]), int(match["month"]), int(match["day"]))
        return uri
    raise ValueError(f"Expected a dated inbox or archive Trajectory URI: {uri}")


def _state(memory: MemoryFile) -> dict:
    value = memory.extra_fields.get("experience_update")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or value.get("state") not in TERMINAL_STATES:
        raise ValueError(f"Cannot archive an unfinished Trajectory: {memory.uri}")
    if memory.memory_type != "trajectories":
        raise ValueError(f"Not a Trajectory: {memory.uri}")
    trajectory_source_session_uri(memory)
    if memory.extra_fields.get("archive_uri") != trajectory_archive_uri(memory.uri):
        raise ValueError(f"Trajectory archive identity mismatch: {memory.uri}")
    return value


def _same_observation(source: MemoryFile, target: MemoryFile) -> bool:
    def fields(memory):
        return {k: v for k, v in memory.extra_fields.items() if k != "experience_update"}

    return (
        source.memory_type == target.memory_type
        and source.plain_content() == target.plain_content()
        and fields(source) == fields(target)
        and trajectory_source_session_uri(source) == trajectory_source_session_uri(target)
        and _state(source)["state"] == _state(target)["state"]
    )


async def archive_trajectory(
    memory: MemoryFile,
    *,
    viking_fs: Any,
    ctx: Any,
    registry: Any,
    vikingdb: Any,
) -> MemoryFile:
    """Copy, enqueue at the final URI, then remove inbox under two exact-file locks.

    Like VikingFS.mv this uses the existing copy/delete operations, but keeps the
    source until embedding enqueue succeeds. A terminal inbox file is the recovery
    record: retries finish archival without rerunning learning. Both copies may
    exist after a failure; never overwrite an unrelated archival file.
    """
    source_uri = memory.uri
    target_uri = trajectory_archive_uri(source_uri)
    if source_uri == target_uri:
        _state(memory)
        return memory
    paths = sorted(viking_fs._uri_to_path(uri, ctx=ctx) for uri in (source_uri, target_uri))
    lock = await viking_fs._async_agfs.pathlock_acquire_exact_batch(paths, timeout_secs=30)
    if lock is None:
        raise RuntimeError("Trajectory archival requires file write locks")
    try:
        source = None
        if await viking_fs.exists(source_uri, ctx=ctx):
            source = MemoryFileUtils.read(
                await viking_fs.read_file(source_uri, ctx=ctx), uri=source_uri
            )
            _state(source)
        if await viking_fs.exists(target_uri, ctx=ctx):
            archived = MemoryFileUtils.read(
                await viking_fs.read_file(target_uri, ctx=ctx), uri=target_uri
            )
            _state(archived)
            if not _same_observation(source or memory, archived):
                raise ValueError(f"Trajectory archive filename collision: {target_uri}")
        else:
            if source is None:
                raise FileNotFoundError(source_uri)
            await viking_fs.mkdir(
                target_uri.rsplit("/", 1)[0], exist_ok=True, ctx=ctx, lease_ref=lock
            )
            await viking_fs.cp(source_uri, target_uri, ctx=ctx, lease_ref=lock)
            archived = source.model_copy(deep=True, update={"uri": target_uri})
            for link in archived.links:
                if "from_uri" in link:
                    link["from_uri"] = target_uri
            await viking_fs.write_file(
                target_uri, MemoryFileUtils.write(archived), ctx=ctx, lease_ref=lock
            )

        state = dict(_state(archived))
        if vikingdb is not None and not state.get("embedding_enqueued"):
            updater = MemoryUpdater(registry=registry, vikingdb=vikingdb)
            updater._viking_fs = viking_fs
            result = MemoryUpdateResult()
            result.add_written(target_uri)
            tags = state.get("search_tags") or [
                f"{TRAJECTORY_OUTCOME_TAG}={archived.extra_fields['outcome']}"
            ]
            await updater._vectorize_memories(
                result,
                ctx,
                uri_memory_type_map={target_uri: "trajectories"},
                search_tags_by_uri={target_uri: tags},
                strict=True,
            )
            # Persist before deleting inbox so a delete failure does not enqueue again.
            # A crash between enqueue and this write may repeat an idempotent upsert;
            # it must never trigger another Experience update.
            state["embedding_enqueued"] = True
            archived.extra_fields["experience_update"] = state
            await viking_fs.write_file(
                target_uri, MemoryFileUtils.write(archived), ctx=ctx, lease_ref=lock
            )
        if source is not None:
            await viking_fs.rm(source_uri, ctx=ctx, lease_ref=lock)
        return archived
    finally:
        await viking_fs._async_agfs.pathlock_release(lock)
