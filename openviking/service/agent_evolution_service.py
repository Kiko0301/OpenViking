# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Agent Evolution lineage queries and deferred Experience learning."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

from openviking.server.identity import RequestContext
from openviking.service.task_tracker import get_task_tracker
from openviking.session.memory.experience_lineage import (
    TRAJECTORY_OUTCOMES,
    canonical_experience_uri,
    experience_source_tag,
    trajectory_outcome_tag,
)
from openviking.storage.expr import And, Eq, PathScope, TimeRange
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import InvalidArgumentError, NotInitializedError
from openviking_cli.utils import get_logger

if TYPE_CHECKING:
    from openviking.session.compressor_v3 import SessionCompressorV3
    from openviking.storage.vikingdb_manager import VikingDBManager

logger = get_logger(__name__)

DEFAULT_TRAJECTORY_PAGE_LIMIT = 50
MAX_TRAJECTORY_PAGE_LIMIT = 1000

_TRAJECTORY_OUTPUT_FIELDS = [
    "uri",
    "name",
    "description",
    "created_at",
    "updated_at",
]


def _trajectory_created_at_range(
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[TimeRange]:
    """Build a UTC, end-date-inclusive filter over trajectory creation time."""
    normalized_start = (start_date or "").strip()
    normalized_end = (end_date or "").strip()
    if not normalized_start and not normalized_end:
        return None

    def parse_date(value: str, field: str) -> date:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise InvalidArgumentError(f"{field} must be a valid YYYY-MM-DD date") from exc
        if parsed.isoformat() != value:
            raise InvalidArgumentError(f"{field} must use YYYY-MM-DD format")
        return parsed

    start = parse_date(normalized_start, "start_date") if normalized_start else None
    end = parse_date(normalized_end, "end_date") if normalized_end else None
    if start is not None and end is not None and start > end:
        raise InvalidArgumentError("start_date must be earlier than or equal to end_date")

    start_time = (
        datetime.combine(start, time.min, tzinfo=timezone.utc).isoformat()
        if start is not None
        else None
    )
    # TimeRange uses an exclusive upper bound. Advancing one day keeps end_date inclusive.
    end_time = (
        datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc).isoformat()
        if end is not None
        else None
    )
    return TimeRange("created_at", start=start_time, end=end_time)


def _experience_trajectory_conditions(
    *,
    trajectory_root: str,
    experience_uri: str,
    created_at_range: Optional[TimeRange],
) -> list[Any]:
    conditions: list[Any] = [
        PathScope("uri", trajectory_root, depth=1),
        Eq("context_type", "memory"),
        Eq("level", 2),
        Eq("search_tags", experience_source_tag(experience_uri)),
    ]
    if created_at_range is not None:
        conditions.append(created_at_range)
    return conditions


class AgentEvolutionService:
    """Serve lineage queries and coordinate explicit Experience learning batches."""

    def __init__(
        self,
        viking_fs: Optional[VikingFS] = None,
        vikingdb: Optional[VikingDBManager] = None,
        session_compressor: Optional[SessionCompressorV3] = None,
    ):
        self._viking_fs = viking_fs
        self._vikingdb = vikingdb
        self._session_compressor = session_compressor
        self._learning_tasks: dict[asyncio.Task[None], tuple[str, dict[str, str]]] = {}

    def set_dependencies(
        self,
        *,
        viking_fs: VikingFS,
        vikingdb: VikingDBManager,
        session_compressor: SessionCompressorV3,
    ) -> None:
        self._viking_fs = viking_fs
        self._vikingdb = vikingdb
        self._session_compressor = session_compressor

    def _ensure_initialized(self) -> VikingFS:
        if self._viking_fs is None:
            raise NotInitializedError("VikingFS")
        return self._viking_fs

    async def learn_experiences(
        self,
        *,
        ctx: RequestContext,
        trajectory_uris: list[str] | None = None,
        case_names: list[str] | None = None,
        max_cases: int = 100,
        retry_failed: bool = False,
        lease_seconds: int = 900,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Consume Case inboxes selected by canonical names or trigger Trajectory URIs."""
        self._ensure_initialized()
        if self._session_compressor is None:
            raise NotInitializedError("SessionCompressorV3")
        kwargs = {
            "ctx": ctx,
            "trajectory_uris": trajectory_uris,
            "case_names": case_names,
            "max_cases": max_cases,
            "retry_failed": retry_failed,
            "lease_seconds": lease_seconds,
        }
        if wait:
            return await self._session_compressor.learn_pending_experiences(**kwargs)

        tracker = get_task_tracker()
        owner = {"account_id": ctx.account_id, "user_id": ctx.user.user_id}
        task = await tracker.create("experience_learn", **owner)

        async def on_progress(progress: dict[str, int]) -> None:
            await tracker.update_stage(
                task.task_id, "experience_learn", meta={"progress": progress}, **owner
            )

        async def run() -> None:
            try:
                await tracker.start(task.task_id, stage="queued", **owner)
                result = await self._session_compressor.learn_pending_experiences(
                    **kwargs,
                    on_progress=on_progress,
                )
                await tracker.complete(task.task_id, result=result, **owner)
            except asyncio.CancelledError:
                await tracker.mark_cancelled(task.task_id, **owner)
                raise
            except Exception as exc:
                await tracker.fail(task.task_id, error=f"{type(exc).__name__}: {exc}", **owner)

        background = asyncio.create_task(run())
        self._learning_tasks[background] = (task.task_id, owner)

        def done(finished: asyncio.Task[None]) -> None:
            self._learning_tasks.pop(finished, None)
            if not finished.cancelled() and (error := finished.exception()) is not None:
                logger.error(
                    "Experience learning task %s could not persist its outcome: %s",
                    task.task_id,
                    error,
                )

        background.add_done_callback(done)
        return {"task_id": task.task_id}

    async def close_background_tasks(self) -> None:
        """Join learning workers before their storage/model dependencies are closed."""
        tasks = dict(self._learning_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for background, (task_id, owner) in tasks.items():
            if background.cancelled():
                # Cancellation before run() first executes cannot enter its handler.
                await get_task_tracker().mark_cancelled(task_id, **owner)
        self._learning_tasks.clear()

    async def _prepare_experience_query(
        self,
        *,
        experience_uri: str,
        ctx: RequestContext,
    ) -> tuple[str, str, VikingDBManager]:
        canonical_uri = canonical_experience_uri(experience_uri, ctx)
        if canonical_uri is None:
            raise InvalidArgumentError(
                "experience_uri must identify an Experience owned by the current user"
            )

        viking_fs = self._ensure_initialized()
        stat = await viking_fs.stat(canonical_uri, ctx=ctx, skip_count=True)
        if stat.get("isDir", False):
            raise InvalidArgumentError("experience_uri must identify an Experience file")

        if self._vikingdb is None:
            raise NotInitializedError("VikingDB")

        trajectory_root = f"viking://user/{ctx.user.user_id}/memories/trajectories"
        return canonical_uri, trajectory_root, self._vikingdb

    async def list_trajectories_by_experience(
        self,
        *,
        experience_uri: str,
        ctx: RequestContext,
        limit: int = DEFAULT_TRAJECTORY_PAGE_LIMIT,
        offset: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """List trajectories produced by commits that read an Experience."""
        if limit < 1 or limit > MAX_TRAJECTORY_PAGE_LIMIT:
            raise InvalidArgumentError(f"limit must be between 1 and {MAX_TRAJECTORY_PAGE_LIMIT}")
        if offset < 0:
            raise InvalidArgumentError("offset must be greater than or equal to 0")
        created_at_range = _trajectory_created_at_range(start_date, end_date)

        canonical_uri, trajectory_root, vikingdb = await self._prepare_experience_query(
            experience_uri=experience_uri,
            ctx=ctx,
        )
        lineage_filter = And(
            _experience_trajectory_conditions(
                trajectory_root=trajectory_root,
                experience_uri=canonical_uri,
                created_at_range=created_at_range,
            )
        )
        records, total = await asyncio.gather(
            vikingdb.filter(
                filter=lineage_filter,
                limit=limit,
                offset=offset,
                output_fields=_TRAJECTORY_OUTPUT_FIELDS,
                order_by="updated_at",
                order_desc=True,
                ctx=ctx,
            ),
            vikingdb.count(filter=lineage_filter, ctx=ctx),
        )
        items = [
            {field: record.get(field) for field in _TRAJECTORY_OUTPUT_FIELDS if field in record}
            for record in records
        ]
        return {
            "experience_uri": canonical_uri,
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(items) < total,
        }

    async def get_experience_outcome_distribution(
        self,
        *,
        experience_uri: str,
        ctx: RequestContext,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """Count application trajectories by outcome for an Experience."""
        created_at_range = _trajectory_created_at_range(start_date, end_date)
        canonical_uri, trajectory_root, vikingdb = await self._prepare_experience_query(
            experience_uri=experience_uri,
            ctx=ctx,
        )
        base_conditions = _experience_trajectory_conditions(
            trajectory_root=trajectory_root,
            experience_uri=canonical_uri,
            created_at_range=created_at_range,
        )
        counts = await asyncio.gather(
            *[
                vikingdb.count(
                    filter=And(
                        [
                            *base_conditions,
                            Eq("search_tags", trajectory_outcome_tag(outcome)),
                        ]
                    ),
                    ctx=ctx,
                )
                for outcome in TRAJECTORY_OUTCOMES
            ]
        )
        return {
            "experience_uri": canonical_uri,
            "outcome_distribution": [
                {"outcome": outcome, "count": count}
                for outcome, count in zip(TRAJECTORY_OUTCOMES, counts, strict=True)
            ],
        }
