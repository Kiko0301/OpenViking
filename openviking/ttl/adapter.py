# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Optional commercial TTL contract. OSS provides wiring, never Redis/product behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncContextManager,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

if TYPE_CHECKING:
    from openviking.server.identity import RequestContext


@dataclass(frozen=True)
class TTLBackfillRequest:
    """Fixed initial scan: reset=False resumes the saved scope/checkpoint.

    reset=True stops the scan, clears done, replaces roots/checkpoint and
    registers pending; indexes survive. After a lost response, inspect state
    and request-id logs before retrying. Confirmed resets resume with False.
    """

    resource_id: str
    account_id: str
    roots: tuple[str, ...]
    #: True for explicit internal repair, (re)opening or widening scope; the adapter
    #: must clear ``done`` / reset the checkpoint before re-running initial.
    reset: bool = False


@dataclass(frozen=True)
class TTLRecycleRequest:
    """Server-internal hourly request to recycle one scope's due directories.

    The Server passes the most conservative (shortest) effective TTL per index
    type so the adapter can query candidates cheaply; the adapter re-resolves
    the per-directory effective policy before deleting and skips ``disabled``
    directories and any not yet due.
    """

    resource_id: str
    account_id: str
    #: Upper bound on the base-time score to scan, in epoch milliseconds.
    max_base_time_ms: int
    limit: int = 256


@dataclass(frozen=True)
class TTLRecycleResult:
    """Outcome of a single recycle sweep, for Server-side logging/metrics."""

    scanned: int = 0
    deleted: int = 0
    skipped: int = 0
    failed: int = 0


@runtime_checkable
class TTLAdapter(Protocol):
    """Commercial directory-TTL integration contract (default absent)."""

    def bind_runtime(self, service: Any, account_manager: Any) -> None:
        """Reuse the service's storage and published configuration lifecycle."""

    def write_guard(
        self, uri: str, ctx: Any, lease_ref: Any = None, *, require_exists: bool = False
    ) -> AsyncContextManager:
        """Protect the final file/vector commit with the existing path lock."""

    def source_guard(
        self, source_uri: str, ctx: Any, lease_ref: Any = None, *, task_id: Optional[str] = None
    ) -> AsyncContextManager:
        """Check a delayed commit's existing source and task owner under its lock."""

    async def requires_existing_session(self, uri: str, ctx: Any) -> bool:
        """Whether an append must reject a missing live-message file."""

    async def event_filter(self, ctx: Any) -> Any:
        """Optional predicate composed before vector ranking and limits."""

    def supports_config(self) -> bool:
        """Return True when explicit TTL configuration writes are allowed.

        The admin configuration routes keep returning 501 until a loaded
        adapter reports support, so the open-source edition stays locked.
        """

    async def on_post_write(
        self,
        uri: str,
        ctx: Optional["RequestContext"],
        written_at_ms: int,
    ) -> None:
        """Handle one committed write: register the owning directory in Redis.

        Runs after the storage write is committed and must never turn a
        successful write into a failure; the kernel already swallows and logs
        exceptions raised here.
        """

    async def register_initial_backfill(self, request: TTLBackfillRequest) -> None:
        """Register (or reset-and-resume) the fixed initial scan for a scope."""

    async def recycle_due(self, request: TTLRecycleRequest) -> TTLRecycleResult:
        """Recycle the due directories for one scope under the directory lock."""

    async def resolve_expires_at(
        self,
        uris: Sequence[str],
        ctx: Optional["RequestContext"],
    ) -> Mapping[str, Optional[str]]:
        """Batch-compute UTC expiry or None; never persist it or stat for display.

        Missing scores/Redis failure yield None without failing the read.
        None alone does not imply TTL disabled or permanent retention.
        """

    async def start(self) -> None:
        """Start background consumers/connections after the service is ready."""

    async def aclose(self) -> None:
        """Stop in-flight work and release Redis connections on shutdown."""


__all__ = [
    "TTLAdapter",
    "TTLBackfillRequest",
    "TTLRecycleRequest",
    "TTLRecycleResult",
]
