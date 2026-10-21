# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Stable contract for the commercial directory-TTL integration layer.

The open-source kernel only *calls* this interface; it never ships a concrete
implementation that touches Redis or performs recycling. A commercial
deployment supplies a class via ``server.ttl_adapter.class_path`` (see
:mod:`openviking.ttl.loader`). When no adapter is configured TTL stays
unsupported: the write hook is never installed and explicit TTL configuration
is rejected with 501 / ``UNIMPLEMENTED``.

The method set mirrors the three responsibilities the technical design assigns
to the commercial OV integration layer:

* ``on_post_write`` — interpret the kernel's post-write notification and ZADD
  the owning directory into the right Redis index (incremental registration).
* ``register_initial_backfill`` — the Server's internal trigger to register and
  resume the single fixed ``initial`` historical scan for one scope.
* ``recycle_due`` — the Server's hourly internal trigger to recycle the due
  candidates for one scope under the directory lock.
* ``resolve_expires_at`` — the read-path echo-back: given a page of URIs it
  returns each directory's computed ``expires_at`` (or ``None``) so the kernel
  can attach it to ``stat`` / ``ls`` / ``tree`` entries. The value is computed
  from the current effective policy (reusing :mod:`openviking.config.ttl`) and
  never persisted.

Registration, backfill and recycling are all driven by the commercial adapter;
the kernel never calls Redis itself. The bodies for ``register_initial_backfill``
and ``recycle_due`` are filled in by the commercial adapter task; this module
only pins the shared shapes so the loader, the write hook and the Server-facing
internal routes can be wired independently.
"""

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
    """Server-internal request to register/resume the fixed initial scan.

    ``roots`` is the set of canonical directory URIs the Server has decided the
    scope must cover (all TTL-governed Events and Session trees). The adapter
    initializes only missing HASH entries and resumes the stored checkpoint
    when ``reset=False``. A completed task with unchanged roots is a no-op;
    changed roots without reset must fail with CONFLICT.

    With ``reset=True``, serialize by scope, stop the old scan and wait for
    in-flight I/O, clear done, replace checkpoint and roots, then add initial
    to pending. Keep directory indexes and incremental registration intact;
    pause physical recycling until this round writes done=1 again. Return
    only after registration is confirmed. A timeout leaves the result unknown:
    inspect Redis and request-id logs before retrying, using reset=False once
    registration is confirmed. Pod recovery never resets an existing task.
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
        """Compute the read-only ``expires_at`` echo-back for a page of URIs.

        Returns a mapping from each input URI to a UTC ISO-8601 string (e.g.
        ``"2030-01-31T00:00:00Z"``) or ``None``. ``None`` means "no available
        expiry this time" — TTL disabled for the directory, a URI outside the
        governed subtrees, a config root / navigation directory, or a missing /
        failed Redis registration lookup. It does **not** imply TTL is off.

        The value is computed from the current effective policy and (for
        Sessions) the already-registered Redis base time; it is never stored to
        object metadata or Redis. The adapter batches Redis reads per page under
        a short timeout and must not add AGFS ``ls``/``stat`` I/O or fall back to
        it on failure — a failed lookup yields ``None`` for that URI without
        disturbing the surrounding listing/stat result.
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
