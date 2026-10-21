# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Effective TTL policy resolution for the directory-recycling product.

:mod:`openviking_cli.utils.config.ttl_config` is a pure data shape; this module
is the single resolver that write-time registration, retrieval filtering and
recycling all share so they agree on one policy for a target directory.

Resolution has two stages, mirroring the technical design:

1. **Per-node merge** — for each policy field, an Account (数据空间) override
   wins over the published Cluster (库/实例) value. The merge is per field, so
   ``Account.global`` never overrides a库's more specific ``sessions`` value; it
   only overrides ``Cluster.global``. The Cluster value handed in here is
   already the startup-config + cluster-runtime-override merge produced by the
   runtime config manager, so this module only needs the account-over-cluster
   layer.
2. **Target selection** — for one directory URI, pick the most specific
   applicable node: a configured specific root > the matching type default
   (``user_events`` / ``peer_events`` / ``sessions``) > ``global`` > default
   off. An absent node inherits (keep looking); ``"disabled"`` stops the
   fallback with TTL off; the first concrete day count wins.

The resolver is a pure function with no I/O. TTL remains off by default: a URI
that is not a TTL-governed Events or Sessions directory, or a policy chain that
never yields a positive day count, resolves to disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from openviking.core.namespace import uri_parts
from openviking_cli.utils.config.ttl_config import TTLConfig, TTLPolicy

#: Sentinel day count meaning "inherit / not decided at this node".
_INHERIT = None
#: Explicit stop value: this node turns TTL off and ends the fallback chain.
_DISABLED = "disabled"


class TTLIndexType(str, Enum):
    """The three directory kinds that carry a TTL policy in the first phase."""

    USER_EVENTS = "user_events"
    PEER_EVENTS = "peer_events"
    SESSIONS = "sessions"


@dataclass(frozen=True)
class EffectiveTTL:
    """One directory's resolved TTL decision.

    ``index_type`` is ``None`` for a URI outside the TTL-governed subtrees;
    such a URI is never recycled. ``enabled`` is ``True`` only when a positive
    ``days`` applies; an explicit ``disabled`` and the default-off case both
    resolve to ``enabled=False`` with ``days=None``.
    """

    index_type: Optional[TTLIndexType]
    enabled: bool
    days: Optional[int]

    @classmethod
    def off(cls, index_type: Optional[TTLIndexType]) -> "EffectiveTTL":
        return cls(index_type=index_type, enabled=False, days=None)


def classify_ttl_uri(uri: str) -> Optional[TTLIndexType]:
    """Classify a target URI into its TTL index type, or ``None`` when ungoverned.

    Any directory within the subtree classifies as its root's type, so a date
    directory under Events or any node under a Session root is covered. Only the
    first-phase public-cloud subtrees (User Events, Peer Events, Sessions) are
    recognized; everything else (resources, skills, memories other than events,
    agent scope) returns ``None`` and is never recycled.
    """
    try:
        parts = uri_parts(uri)
    except ValueError:
        return None
    if len(parts) < 3 or parts[0] != "user":
        return None
    # user/{uid}/peers/{peer_id}/memories/events[/...]
    if len(parts) >= 6 and parts[2] == "peers" and parts[4] == "memories" and parts[5] == "events":
        return TTLIndexType.PEER_EVENTS
    # user/{uid}/memories/events[/...]
    if len(parts) >= 4 and parts[2] == "memories" and parts[3] == "events":
        return TTLIndexType.USER_EVENTS
    # user/{uid}/sessions[/...]
    if parts[2] == "sessions":
        return TTLIndexType.SESSIONS
    return None


def _node_value(account: Optional[TTLPolicy], cluster: Optional[TTLPolicy]) -> Optional[TTLPolicy]:
    """Account override wins per node; an unset Account field inherits Cluster."""
    return account if account is not None else cluster


def _merged_root_policy(
    account: Optional[TTLConfig], cluster: Optional[TTLConfig], target_parts: list[str]
) -> Optional[TTLPolicy]:
    """Return the most specific configured root policy covering ``target_parts``.

    Roots are matched as ancestor-or-equal URI prefixes; the longest match wins.
    An Account root override replaces a Cluster root at the same URI (per-node
    merge by key). ``None`` means no configured root applies.
    """
    merged: dict[str, TTLPolicy] = {}
    if cluster is not None and cluster.roots:
        merged.update(cluster.roots)
    if account is not None and account.roots:
        merged.update(account.roots)
    best_len = -1
    best_policy: Optional[TTLPolicy] = None
    for root_uri, policy in merged.items():
        try:
            root_parts = uri_parts(root_uri)
        except ValueError:
            continue
        if target_parts[: len(root_parts)] != root_parts:
            continue
        if len(root_parts) > best_len:
            best_len = len(root_parts)
            best_policy = policy
    return best_policy


def _decide(chain: list[Optional[TTLPolicy]], index_type: TTLIndexType) -> EffectiveTTL:
    """Walk the precedence chain: inherit past ``None``, stop at the first value."""
    for value in chain:
        if value is _INHERIT:
            continue
        if value == _DISABLED:
            return EffectiveTTL.off(index_type)
        # The config model has already validated this is a positive day count.
        return EffectiveTTL(index_type=index_type, enabled=True, days=int(value))
    return EffectiveTTL.off(index_type)


def resolve_effective_ttl(
    cluster_ttl: Optional[TTLConfig],
    account_ttl: Optional[TTLConfig],
    uri: str,
) -> EffectiveTTL:
    """Resolve the effective TTL for one directory URI.

    ``cluster_ttl`` is the published Cluster policy (already including any
    cluster runtime override); ``account_ttl`` is the Account policy or ``None``
    when the Account has no TTL section. The result is the single policy that
    write registration, retrieval filtering and recycling must all use.
    """
    index_type = classify_ttl_uri(uri)
    if index_type is None:
        return EffectiveTTL.off(None)

    target_parts = uri_parts(uri)
    root_policy = _merged_root_policy(account_ttl, cluster_ttl, target_parts)

    account_type = getattr(account_ttl, index_type.value, None) if account_ttl else None
    cluster_type = getattr(cluster_ttl, index_type.value, None) if cluster_ttl else None
    type_policy = _node_value(account_type, cluster_type)

    account_global = account_ttl.global_ if account_ttl else None
    cluster_global = cluster_ttl.global_ if cluster_ttl else None
    global_policy = _node_value(account_global, cluster_global)

    return _decide([root_policy, type_policy, global_policy], index_type)


__all__ = [
    "EffectiveTTL",
    "TTLIndexType",
    "classify_ttl_uri",
    "resolve_effective_ttl",
]
