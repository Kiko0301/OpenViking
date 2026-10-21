# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Process-wide registry and service wiring for the TTL adapter.

The adapter is a singleton per server process. ``app.py`` builds it from
``server.ttl_adapter`` config, registers it here, and installs its post-write
hook on the VikingFS instance. The admin configuration routes consult
:func:`ttl_config_supported` to decide whether an explicit TTL section is
allowed or must be rejected with 501. When no adapter is registered
(open-source default) every TTL entry point is inert.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from openviking_cli.utils.logger import get_logger

if TYPE_CHECKING:
    from openviking.server.identity import RequestContext
    from openviking.storage.viking_fs import VikingFS
    from openviking.ttl.adapter import TTLAdapter

logger = get_logger(__name__)

_adapter: Optional["TTLAdapter"] = None


def build_ttl_adapter(config):
    """Import the configured commercial class only when explicitly enabled."""
    if config is None or not config.class_path:
        return None
    import importlib

    module, name = config.class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    if not isinstance(cls, type):
        raise TypeError(f"{config.class_path} does not resolve to a class")
    return cls(**config.config)


def set_ttl_adapter(adapter: Optional["TTLAdapter"]) -> None:
    """Register (or clear) the process-wide TTL adapter."""
    global _adapter
    _adapter = adapter


def get_ttl_adapter() -> Optional["TTLAdapter"]:
    """Return the registered TTL adapter, or ``None`` when TTL is off."""
    return _adapter


def ttl_config_supported() -> bool:
    """Return True when explicit TTL configuration writes are allowed.

    False in the open-source edition (no adapter) so the admin routes keep
    returning 501 / ``UNIMPLEMENTED`` for an explicit TTL section.
    """
    adapter = _adapter
    if adapter is None:
        return False
    try:
        return bool(adapter.supports_config())
    except Exception:  # noqa: BLE001 - a misbehaving adapter must not open the gate
        logger.exception("TTL adapter supports_config() failed; treating TTL as unsupported")
        return False


def install_ttl_hook(viking_fs: "VikingFS", adapter: Optional["TTLAdapter"]) -> None:
    """Install the adapter's post-write hook, or leave the default-off hook.

    Called once after the VikingFS instance exists. When no adapter is present
    the hook stays unset, so a committed write never reads the clock or does any
    TTL I/O (open-source default-off behavior).
    """
    if adapter is None:
        return
    viking_fs.set_post_write_hook(adapter.on_post_write)
    viking_fs._write_guard = getattr(adapter, "write_guard", None)
    logger.info("Installed TTL post-write hook")


def _entry_uri(entry: Any) -> Optional[str]:
    if isinstance(entry, Mapping):
        uri = entry.get("uri")
        return uri if isinstance(uri, str) and uri else None
    return None


async def attach_expires_at(
    entries: Sequence[Any],
    ctx: Optional["RequestContext"],
) -> None:
    """Default-off read-path ``expires_at`` echo-back (see the technical design).

    Attaches ``expires_at`` in place to each dict entry that carries a ``uri``;
    a single ``stat`` result is just a one-element sequence. No-op in the
    open-source edition (no adapter), so the read path does zero TTL I/O. When a
    commercial adapter is registered it computes the value from the current
    effective policy; the field is only written for URIs the adapter actually
    covers (value string or ``None``), so the kernel never fabricates a field
    the product layer did not compute. Any adapter failure degrades to "no echo
    this page" instead of disturbing the underlying listing/stat result.
    """
    adapter = _adapter
    if adapter is None or not entries:
        return
    uris = list(dict.fromkeys(u for u in (_entry_uri(e) for e in entries) if u is not None))
    if not uris:
        return
    try:
        resolved = await adapter.resolve_expires_at(uris, ctx)
    except Exception:  # noqa: BLE001 - echo-back must never break stat/ls/tree
        logger.exception("TTL adapter resolve_expires_at() failed; omitting expires_at echo")
        return
    for entry in entries:
        uri = _entry_uri(entry)
        if uri is not None and uri in resolved:
            entry["expires_at"] = resolved[uri]


__all__ = [
    "attach_expires_at",
    "get_ttl_adapter",
    "install_ttl_hook",
    "set_ttl_adapter",
    "ttl_config_supported",
]
