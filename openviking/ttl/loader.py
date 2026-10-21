# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Load the commercial directory-TTL adapter from ``class_path`` config.

Mirrors the usage-reporter sink loader (:mod:`openviking.usage_reporter.config`)
so a commercial deployment opts in with::

    "server": {"ttl_adapter": {"class_path": "pkg.mod.ClassName", "config": {...}}}

The open-source edition leaves ``ttl_adapter`` unset, so :func:`build_ttl_adapter`
returns ``None`` and TTL stays unsupported. The loader only imports the
commercial module when a ``class_path`` is configured, so the open-source server
never takes a Redis dependency.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Optional

from openviking_cli.utils.logger import get_logger

if TYPE_CHECKING:
    from openviking.server.config import TTLAdapterConfig
    from openviking.ttl.adapter import TTLAdapter

logger = get_logger(__name__)


def _load_class(class_path: str) -> type:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        raise TypeError(f"{class_path} does not resolve to a class")
    return cls


def build_ttl_adapter(config: Optional["TTLAdapterConfig"]) -> Optional["TTLAdapter"]:
    """Construct the configured TTL adapter, or ``None`` when TTL is off.

    Returns ``None`` when no adapter is configured (open-source default) so the
    caller leaves the write hook uninstalled and the 501 gate in place.
    """
    if config is None or not config.class_path:
        return None

    cls = _load_class(config.class_path)
    kwargs: dict[str, Any] = dict(config.config)
    adapter = cls(**kwargs)
    logger.info("Loaded commercial TTL adapter %s", config.class_path)
    return adapter


__all__ = ["build_ttl_adapter"]
