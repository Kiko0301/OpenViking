# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Native TTL initialization preserves the default-off and reader boundaries."""

import sys

import pytest

from openviking.server.config import TTLAdapterConfig
from openviking.ttl.wiring import build_ttl_adapter


@pytest.mark.parametrize("config", [None, TTLAdapterConfig(), TTLAdapterConfig(enabled=False)])
def test_disabled_does_not_load_runtime(monkeypatch, config):
    monkeypatch.setitem(sys.modules, "openviking.ttl.runtime", None)
    assert build_ttl_adapter(config) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", [True, False])
async def test_native_lifecycle_is_lazy_and_reader_has_no_worker(writer):
    from openviking.ttl.runtime import DirectoryTTL

    engine = build_ttl_adapter(
        TTLAdapterConfig(enabled=True, config={"resource_id": "native-test", "writer": writer})
    )
    assert isinstance(engine, DirectoryTTL)
    try:
        await engine.start()
        assert (engine._task is not None) is writer
        assert engine.redis is None
        assert not engine._pending and not engine._retry
    finally:
        await engine.aclose()
