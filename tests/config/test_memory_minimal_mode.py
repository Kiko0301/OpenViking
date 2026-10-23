from __future__ import annotations

import pytest
from pydantic import ValidationError

from openviking_cli.utils.config.memory_config import MemoryConfig


@pytest.mark.parametrize("values", [{}, {"minimal_mode": False}, {"minimal_mode": True}])
def test_minimal_mode_is_opt_in_and_round_trips(values):
    config = MemoryConfig.model_validate(values)
    assert config.minimal_mode is values.get("minimal_mode", False)
    assert config.event_range_padding == 1
    assert MemoryConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("padding", [0, 1, 2, 100])
def test_minimal_mode_accepts_nonnegative_message_padding(padding):
    config = MemoryConfig(minimal_mode=True, event_range_padding=padding)
    assert config.event_range_padding == padding


@pytest.mark.parametrize("padding", [-1, True, False, 1.5, "2", None])
def test_minimal_mode_rejects_invalid_message_padding(padding):
    with pytest.raises(ValidationError):
        MemoryConfig(minimal_mode=True, event_range_padding=padding)
