# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Use the production range expansion for controlled frozen-memory evaluations."""

from __future__ import annotations

from openviking.session.memory.range_padding import expand_event_ranges, parse_ranges

__all__ = ["expand_event_ranges", "parse_ranges"]
