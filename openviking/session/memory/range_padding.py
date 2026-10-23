# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deterministic event-range expansion by original conversation messages."""

from __future__ import annotations

import re
from collections.abc import Iterable

from openviking.message import Message
from openviking.session.memory.memory_updater import ExtractContext


def parse_ranges(ranges: str, message_count: int) -> set[int]:
    """Reject malformed/out-of-bounds inputs before any clipping or expansion."""
    selected: set[int] = set()
    for part in ranges.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if match is None:
            raise ValueError(f"Invalid source ranges: {ranges!r}")
        start = int(match[1])
        end = int(match[2]) if match[2] is not None else start
        if not 0 <= start <= end < message_count:
            raise ValueError(f"Source ranges out of bounds: {ranges!r}")
        selected.update(range(start, end + 1))
    return selected


def _format_ranges(indices: Iterable[int]) -> str:
    ordered = sorted(set(indices))
    spans: list[str] = []
    start = end = ordered[0]
    for index in ordered[1:]:
        if index == end + 1:
            end = index
        else:
            spans.append(str(start) if start == end else f"{start}-{end}")
            start = end = index
    spans.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(spans)


def expand_event_ranges(
    context: ExtractContext,
    source_messages: list[Message],
    ranges: str,
    *,
    padding: int = 2,
) -> str:
    """Expand each selected span by original messages, not extraction chunks.

    Each original message touched by a selected chunk and its neighboring messages
    are retained in full. Spans are unioned only within this event; caller-owned
    event identities remain separate even if their padded selections overlap.
    """
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be a nonnegative integer")
    selected = parse_ranges(ranges, len(context.messages))
    if padding == 0:
        return _format_ranges(selected)
    source_positions = {message.id: i for i, message in enumerate(source_messages)}
    if len(source_positions) != len(source_messages):
        raise ValueError("Original message IDs must be unique")
    unit_sources = []
    for message in context.messages:
        chunk = context.chunk_meta.get(id(message))
        source_id = chunk.source_message_id if chunk is not None else message.id
        if source_id not in source_positions:
            raise ValueError(f"Extraction unit has no original source message: {source_id}")
        unit_sources.append(source_positions[source_id])
    if unit_sources != sorted(unit_sources) or set(unit_sources) != set(
        range(len(source_messages))
    ):
        raise ValueError("Extraction units must cover the original messages in order")
    expanded_sources = {
        source_index
        for unit_index in selected
        for source_index in range(
            max(0, unit_sources[unit_index] - padding),
            min(len(source_messages), unit_sources[unit_index] + padding + 1),
        )
    }
    return _format_ranges(i for i, source in enumerate(unit_sources) if source in expanded_sources)
