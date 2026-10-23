from __future__ import annotations

import pytest

from benchmark.locomo.vikingbot.range_padding import expand_event_ranges, parse_ranges
from openviking.message import Message, TextPart
from openviking.session.memory.memory_updater import ExtractContext


def messages(count):
    return [
        Message(id=f"m{i}", role="user", parts=[TextPart(text=f"Message {i}.")])
        for i in range(count)
    ]


@pytest.mark.parametrize(
    ("ranges", "expected"),
    [
        ("3", "1-5"),
        ("0", "0-2"),
        ("10", "8-10"),
        ("0-10", "0-10"),
        ("1,9", "0-3,7-10"),
        ("2,5", "0-7"),
        ("2-4,3-5", "0-7"),
        ("9,1,1", "0-3,7-10"),
    ],
)
def test_span_padding_clips_and_unions_within_event(ranges, expected):
    source = messages(11)
    context = ExtractContext(source)
    actual = expand_event_ranges(context, source, ranges)
    assert actual == expected
    assert parse_ranges(ranges, 11) <= parse_ranges(actual, 11)


@pytest.mark.parametrize("padding", [1, 2])
def test_padding_counts_original_messages_not_chunks(padding):
    source = messages(8)
    source[2].parts = [TextPart(text="A sentence with lots of source context. " * 20)]
    source[3].parts = [TextPart(text="Another long original message about the event. " * 20)]
    context = ExtractContext(source)

    def source_id(m):
        chunk = context.chunk_meta.get(id(m))
        return chunk.source_message_id if chunk is not None else m.id

    source3_units = [i for i, m in enumerate(context.messages) if source_id(m) == "m3"]
    assert len(source3_units) > 2
    selected = source3_units[1]
    expanded = parse_ranges(
        expand_event_ranges(context, source, str(selected), padding=padding), len(context.messages)
    )
    expected_ids = {f"m{i}" for i in range(3 - padding, 3 + padding + 1)}
    expected = {i for i, m in enumerate(context.messages) if source_id(m) in expected_ids}
    assert expanded == expected
    assert len(expanded) > 5
    assert parse_ranges(
        expand_event_ranges(context, source, str(selected), padding=0), len(context.messages)
    ) == {selected}


@pytest.mark.parametrize("ranges", ["", "bad", "-1", "11", "0-11", "3-1", "1,", "1.5"])
def test_invalid_input_is_not_hidden_by_clipping(ranges):
    source = messages(11)
    with pytest.raises(ValueError):
        expand_event_ranges(ExtractContext(source), source, ranges)


@pytest.mark.parametrize("padding", [-1, True, 1.5])
def test_invalid_padding(padding):
    source = messages(3)
    with pytest.raises(ValueError, match="padding"):
        expand_event_ranges(ExtractContext(source), source, "1", padding=padding)


def test_large_padding_never_crosses_session_bounds():
    source = messages(3)
    assert expand_event_ranges(ExtractContext(source), source, "1", padding=100) == "0-2"


def test_missing_source_mapping_is_rejected():
    source = messages(3)
    with pytest.raises(ValueError, match="no original source"):
        expand_event_ranges(ExtractContext(source), source[:2], "1")


@pytest.mark.parametrize(
    ("ranges", "expected"),
    [
        ("3", "2-4"),
        ("0", "0-1"),
        ("10", "9-10"),
        ("0-10", "0-10"),
        ("1,9", "0-2,8-10"),
        ("2,4", "1-5"),
        ("2,5", "1-6"),
        ("2-4", "1-5"),
    ],
)
def test_one_message_padding(ranges, expected):
    source = messages(11)
    context = ExtractContext(source)
    one = expand_event_ranges(context, source, ranges, padding=1)
    two = expand_event_ranges(context, source, ranges, padding=2)
    assert one == expected
    assert parse_ranges(ranges, 11) <= parse_ranges(one, 11) <= parse_ranges(two, 11)
