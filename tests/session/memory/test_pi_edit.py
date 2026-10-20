# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Regression vectors for Pi 482c1381 Edit, including its normalization boundaries."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation, ResolvedOperations
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.extraction_output_protocol import create_extraction_output_protocol
from openviking.session.memory.merge_op import (
    DeleteBlock,
    FieldType,
    PatchOp,
    PatchParseError,
    SearchReplaceBlock,
    StrPatch,
    apply_str_patch,
)
from openviking.session.memory.merge_op.patch_handler import normalize_for_fuzzy_match
from tests.session.memory.test_extraction_output_protocol import (
    _bind,
    _context,
    _existing_preference,
    _preference_schema,
)


def edits(*pairs: tuple[str, str]) -> StrPatch:
    return StrPatch(blocks=[SearchReplaceBlock(search=old, replace=new) for old, new in pairs])


@pytest.mark.parametrize(
    ("source", "old", "new", "expected"),
    [
        ("alpha\nbeta", "alpha", "A", "A\nbeta"),
        ("say(“yes”)\nkeep，line", 'say("yes")', 'say("no")', 'say("no")\nkeep，line'),
        ("你好，世界\n", "你好,世界", "更新", "更新\n"),
        ("① 𝒜 １", "1 A 1", "done", "done"),
        ("x—y", "x-y", "done", "done"),
        ("a\u00a0b", "a b", "done", "done"),
        ("first   \nsecond  \nkeep   \n", "first\nsecond\n", "changed\n", "changed\nkeep   \n"),
        ("\ufefffirst\r\nsecond\r\n", "first\nsecond", "A\nB", "\ufeffA\r\nB\r\n"),
        ("😀 “old”\nkeep", '😀 "old"', "changed", "changed\nkeep"),
        ("a\rb\r", "a\nb", "A\nB", "A\nB\n"),
    ],
)
def test_pi_exact_and_normalized_replacements(source, old, new, expected):
    assert apply_str_patch(source, edits((old, new))) == expected


@pytest.mark.parametrize(
    ("source", "old"),
    [
        ("hello world", "hello worle"),
        ("alpha\nbeta\nalpha\nbeta", "alpha\nbetx"),
        ('tell(\n    "text"\n)', 'tell("text")'),
        ("one\ntwo", "2\ttwo"),
        ("hello world", "hello  world"),
    ],
)
def test_pi_does_not_guess_similar_blocks_or_strip_line_numbers(source, old):
    with pytest.raises(PatchParseError, match="Could not find the exact text"):
        apply_str_patch(source, edits((old, "changed")))


def test_pi_matches_all_edits_against_original_and_applies_backwards():
    assert apply_str_patch("alpha\nbeta\nkeep", edits(("alpha", "beta"), ("beta", "gamma"))) == (
        "beta\ngamma\nkeep"
    )
    assert apply_str_patch("alpha\nbeta", edits(("beta", "B"), ("alpha", "A"))) == "A\nB"


def test_pi_rejects_a_later_edit_that_only_matches_generated_text():
    with pytest.raises(PatchParseError, match=r"Could not find edits\[1\]") as caught:
        apply_str_patch("alpha", edits(("alpha", "beta"), ("beta", "gamma")), path="uri#content")
    assert caught.value.edit_index == 1
    assert "uri#content" in str(caught.value)


def test_pi_rejects_overlapping_and_nested_regions():
    with pytest.raises(PatchParseError, match=r"edits\[0\] and edits\[1\] overlap"):
        apply_str_patch("abcdef", edits(("abcd", "A"), ("bc", "B")))


def test_pi_rejects_normalized_ambiguity_even_if_one_exact_match_exists():
    with pytest.raises(PatchParseError, match="Found 2 occurrences"):
        apply_str_patch('“yes”\n"yes"', edits(('"yes"', "new")))


@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.asyncio
async def test_pi_batch_failure_never_returns_a_partial_result(checked):
    merge = PatchOp(FieldType.STRING)
    apply = merge.apply_checked if checked else merge.apply
    with pytest.raises(PatchParseError, match=r"Could not find edits\[1\]"):
        await apply("alpha\nbeta", edits(("alpha", "A"), ("missing", "B")))


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        (StrPatch(), "at least one replacement"),
        (edits(("", "new")), "search must not be empty"),
        (edits(("alpha", "alpha")), "No changes made"),
    ],
)
def test_pi_rejects_empty_or_no_change_edits(patch, message):
    with pytest.raises(PatchParseError, match=message):
        apply_str_patch("alpha", patch)


def test_pi_does_not_expand_delete_to_whole_lines():
    assert apply_str_patch("a\nb\nc", StrPatch(blocks=[DeleteBlock(delete="b")])) == "a\n\nc"
    assert apply_str_patch("a\nb\nc", StrPatch(blocks=[DeleteBlock(delete="b\n")])) == "a\nc"


def test_pi_preserves_untouched_duplicate_lines_after_fuzzy_edit():
    assert apply_str_patch("replace me   \nafter   \n", edits(("replace me\n", "after\n"))) == (
        "after\nafter   \n"
    )


def test_pi_normalizes_touched_lines_not_only_the_matched_substring():
    source = 'message = "请确认 “yes”，备注“保留”"\nkeep = "甲，乙"\n'
    result = apply_str_patch(source, edits(('请确认 "yes"', "请确认 “no”")))
    assert result == 'message = "请确认 “no”,备注"保留""\nkeep = "甲，乙"\n'
    # Deliberately preserve upstream behavior: the type-owned gate must reject this,
    # rather than silently installing our own normalization-span algorithm here.
    with pytest.raises(SyntaxError):
        ast.parse(result)


def test_pi_uses_ecmascript_trim_end_not_python_whitespace_rules():
    assert normalize_for_fuzzy_match("a\u0085") == "a\u0085"
    assert normalize_for_fuzzy_match("a\ufeff") == "a"


@pytest.mark.asyncio
async def test_sdk_chained_edits_are_one_original_snapshot_batch():
    uri = "viking://user/alice/memories/preferences/editor.md"
    source = "alpha\nbeta\nkeep"
    context = _context([_preference_schema()], files=[_existing_preference(uri, "editor", source)])
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    operations, error = protocol.parse(
        "preferences_1.content.edit(search='alpha', replace='beta')"
        ".edit(search='beta', replace='gamma')\nsdk.commit()",
        context,
    )
    assert error is None
    change = operations.model_dump()["preferences"][0]["content"]
    assert await PatchOp(FieldType.STRING).apply_checked(source, change) == "beta\ngamma\nkeep"


@pytest.mark.asyncio
async def test_extraction_preview_uses_the_target_field_and_pi_batch_errors():
    uri = "viking://user/alice/memories/cases/case.md"
    memory = MemoryFile(
        uri=uri, content="not the description", extra_fields={"description": "alpha\nbeta"}
    )
    operation = ResolvedOperation(
        memory_type="cases",
        uris=[uri],
        old_memory_file_content=memory,
        memory_fields={"description": edits(("alpha", "beta"), ("beta", "gamma"))},
    )
    loop = ExtractLoop.__new__(ExtractLoop)
    loop.context_provider = SimpleNamespace(read_file_contents={uri: memory})
    batch = ResolvedOperations(upsert_operations=[operation], delete_file_contents=[], errors=[])
    assert await loop._validate_patch_operations(batch) == []
    operation.memory_fields["description"] = edits(("alpha", "beta"), ("alpha", "gamma"))
    errors = await loop._validate_patch_operations(batch)
    assert len(errors) == 1
    assert errors[0]["reason"] == "overlap"
    assert errors[0]["field"] == "description"
    assert f"{uri}#description" in errors[0]["error"]


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        ([("", "new")], "search must not be empty"),
        ([("beta", "B"), ("", "new")], "edits[1].search must not be empty"),
        ([("missing", "new")], "The search must match exactly"),
        ([("beta", "B"), ("missing", "new")], "The search must match exactly"),
        ([("alpha", "new")], "Found 2 occurrences"),
        ([("beta", "B"), ("alpha", "new")], "Each search must be unique"),
    ],
)
def test_pi_error_and_retry_instructions_use_sdk_parameter_names(pairs, expected):
    with pytest.raises(PatchParseError) as caught:
        apply_str_patch("alpha\nalpha\nbeta", edits(*pairs), path="memory#content")
    message = str(caught.value)
    assert expected in message
    assert "memory#content" in message
    assert "old_text" not in message and "new_text" not in message
    assert "old text" not in message
    for output_format in ("python", "json"):
        protocol = create_extraction_output_protocol(output_format)
        repair = protocol.render_patch_repair([{"field": "content", "error": message}])
        assert '"old_text"' not in repair and '"new_text"' not in repair
