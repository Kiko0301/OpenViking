# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Tests for MergeOp architecture - type-safe merge operations.
"""

import asyncio
import threading

import pytest

from openviking.session.memory.dataclass import (
    MemoryField,
)
from openviking.session.memory.extraction_output_protocol.edit_format import EDIT_TOOL_DESCRIPTION
from openviking.session.memory.merge_op import (
    DeleteBlock,
    ImmutableOp,
    MergeOp,
    MergeOpFactory,
    PatchOp,
    PatchParseError,
    SearchReplaceBlock,
    StrPatch,
    SumOp,
    apply_str_patch,
)
from openviking.session.memory.merge_op.base import FieldType

# ============================================================================
# Test MergeOp Base Classes
# ============================================================================


class TestPatchOp:
    """Tests for PatchOp."""

    def test_get_output_schema_type_string(self):
        """String field with patch should return StrPatch."""
        op = PatchOp(FieldType.STRING)
        assert op.get_output_schema_type(FieldType.STRING) == StrPatch

    def test_get_output_schema_type_int(self):
        """Int field with patch should return int."""
        op = PatchOp(FieldType.INT64)
        assert op.get_output_schema_type(FieldType.INT64) is int

    def test_get_output_schema_type_float(self):
        """Float field with patch should return float."""
        op = PatchOp(FieldType.FLOAT32)
        assert op.get_output_schema_type(FieldType.FLOAT32) is float

    def test_get_output_schema_type_bool(self):
        """Bool field with patch should return bool."""
        op = PatchOp(FieldType.BOOL)
        assert op.get_output_schema_type(FieldType.BOOL) is bool

    def test_get_output_schema_description_string(self):
        """String field description should mention PATCH."""
        op = PatchOp(FieldType.STRING)
        desc = op.get_output_schema_description("test content")
        assert "PATCH" in desc
        assert "test content" in desc

    def test_get_output_schema_description_string_mentions_shared_edit_instructions(self):
        """String patch description should defer to the shared Edit instructions."""
        op = PatchOp(FieldType.STRING)
        desc = op.get_output_schema_description("test content")
        assert "Follow the shared Edit instructions." in desc

    def test_get_output_schema_description_string_drops_line_number_prefix_reminder(self):
        """String patch description should rely on the shared line-prefix guidance."""
        op = PatchOp(FieldType.STRING)
        desc = op.get_output_schema_description("test content")
        assert "line_number<TAB>" not in desc

    def test_get_output_schema_description_other(self):
        """Non-string field description should mention replace."""
        op = PatchOp(FieldType.INT64)
        desc = op.get_output_schema_description("score")
        assert "Replace" in desc
        assert "score" in desc

    @pytest.mark.asyncio
    async def test_apply(self):
        """PatchOp apply should just return the patch value."""
        op_str = PatchOp(FieldType.STRING)
        assert await op_str.apply("old", "new") == "new"

        op_int = PatchOp(FieldType.INT64)
        assert await op_int.apply(100, 200) == 200

    @pytest.mark.asyncio
    async def test_apply_dict_patch(self, monkeypatch):
        """Dict-form string patches should be applied without blocking the event loop."""
        from openviking.session.memory.merge_op import patch_handler

        op = PatchOp(FieldType.STRING)
        original_apply = patch_handler.apply_str_patch
        loop_progressed = threading.Event()
        merge_observed_progress = None

        def apply_and_observe(current_value, patch_value, **kwargs):
            nonlocal merge_observed_progress
            merge_observed_progress = loop_progressed.wait(timeout=0.5)
            return original_apply(current_value, patch_value, **kwargs)

        monkeypatch.setattr(patch_handler, "apply_str_patch", apply_and_observe)
        patch = {"blocks": [{"search": "hello world", "replace": "hello there"}]}

        merge_task = asyncio.create_task(op.apply("hello world", patch))
        await asyncio.sleep(0)
        loop_progressed.set()

        assert await merge_task == "hello there"
        assert merge_observed_progress is True

    @pytest.mark.asyncio
    async def test_apply_invalid_dict_patch_is_rejected(self):
        """Malformed edits must not become literal dictionary text in a field."""
        op = PatchOp(FieldType.STRING)
        patch = {"blocks": [{"search": "hello world"}]}

        with pytest.raises(ValueError):
            await op.apply("hello world", patch)

    @pytest.mark.asyncio
    async def test_apply_dict_patch_propagates_patch_parse_error(self):
        """Patch errors raised after dict conversion must reach the caller."""
        op = PatchOp(FieldType.STRING)
        patch = {"blocks": [{"search": "status: pending", "replace": "status: done"}]}

        with pytest.raises(PatchParseError, match="Found 2 occurrences"):
            await op.apply("status: pending\nstatus: pending", patch)

    @pytest.mark.asyncio
    async def test_apply_dict_delete_patch(self):
        """Dict-form DELETE blocks should remove complete lines."""
        op = PatchOp(FieldType.STRING)
        patch = {"blocks": [{"delete": "line 2\nline 3\n"}]}

        assert await op.apply("line 1\nline 2\nline 3\nline 4", patch) == "line 1\nline 4"


class TestSumOp:
    """Tests for SumOp."""

    def test_get_output_schema_type(self):
        """SumOp should return appropriate numeric types."""
        op = SumOp()
        assert op.get_output_schema_type(FieldType.INT64) is int
        assert op.get_output_schema_type(FieldType.FLOAT32) is float

    def test_get_output_schema_description(self):
        """Description should have 'add for' format."""
        op = SumOp()
        desc = op.get_output_schema_description("打分合")
        assert desc == "add for '打分合'"

    @pytest.mark.asyncio
    async def test_apply_both_int(self):
        """Sum of two ints."""
        op = SumOp()
        assert await op.apply(10, 5) == 15

    @pytest.mark.asyncio
    async def test_apply_both_float(self):
        """Sum of two floats."""
        op = SumOp()
        assert await op.apply(10.5, 3.5) == 14.0

    @pytest.mark.asyncio
    async def test_apply_mixed(self):
        """Sum of int and float."""
        op = SumOp()
        assert await op.apply(10, 3.5) == 13.5

    @pytest.mark.asyncio
    async def test_apply_current_none(self):
        """Current is None should return patch."""
        op = SumOp()
        assert await op.apply(None, 10) == 10

    @pytest.mark.asyncio
    async def test_apply_invalid_values(self):
        """Invalid values should keep current."""
        op = SumOp()
        assert await op.apply("not a number", 10) == "not a number"


class TestImmutableOp:
    """Tests for ImmutableOp."""

    def test_get_output_schema_type(self):
        """ImmutableOp should return base types."""
        op = ImmutableOp()
        assert op.get_output_schema_type(FieldType.STRING) is str
        assert op.get_output_schema_type(FieldType.INT64) is int

    def test_get_output_schema_description(self):
        """Description should mention immutable."""
        op = ImmutableOp()
        desc = op.get_output_schema_description("name")
        assert "Immutable" in desc
        assert "name" in desc
        assert "can only be set once" in desc

    @pytest.mark.asyncio
    async def test_apply_current_none(self):
        """Current is None should set to patch."""
        op = ImmutableOp()
        assert await op.apply(None, "new value") == "new value"

    @pytest.mark.asyncio
    async def test_apply_current_exists(self):
        """Current exists should keep current."""
        op = ImmutableOp()
        assert await op.apply("existing", "new value") == "existing"


class TestMergeOpFactory:
    """Tests for MergeOpFactory."""

    def test_create_patch(self):
        """Factory should create PatchOp for PATCH."""
        op = MergeOpFactory.create(MergeOp.PATCH, FieldType.STRING)
        assert isinstance(op, PatchOp)

    def test_create_sum(self):
        """Factory should create SumOp for SUM."""
        op = MergeOpFactory.create(MergeOp.SUM, FieldType.INT64)
        assert isinstance(op, SumOp)

    def test_create_immutable(self):
        """Factory should create ImmutableOp for IMMUTABLE."""
        op = MergeOpFactory.create(MergeOp.IMMUTABLE, FieldType.STRING)
        assert isinstance(op, ImmutableOp)

    def test_from_field(self):
        """Factory should create from MemoryField."""
        field = MemoryField(
            name="test",
            field_type=FieldType.STRING,
            merge_op=MergeOp.SUM,
        )
        op = MergeOpFactory.from_field(field)
        assert isinstance(op, SumOp)


# ============================================================================
# Test Structured Patch Models
# ============================================================================


class TestSearchReplaceBlock:
    """Tests for SearchReplaceBlock."""

    def test_create_basic(self):
        """Create a basic SearchReplaceBlock."""
        block = SearchReplaceBlock(
            search="old content",
            replace="new content",
        )
        assert block.search == "old content"
        assert block.replace == "new content"

    def test_search_description_describes_value_not_another_edit_tutorial(self):
        description = SearchReplaceBlock.model_fields["search"].description
        assert (
            description
            == "Exact, unique text in the original field; must not overlap another block."
        )
        assert "original field, not after earlier edits are applied" in EDIT_TOOL_DESCRIPTION

    def test_exact_matching_instruction_has_one_shared_owner(self):
        description = SearchReplaceBlock.model_fields["search"].description
        assert "Edit a single memory field using exact text replacement" in EDIT_TOOL_DESCRIPTION
        assert "usually 2-4" not in description
        assert "Multi-line SEARCH must be contiguous" not in description

    def test_original_snapshot_semantics_live_in_shared_description(self):
        description = SearchReplaceBlock.model_fields["search"].description
        assert "line_number<TAB>" not in description
        assert "Do not emit overlapping or nested edits" in EDIT_TOOL_DESCRIPTION
        assert "Keep search as small as possible" in EDIT_TOOL_DESCRIPTION

    def test_replace_description_only_describes_replacement_value(self):
        description = SearchReplaceBlock.model_fields["replace"].description
        assert description == "Replacement text for this targeted edit."
        assert "line_number<TAB>" not in description
        assert "Keep search as small as possible" in EDIT_TOOL_DESCRIPTION


class TestDeleteBlock:
    """Tests for DeleteBlock."""

    def test_create_basic(self):
        block = DeleteBlock(delete="line 2\nline 3")

        assert block.delete == "line 2\nline 3"
        assert block.search == "line 2\nline 3"
        assert block.replace == ""

    def test_delete_description_explains_literal_removal(self):
        description = DeleteBlock.model_fields["delete"].description

        assert description is not None
        assert "Include the newline to remove a complete line" in description
        assert "unique" in description
        assert "line_number<TAB>" not in description
        assert "submit all replacements together" in EDIT_TOOL_DESCRIPTION
        assert DeleteBlock.model_json_schema()["properties"]["delete"]["minLength"] == 1


class TestStrPatch:
    """Tests for StrPatch."""

    def test_create_empty(self):
        """Create empty StrPatch."""
        patch = StrPatch()
        assert len(patch.blocks) == 0

    def test_create_with_blocks(self):
        """Create with blocks."""
        block1 = SearchReplaceBlock(search="a", replace="b")
        block2 = SearchReplaceBlock(search="c", replace="d")
        patch = StrPatch(blocks=[block1, block2])
        assert len(patch.blocks) == 2

    def test_create_with_mixed_blocks(self):
        patch = StrPatch(
            blocks=[
                SearchReplaceBlock(search="a", replace="A"),
                DeleteBlock(delete="b"),
            ]
        )

        assert isinstance(patch.blocks[0], SearchReplaceBlock)
        assert isinstance(patch.blocks[1], DeleteBlock)

    def test_parse_delete_block_from_json_shape(self):
        patch = StrPatch.model_validate({"blocks": [{"delete": "line 2"}]})

        assert patch.blocks == [DeleteBlock(delete="line 2")]

    def test_json_schema_exposes_delete_block(self):
        schema = StrPatch.model_json_schema()

        assert schema["$defs"]["DeleteBlock"]["required"] == ["delete"]
        block_refs = {option["$ref"] for option in schema["properties"]["blocks"]["items"]["anyOf"]}
        assert block_refs == {
            "#/$defs/SearchReplaceBlock",
            "#/$defs/DeleteBlock",
        }


# ============================================================================
# Test StrPatch Conversion
# ============================================================================


class TestApplyStrPatch:
    """Tests for apply_str_patch."""

    def test_empty_patch(self):
        """Pi requires at least one edit."""
        with pytest.raises(PatchParseError, match="at least one replacement"):
            apply_str_patch("line1\nline2\nline3", StrPatch())

    @pytest.mark.parametrize(
        ("original", "search", "expected"),
        [
            ("hello world", "hello world", "updated"),
            ("标题：‘天气’晴朗🌞", "标题：'天气'晴朗🌞", "updated"),
        ],
    )
    def test_simple_replace(self, original, search, expected):
        """Exact and fuzzy replacement preserve normalization and match selection."""
        patch = StrPatch(blocks=[SearchReplaceBlock(search=search, replace="updated")])
        assert apply_str_patch(original, patch) == expected

    @pytest.mark.parametrize(
        ("original", "delete", "expected"),
        [
            ("line 1\nline 2\nline 3", "line 2\n", "line 1\nline 3"),
            ("line 1\nline 2\nline 3\nline 4", "line 2\nline 3\n", "line 1\nline 4"),
            ("line 1\nline 2\nline 3", "\nline 2\nline 3", "line 1"),
            ("line 1\nline 2", "line 1\n", "line 2"),
            ("line 1", "line 1", ""),
            ("line 1\r\nline 2\r\nline 3", "line 2\n", "line 1\r\nline 3"),
        ],
    )
    def test_delete_complete_lines(self, original, delete, expected):
        patch = StrPatch(blocks=[DeleteBlock(delete=delete)])

        assert apply_str_patch(original, patch) == expected

    def test_delete_is_literal_replacement_with_empty_text(self):
        patch = StrPatch(blocks=[DeleteBlock(delete="world")])

        assert apply_str_patch("hello world", patch) == "hello "

    def test_duplicate_delete_is_rejected(self):
        patch = StrPatch(blocks=[DeleteBlock(delete="duplicate")])

        with pytest.raises(PatchParseError, match="Found 2 occurrences"):
            apply_str_patch("duplicate\nduplicate", patch)

    def test_duplicate_search_is_rejected(self):
        """Ambiguous SEARCH content must fail instead of replacing globally."""
        original = "status: pending\nstatus: pending"
        patch = StrPatch(
            blocks=[SearchReplaceBlock(search="status: pending", replace="status: done")]
        )

        with pytest.raises(
            PatchParseError,
            match="Please provide more context to make it unique",
        ):
            apply_str_patch(original, patch)

    def test_duplicate_search_after_prior_block_is_rejected(self):
        """A later ambiguous block must not return a partially applied patch."""
        original = "title\nstatus: pending\nstatus: pending"
        patch = StrPatch(
            blocks=[
                SearchReplaceBlock(search="title", replace="updated title"),
                SearchReplaceBlock(search="status: pending", replace="status: done"),
            ]
        )

        with pytest.raises(PatchParseError, match="Found 2 occurrences"):
            apply_str_patch(original, patch)

    def test_numbered_multiline_patch_does_not_guess_a_line_range(self):
        """Display line numbers are not a Pi edit addressing mechanism."""
        original = "keep\nsame\nkeep\nsame"
        patch = StrPatch(
            blocks=[
                SearchReplaceBlock(
                    search="3\tkeep\n4\tsame",
                    replace="3\tKEEP\n4\tSAME",
                )
            ]
        )

        with pytest.raises(PatchParseError, match="Could not find"):
            apply_str_patch(original, patch)

    def test_numbered_patch_does_not_strip_the_search(self):
        """Pi only applies its listed normalizations, not line-number stripping."""
        original = "alpha\nbeta\ngamma"
        patch = StrPatch(
            blocks=[
                SearchReplaceBlock(
                    search=" 2\tbeta",
                    replace=" 2\tBETA",
                )
            ]
        )

        with pytest.raises(PatchParseError, match="Could not find"):
            apply_str_patch(original, patch)


# ============================================================================
# Test Schema Generation Integration - tested in test_schema_models.py
# ============================================================================
