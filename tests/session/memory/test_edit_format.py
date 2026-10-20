# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Verify the Pi-derived Edit instructions and our SDK-only adaptations."""

from __future__ import annotations

import re

import pytest

from openviking.session.memory.extraction_output_protocol import create_extraction_output_protocol
from openviking.session.memory.extraction_output_protocol.edit_format import EDIT_TOOL_DESCRIPTION
from openviking.session.memory.merge_op import FieldType, PatchOp, SearchReplaceBlock, StrPatch
from tests.session.memory.test_extraction_output_protocol import (
    _bind,
    _context,
    _existing_preference,
    _preference_schema,
    _profile_schema,
)


@pytest.mark.parametrize("output_format", ["python", "json"])
def test_edit_description_is_shared_once_not_per_field_or_retry(output_format):
    context = _context([_preference_schema(), _profile_schema()])
    protocol = create_extraction_output_protocol(output_format)
    contract = protocol.render_contract(context)
    assert contract.count(EDIT_TOOL_DESCRIPTION) == 1
    assert contract.count("Keep search as small as possible") == 1
    assert contract.count("not after earlier edits are applied") == 1
    assert "Do not emit overlapping or nested edits" in contract
    assert "replace_all" not in contract
    assert "old_string" not in contract and "new_string" not in contract
    assert "search" in contract and "replace" in contract
    assert "oldText" not in contract and "newText" not in contract
    assert "old_text" not in contract and "new_text" not in contract
    assert "preserve the exact indentation (tabs/spaces)" not in contract
    assert "Only use emojis" not in contract
    errors = [{"uri": "example", "field": "content", "search": "missing", "match_count": 0}]
    repair = protocol.render_patch_repair(errors)
    assert EDIT_TOOL_DESCRIPTION not in repair
    assert "Follow the Edit instructions in the initial output contract." in repair
    assert '"match_count": 0' in repair
    assert "complete" in repair and "source associations" in repair


def test_python_edit_description_is_adjacent_to_method_signature():
    contract = create_extraction_output_protocol("python").render_contract(
        _context([_preference_schema(), _profile_schema()])
    )
    marker = "### obj.field.edit(search=..., replace=...)"
    assert contract.count(marker) == 1
    assert f"{marker}\n{EDIT_TOOL_DESCRIPTION}\nLocal replacement example" in contract
    assert contract.index("Local replacement example") < contract.index("### obj.field.drop")
    assert contract.rindex("content [editable string:") < contract.index(marker)
    assert "edit()/drop() may be chained" in contract
    assert "All old content omitted from the new string is removed" in contract
    assert "For existing memories, prefer the smallest unique" not in contract
    assert "Each search= (and drop text=) MUST" not in contract


@pytest.mark.parametrize(
    "arguments",
    [
        "old_string='old', new_string='new'",
        "old_text='old', new_text='new'",
        "oldText='old', newText='new'",
        "search='old', replace='new', oldText='old'",
        "search='old', replace='new', old_text='old'",
        "edits=[{'search': 'old', 'replace': 'new'}]",
        "search='old', replace='new', replace_all=True",
    ],
)
def test_file_tool_parameter_names_and_replace_all_are_not_added_to_sdk(arguments):
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context([_preference_schema()], files=[_existing_preference(uri, "editor", "old")])
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    operations, error = protocol.parse(f"preferences_1.content.edit({arguments})", context)
    assert operations is None
    assert "requires exactly search= and replace=" in error


@pytest.mark.parametrize("output_format", ["python", "json"])
def test_search_replace_schema_keeps_format_but_not_another_edit_tutorial(output_format):
    context = _context([_preference_schema()])
    text = create_extraction_output_protocol(output_format).render_contract(context)
    for old_rule in (
        "Use the smallest unique fragment - usually 2-4 adjacent lines",
        "Multi-line SEARCH must be contiguous; split non-adjacent edits",
        "Never use SEARCH text from another memory or page",
    ):
        assert old_rule not in text


def test_python_edit_selection_rule_and_example_have_one_shared_owner():
    protocol = create_extraction_output_protocol("python")
    contract = protocol.render_contract(_context([_preference_schema(), _profile_schema()]))
    selection_rule = "Use overwrite only for complete field rewrites."
    assert contract.count(selection_rule) == 1
    assert contract.index(selection_rule) > contract.index("### obj.field.overwrite")
    assert contract.count("Local replacement example") == 1
    assert selection_rule not in protocol.render_patch_repair([])
    assert "Local replacement example" not in protocol.render_patch_repair([])
    examples = re.findall(r"```python\n(.*?)\n```", contract, re.DOTALL)
    assert not any(".overwrite(" in example for example in examples)


@pytest.mark.asyncio
async def test_python_shared_edit_example_parses_and_preserves_unrelated_text():
    uri = "viking://user/alice/memories/preferences/editor.md"
    original = "Editor: Vim\nTheme: dark"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", original)]
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    contract = protocol.render_contract(context)
    edit_section = contract.split("### obj.field.edit", 1)[1].split("### obj.field.drop", 1)[0]
    examples = re.findall(r"```python\n(.*?)\n```", edit_section, re.DOTALL)
    assert len(examples) == 1
    program = examples[0].replace("obj.content", "preferences_1.content")
    operations, error = protocol.parse(program, context)
    assert error is None
    assert operations is not None
    changes = operations.model_dump()["preferences"]
    assert len(changes) == 1
    assert changes[0]["content"] == {
        "blocks": [{"search": "Editor: Vim", "replace": "Editor: Emacs"}]
    }
    result = await PatchOp(FieldType.STRING).apply_checked(original, changes[0]["content"])
    assert result == "Editor: Emacs\nTheme: dark"


def test_edit_block_schema_and_serialization_only_use_renamed_parameters():
    schema = SearchReplaceBlock.model_json_schema()
    assert set(schema["properties"]) == {"search", "replace"}
    assert set(schema["required"]) == {"search", "replace"}
    payload = {"blocks": [{"search": "old", "replace": "new"}]}
    assert StrPatch.model_validate(payload).model_dump() == payload
    with pytest.raises(ValueError):
        StrPatch.model_validate({"blocks": [{"old_text": "old", "new_text": "new"}]})
    with pytest.raises(ValueError):
        StrPatch.model_validate({"blocks": [{"oldText": "old", "newText": "new"}]})
