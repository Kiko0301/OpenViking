# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_content_gate import (
    CTX,
    OTHER,
    SOURCE,
    URI,
    Store,
    batch,
    extraction_case,
    file,
    operation,
    patch,
    registry,
    updater,
)

from openviking.models.vlm.base import ToolCall, VLMResponse
from openviking.session.memory.dataclass import MemoryField, MemoryTypeSchema, StoredLink
from openviking.session.memory.edit_repair import EditRepair
from openviking.session.memory.experience.compiler import compile_dag
from openviking.session.memory.merge_op import DeleteBlock, FieldType, MergeOp, PatchOp, StrPatch
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils


def reply(search="old reminder", replace="new reminder", *, edit_index=0):
    return {
        "status": "repaired",
        "repairs": [{"edit_index": edit_index, "search": search, "replace": replace}],
    }


def repairer(response):
    response = json.dumps(response) if isinstance(response, dict) else response
    model = SimpleNamespace(get_completion_async=AsyncMock(return_value=response))
    return EditRepair(model, thinking=True, max_output_tokens=4096), model


def broken():
    return batch(operation(patch("missing reminder", "new reminder"), file()))


async def run(repair, ops=None, *, errors=None, schemas=None):
    return await repair.repair(
        ops if ops is not None else broken(),
        schemas=schemas or registry().list_all(),
        errors=errors or [{"uri": URI, "field": "content", "error": "Not found"}],
    )


@pytest.mark.asyncio
async def test_isolated_json_repairs_both_strings_and_preserves_plan():
    source = (
        'dag = workflow("review")\n# 17. No duplicate DOB collection\n'
        'condition = check("""DOB already available""")\n'
        'reminder = tell("""Use existing DOB""")\ncondition.then(reminder)'
    )
    bad_old = source.replace("# 17.", "#17.").replace('"""', '"')
    bad_new = bad_old + '\nextra = tell("first line\nsecond line")'
    anchor = "condition.then(reminder)"
    corrected_new = anchor + '\nextra = tell("""first line\nsecond line""")'
    op = operation(patch(bad_old, bad_new), file(source))
    op.memory_fields["execution_count"] = 1
    op.old_memory_file_content.extra_fields["execution_count"] = 4
    ops = batch(op, operation("unrelated", uri=OTHER, memory_type="records"))
    ops.case_references = [op.model_copy(deep=True)]
    ops.resolved_links = [StoredLink(from_uri=URI, to_uri=OTHER, relation="supported_by")]
    ops.delete_replacements = {"old": "new"}
    before = ops.model_copy(deep=True)
    repair, model = repairer(reply(anchor, corrected_new))
    result = await run(repair, ops)
    assert result.repaired and result.operations is not ops
    assert ops == before
    candidate = result.operations
    updated = candidate.upsert_operations[0].memory_fields.pop("content")
    expected = before.model_copy(deep=True)
    expected.upsert_operations[0].memory_fields.pop("content")
    assert candidate == expected  # Counters, source, identities, links and other writes untouched.
    text = await PatchOp(FieldType.STRING).apply(source, updated)
    assert text == source + '\nextra = tell("""first line\nsecond line""")'
    assert compile_dag(text)
    request = model.get_completion_async.await_args.kwargs
    assert request["tools"] == [] and request["tool_choice"] is None
    assert request["thinking"] is True and request["max_tokens"] == 4096
    assert len(request["messages"]) == 2
    payload = json.loads(request["messages"][1]["content"])
    assert payload["targets"][0]["original_text"] == source
    assert payload["targets"][0]["edits"][0]["replace"] == bad_new
    assert "execution_count" not in request["messages"][1]["content"]
    assert "unrelated" not in request["messages"][1]["content"]


@pytest.mark.asyncio
async def test_one_request_covers_multiple_fields_with_global_indices():
    reg = registry()
    schema = reg.get("experiences")
    schema.fields.append(MemoryField(name="description", field_type=FieldType.STRING))
    ops = broken()
    op = ops.upsert_operations[0]
    op.old_memory_file_content.extra_fields["description"] = "3) Actual text"
    op.memory_fields["description"] = patch("3. Actual text", "3) Updated text").model_dump()
    response = reply()
    response["repairs"] += reply("3) Actual text", "3) Updated text", edit_index=1)["repairs"]
    repair, model = repairer(response)
    result = await run(repair, ops, schemas=reg.list_all())
    assert result.operations is not ops
    assert model.get_completion_async.await_count == 1
    assert result.operations.upsert_operations[0].memory_fields["description"] == {
        "blocks": [{"search": "3) Actual text", "replace": "3) Updated text"}]
    }
    payload = json.loads(model.get_completion_async.await_args.kwargs["messages"][1]["content"])
    assert [t["edits"][0]["edit_index"] for t in payload["targets"]] == [0, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"status": "cannot_repair", "reason": "Unclear intent"},
        "not json",
        "```json\n{}\n```",
        "sdk.commit()",
        "null",
        "[]",
        "{}",
        reply(edit_index=True),
        reply(edit_index=-1),
        reply(edit_index=1),
        reply(edit_index="0"),
        reply(search=1),
        reply(replace=None),
        reply(search=""),
        {"status": "repaired", "repairs": []},
        {"status": "repaired", "repairs": reply()["repairs"] * 2},
        {**reply(), "uri": OTHER},
        {"status": "repaired", "repairs": [{**reply()["repairs"][0], "old_text": "old reminder"}]},
        {
            "status": "repaired",
            "repairs": [{"edit_index": 0, "old_text": "old reminder", "new_text": "new"}],
        },
        '{"status":"cannot_repair","status":"repaired","repairs":[]}',
        reply(search="still absent"),
        reply(search="reminder"),
        reply(replace="invalid\nstring"),
        VLMResponse(content=json.dumps(reply()), finish_reason="length"),
        VLMResponse(content=json.dumps(reply()), finish_reason="error"),
        VLMResponse(content=json.dumps(reply()), tool_calls=[ToolCall("1", "write", {})]),
        None,
    ],
)
async def test_rejects_bad_repairs_atomically_on_each_call(response):
    ops = broken()
    before = ops.model_copy(deep=True)
    repair, model = repairer(response)
    for _ in range(2):
        result = await run(repair, ops)
        assert not result.repaired and result.failure_reason
        assert result.operations is ops and ops == before
    assert model.get_completion_async.await_count == 2


@pytest.mark.asyncio
async def test_service_error_does_not_disable_later_repairs_and_cancellation_propagates():
    import asyncio

    repair, model = repairer(reply())
    model.get_completion_async.side_effect = [RuntimeError("offline"), json.dumps(reply())]
    assert (await run(repair)).failure_reason == "model_error: offline"
    assert (await run(repair)).repaired
    assert model.get_completion_async.await_count == 2
    repair, model = repairer(reply())
    model.get_completion_async.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await run(repair)


@pytest.mark.asyncio
async def test_revalidates_batch_overlap_and_keeps_unchanged_siblings():
    ops = broken()
    ops.upsert_operations[0].memory_fields["content"].blocks.extend(
        patch("Current request applies", "Other condition").blocks
    )
    repair, model = repairer(reply("Current request applies", "Another condition"))
    result = await run(repair, ops)
    assert not result.repaired and result.failure_reason
    assert result.operations is ops  # Repair overlaps the second edit.
    payload = json.loads(model.get_completion_async.await_args.kwargs["messages"][1]["content"])
    assert len(payload["targets"][0]["edits"]) == 2
    repair, _ = repairer(reply())
    result = await run(repair, ops)
    value = result.operations.upsert_operations[0].memory_fields["content"]
    assert value.blocks[1] == ops.upsert_operations[0].memory_fields["content"].blocks[1]
    text = await PatchOp(FieldType.STRING).apply(SOURCE, value)
    assert "Other condition" in text and "new reminder" in text


@pytest.mark.asyncio
async def test_drop_siblings_are_read_only_and_keep_literal_deletion_semantics():
    ops = broken()
    op = ops.upsert_operations[0]
    op.old_memory_file_content.content = SOURCE + "\n# obsolete note"
    op.memory_fields["content"].blocks.append(DeleteBlock(delete="obsolete"))
    repair, _ = repairer(reply("obsolete", "invented", edit_index=1))
    assert (await run(repair, ops)).operations is ops
    repair, model = repairer(reply())
    result = await run(repair, ops)
    value = result.operations.upsert_operations[0].memory_fields["content"]
    assert isinstance(value.blocks[1], DeleteBlock)
    text = await PatchOp(FieldType.STRING).apply(op.old_memory_file_content.content, value)
    assert text.endswith("\n#  note")
    payload = json.loads(model.get_completion_async.await_args.kwargs["messages"][1]["content"])
    assert payload["targets"][0]["edits"][1]["editable"] is False
    assert "match_hints" not in payload["targets"][0]["edits"][1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "create",
        "overwrite",
        "drop",
        "multi_uri",
        "sequential",
        "missing_original",
        "wrong_snapshot",
        "immutable",
        "semantic",
        "mixed",
        "valid",
        "malformed_patch",
    ],
)
async def test_non_edit_or_ambiguous_targets_keep_existing_handling(kind):
    ops = broken()
    op = ops.upsert_operations[0]
    schemas = registry().list_all()
    errors = None
    if kind == "create":
        op.create_only = True
    elif kind == "overwrite":
        op.memory_fields["content"] = "invalid source"
    elif kind == "drop":
        op.memory_fields["content"] = StrPatch(blocks=[DeleteBlock(delete="absent")])
    elif kind == "multi_uri":
        op.uris.append(OTHER)
    elif kind == "sequential":
        ops.upsert_operations.append(op.model_copy(deep=True))
    elif kind == "missing_original":
        op.old_memory_file_content = None
    elif kind == "wrong_snapshot":
        op.old_memory_file_content.uri = OTHER
    elif kind == "immutable":
        for field in schemas[0].fields:
            if field.name == "content":
                field.merge_op = MergeOp.IMMUTABLE
    elif kind == "semantic":
        errors = [{"uri": URI, "gate": "ExperienceIdentifierGate", "error": "Identifier"}]
    elif kind == "mixed":
        errors = [{"uri": URI, "error": "Bad edit"}, {"uri": OTHER, "error": "Bad create"}]
    elif kind == "valid":
        op.memory_fields["content"] = patch()
    elif kind == "malformed_patch":
        op.memory_fields["content"] = {"blocks": [{"search": "missing"}]}
    repair, model = repairer(reply())
    result = await run(repair, ops, errors=errors, schemas=schemas)
    assert not result.repaired and result.failure_reason is None
    assert result.operations is ops
    model.get_completion_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_content_field_is_validated_by_its_gate():
    schema = MemoryTypeSchema(
        memory_type="cases",
        fields=[
            MemoryField(name="rubric", field_type=FieldType.STRING),
        ],
    )
    old = file()
    old.memory_type = "cases"
    old.extra_fields["rubric"] = '[{"id":"scope","description":"old"}]'
    op = operation("unused", old, memory_type="cases")
    op.memory_fields = {"rubric": patch('"old"', '"broken\nstring"')}
    repair, _ = repairer(reply('"old"', '"new"'))
    result = await run(
        repair, batch(op), schemas=[schema], errors=[{"uri": URI, "field": "rubric"}]
    )
    assert result.repaired and result.operations.upsert_operations[0].memory_fields[
        "rubric"
    ] == patch('"old"', '"new"')


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_extract_loop_does_not_replay_sdk_or_include_learning_context(
    monkeypatch, output_format
):
    bad = 'experience.content.edit(search="missing reminder", replace="new reminder")\nsdk.commit()'
    if output_format == "json":
        bad = json.dumps(
            {
                "experiences": [
                    {
                        "experience_name": "case_a",
                        "page_id": 1,
                        "content": patch("missing reminder", "new reminder").model_dump(),
                    }
                ]
            }
        )
    extraction, model, fs = extraction_case(
        monkeypatch, [bad, json.dumps(reply())], output_format=output_format
    )
    extraction.thinking = True
    extraction.max_output_tokens = 5678
    result, _ = await extraction.run()
    assert model.get_completion_async.await_count == 2
    initial, repaired = [call.kwargs for call in model.get_completion_async.await_args_list]
    assert repaired["thinking"] == initial["thinking"] is True
    assert repaired["max_tokens"] == initial["max_tokens"] == 5678
    assert len(repaired["messages"]) == 2
    payload = repaired["messages"][1]["content"]
    assert "The old reminder needs correction" not in payload
    assert "archive_001" not in payload and "trajectories/observation" not in payload
    assert bad not in payload
    applied = await updater(fs).apply_operations(result, CTX)
    assert not applied.errors
    assert fs.writes == [URI]


@pytest.mark.asyncio
async def test_late_repair_must_pass_provider_gate_before_accepting_candidate(monkeypatch):
    bad = 'experience.content.edit(search="missing reminder", replace="new reminder")\nsdk.commit()'
    good = (
        'experience.content.edit(search="old reminder", replace="main loop reminder")\nsdk.commit()'
    )
    extraction, model, fs = extraction_case(monkeypatch, [bad, json.dumps(reply()), good])
    extraction.context_provider.operation_validator = AsyncMock(
        side_effect=[
            [],
            [{"uri": URI, "gate": "ProtectedGate", "error": "rejected"}],
            [],
        ]
    )
    result, _ = await extraction.run()
    assert model.get_completion_async.await_count == 3
    feedback = "\n".join(
        m["content"] or "" for m in model.get_completion_async.await_args.kwargs["messages"]
    )
    assert "candidate_validation_failed" in feedback and "ProtectedGate" in feedback
    assert not fs.writes
    assert result.upsert_operations[0].memory_fields["content"] == patch(
        replace="main loop reminder"
    )


@pytest.mark.asyncio
async def test_repair_does_not_bypass_stale_read_guard():
    ops = broken()
    repair, _ = repairer(reply())
    result = await run(repair, ops)
    changed = SOURCE.replace("old reminder", "old  reminder")
    fs = Store({URI: MemoryFileUtils.write(file(changed))})
    applied = await updater(fs).apply_operations(result.operations, CTX)
    assert applied.errors and not fs.writes
    assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == changed


@pytest.mark.asyncio
@pytest.mark.parametrize("repaired", [True, False])
async def test_case_counter_is_not_replayed_and_existing_rejection_policy_is_unchanged(repaired):
    from openviking.session.memory.memory_type_registry import resolve_memory_templates_dir

    reg = registry()
    reg.load_from_yaml(str(resolve_memory_templates_dir() / "cases.yaml"))
    uri = "viking://user/alice/memories/cases/library.md"
    old = file(uri=uri)
    old.memory_type = "cases"
    old.content = ""
    old.extra_fields = {
        "case_name": "library",
        "domain": "library",
        "description": "3) Original description",
        "rubric": "[]",
        "execution_count": 4,
    }
    op = operation("", old, uri=uri, memory_type="cases")
    op.memory_fields = {
        "case_name": "library",
        "execution_count": 1,
        "description": patch("3. Original description", "3) Revised description"),
    }
    response = (
        reply("3) Original description", "3) Revised description")
        if repaired
        else {"status": "cannot_repair", "reason": "Unclear"}
    )
    repair, model = repairer(response)
    result = await run(
        repair,
        batch(op),
        schemas=reg.list_all(),
        errors=[
            {
                "uri": uri,
                "field": "description",
                "error": "Not found",
            }
        ],
    )
    assert result.repaired is repaired
    assert bool(result.failure_reason) is not repaired
    assert model.get_completion_async.await_count == 1
    assert result.operations.upsert_operations[0].memory_fields["execution_count"] == 1
    fs = Store({uri: MemoryFileUtils.write(old)})
    writer = updater(fs)
    writer._registry = reg
    await writer.apply_operations(result.operations, CTX)
    memory = MemoryFileUtils.read(fs.files[uri], uri=uri)
    # A failed Case patch still rejects the candidate as before; successful repair
    # applies the original counter delta exactly once, never replaying an SDK update.
    assert memory.extra_fields["execution_count"] == (5 if repaired else 4)
    assert memory.extra_fields["description"] == (
        "3) Revised description" if repaired else "3) Original description"
    )
    assert fs.writes == ([uri] if repaired else [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (
            '{"status":"cannot_repair","reason":"Ambiguous intent"}',
            "cannot_repair: Ambiguous intent",
        ),
        ("not JSON", "invalid_repair:"),
        (json.dumps(reply(edit_index=999)), "Invalid or duplicate editable index: 999"),
        (
            VLMResponse(content=json.dumps(reply()), finish_reason="length"),
            "without truncation/tools",
        ),
        (RuntimeError("repair unavailable"), "model_error: repair unavailable"),
    ],
)
async def test_failed_local_repair_feeds_reason_to_original_loop_and_recovers(
    monkeypatch, response, reason
):
    bad = 'experience.content.edit(search="missing reminder", replace="new reminder")\nsdk.commit()'
    good = 'experience.content.edit(search="old reminder", replace="new reminder")\nsdk.commit()'
    extraction, model, fs = extraction_case(monkeypatch, [bad, response, good])
    result, _ = await extraction.run()
    assert model.get_completion_async.await_count == 3
    local_call = model.get_completion_async.await_args_list[1].kwargs
    assert len(local_call["messages"]) == 2 and local_call["tools"] == []
    fallback = model.get_completion_async.await_args_list[2].kwargs["messages"]
    assert {"role": "assistant", "content": bad} in fallback
    assert any(reason in m["content"] for m in fallback)
    assert not fs.writes  # Both drafts were pending plans, not committed batches.
    applied = await updater(fs).apply_operations(result, CTX)
    assert not applied.errors and fs.writes == [URI]
    assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == SOURCE.replace(
        "old reminder", "new reminder"
    )


@pytest.mark.asyncio
async def test_each_failed_edit_is_repaired_when_main_retry_moves_between_validation_paths(
    monkeypatch,
):
    bad = 'experience.content.edit(search="missing reminder", replace="new reminder")\nsdk.commit()'
    cannot = '{"status":"cannot_repair","reason":"Unclear"}'
    extraction, model, _fs = extraction_case(monkeypatch, [bad, cannot, bad, json.dumps(reply())])
    extraction.context_provider.operation_validator = AsyncMock(
        side_effect=[
            [{"uri": URI, "gate": "candidate_content", "error": "Not found"}],
            [],  # The next draft reaches the generic content/patch path.
            [],
        ]
    )
    result, _ = await extraction.run()
    assert result.upsert_operations[0].memory_fields["content"] == patch()
    assert model.get_completion_async.await_count == 4
    calls = model.get_completion_async.await_args_list
    assert sum(len(call.kwargs["messages"]) == 2 for call in calls) == 2
    assert json.loads(calls[3].kwargs["messages"][1]["content"])["targets"]


@pytest.mark.asyncio
async def test_main_loop_fallback_rebuilds_case_counter_delta_instead_of_accumulating(monkeypatch):
    from test_case_selection import selection_loop

    extraction, model, files, fs, ctx, reg, memories = selection_loop(
        monkeypatch, output_format="python", execution_count=7
    )
    counter = "second.update(execution_count=1)\n"
    bad = (
        counter
        + 'second.domain.edit(search="missing domain", replace="Updated domain")\nsdk.commit()'
    )
    good = (
        counter
        + 'second.domain.edit(search="Domain second", replace="Updated domain")\nsdk.commit()'
    )
    model.get_completion_async.side_effect = [
        bad,
        '{"status":"cannot_repair","reason":"Unclear"}',
        good,
    ]
    result, _ = await extraction.run()
    assert model.get_completion_async.await_count == 3
    assert len(result.upsert_operations) == 1
    assert result.upsert_operations[0].memory_fields["execution_count"] == 1
    fs.write_file.assert_not_awaited()
    assert memories[1].extra_fields["execution_count"] == 7
    store = Store(files)
    writer = updater(store)
    writer._registry = reg
    applied = await writer.apply_operations(result, ctx)
    assert not applied.errors and store.writes == [memories[1].uri]
    saved = MemoryFileUtils.read(store.files[memories[1].uri], uri=memories[1].uri)
    assert saved.extra_fields["execution_count"] == 8
    assert saved.extra_fields["domain"] == "Updated domain"


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
@pytest.mark.parametrize("last_repair_succeeds", [True, False])
async def test_loop_bounds_failed_edit_retries_but_repairs_the_last_draft(
    monkeypatch, output_format, last_repair_succeeds
):
    bad = 'experience.content.edit(search="missing reminder", replace="new reminder")\nsdk.commit()'
    if output_format == "json":
        bad = json.dumps(
            {
                "experiences": [
                    {
                        "experience_name": "case_a",
                        "page_id": 1,
                        "content": patch("missing reminder", "new reminder").model_dump(),
                    }
                ]
            }
        )
    cannot = '{"status":"cannot_repair","reason":"Unclear intended change"}'
    extraction, model, fs = extraction_case(
        monkeypatch,
        [bad, cannot, bad, json.dumps(reply()) if last_repair_succeeds else cannot],
        output_format=output_format,
    )
    result, _ = await extraction.run()
    assert model.get_completion_async.await_count == 4  # Two drafts, two isolated repairs.
    calls = model.get_completion_async.await_args_list
    assert all(len(calls[i].kwargs["messages"]) == 2 for i in (1, 3))
    assert not fs.writes
    if last_repair_succeeds:
        assert result.upsert_operations[0].memory_fields["content"] == patch()
        applied = await updater(fs).apply_operations(result, CTX)
        assert not applied.errors and fs.writes == [URI]
    else:
        assert not result.upsert_operations  # Existing terminal policy discards invalid content.
        assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == SOURCE
