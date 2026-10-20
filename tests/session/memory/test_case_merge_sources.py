from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_case_selection import case_memory, selection_loop

from openviking.session.compressor_v3 import _canonical_cases_from_update_result
from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
from openviking.session.memory.memory_updater import MemoryUpdateResult
from openviking.session.memory.patch_merge_context_provider import (
    PatchMergeContextProvider,
    PatchMergePatch,
)
from openviking.session.memory.streaming_memory_updater import (
    MemoryUpdateRequest,
    StreamingMemoryUpdateResult,
    scope_memory_update_result_to_submitter,
)


def merge_loop(monkeypatch, *, output_format="python", names=("first", "alias"), invalid=None):
    loop, model, files, fs, ctx, registry, memories = selection_loop(
        monkeypatch, output_format=output_format, invalid=invalid
    )
    original = loop.context_provider
    fs._uri_to_path = lambda uri, **kwargs: uri
    requests = []
    patches = []
    for index, name in enumerate(names, 1):
        memory = case_memory(name)
        source = f"source-{index}"
        requests.append(
            MemoryUpdateRequest(
                operations=ResolvedOperations(
                    upsert_operations=[
                        ResolvedOperation(
                            memory_type="cases",
                            memory_fields=memory.extra_fields,
                            uris=[memory.uri],
                        )
                    ],
                    delete_file_contents=[],
                    errors=[],
                ),
                messages=[],
                ctx=ctx,
                metadata={"source_extraction_id": source},
            )
        )
        patches.append(
            PatchMergePatch(
                before_file=None, after_file=memory, metadata={"source_extraction_ids": [source]}
            )
        )
    provider = PatchMergeContextProvider(
        memory_type="cases", patches=patches, memory_registry=registry, output_language="en"
    )
    provider._ctx = ctx
    provider._viking_fs = fs
    provider._extract_context = original.get_extract_context()
    provider._isolation_handler = original._isolation_handler
    provider.read_file_contents.update(original.read_file_contents)
    provider.prefetch = original.prefetch
    loop.context_provider = provider
    return SimpleNamespace(
        loop=loop, model=model, files=files, fs=fs, ctx=ctx, requests=requests, memories=memories
    )


def response(output_format, bindings):
    if output_format == "python":
        entries = ", ".join(f"({name}, {ids!r})" for name, ids in bindings)
        return f"sdk.commit(case_sources=[{entries}])"
    return json.dumps(
        {
            "case_sources": [
                {"case_id": {"first": 1, "second": 2}[name], "patch_ids": ids}
                for name, ids in bindings
            ]
        }
    )


async def cases_for_request(env, operations, request):
    result = scope_memory_update_result_to_submitter(
        StreamingMemoryUpdateResult(
            operations=operations,
            apply_result=MemoryUpdateResult(),
            request_count=len(env.requests),
        ),
        request,
    )
    cases = await _canonical_cases_from_update_result(
        operations=result.operations,
        requested_operations=request.operations,
        result=result.apply_result,
        viking_fs=env.fs,
        ctx=env.ctx,
    )
    return cases, result


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_merge_keeps_all_47_sources_when_44_match_the_final_case(monkeypatch, output_format):
    names = ["first"] * 44 + ["reservation_management"] * 2 + ["reservation_services"]
    env = merge_loop(monkeypatch, output_format=output_format, names=names)
    env.model.get_completion_async.return_value = response(
        output_format, [("first", list(range(1, 48)))]
    )
    before = dict(env.files)

    operations, _ = await env.loop.run()

    assert operations.upsert_operations == []  # Unchanged Case, not a fake write.
    assert len(operations.case_references) == 1
    for request in env.requests:
        cases, result = await cases_for_request(env, operations, request)
        assert [case.name for case in cases] == ["first"]
        assert result.apply_result.written_uris == []
    assert env.files == before
    env.fs.write_file.assert_not_awaited()
    env.model.get_completion_async.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_final_case_mapping_overrides_original_uri_without_cross_association(
    monkeypatch, output_format
):
    # Same original URI does NOT mean both requests belong to both final Cases.
    env = merge_loop(monkeypatch, output_format=output_format, names=("first", "first"))
    env.model.get_completion_async.return_value = response(
        output_format, [("first", [1]), ("second", [2])]
    )
    operations, _ = await env.loop.run()

    for request, expected in zip(env.requests, ["first", "second"], strict=True):
        cases, result = await cases_for_request(env, operations, request)
        assert [case.name for case in cases] == [expected]
        assert len(result.operations.case_references) == 1
    assert all("case_sources" not in ref.memory_fields for ref in operations.case_references)


@pytest.mark.asyncio
async def test_one_input_may_contribute_to_multiple_final_cases(monkeypatch):
    env = merge_loop(monkeypatch)
    env.model.get_completion_async.return_value = response(
        "python", [("first", [1, 2]), ("second", [2])]
    )
    operations, _ = await env.loop.run()
    first, _ = await cases_for_request(env, operations, env.requests[0])
    second, _ = await cases_for_request(env, operations, env.requests[1])
    assert [case.name for case in first] == ["first"]
    assert {case.name for case in second} == {"first", "second"}


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
@pytest.mark.parametrize("patch_ids", [[1], [1, 2, 3], [1, 1, 2], [True, 2], ["1", 2]])
async def test_invalid_source_bindings_cannot_silently_drop_or_forge_associations(
    monkeypatch, output_format, patch_ids
):
    env = merge_loop(monkeypatch, output_format=output_format)
    env.model.get_completion_async.return_value = response(output_format, [("first", patch_ids)])
    with pytest.raises(RuntimeError, match="could not be parsed"):
        await env.loop.run()
    assert env.model.get_completion_async.await_count == 2  # Existing format repair, no extra loop.
    env.fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["foreign", "wrong_type", "unread", "deleted"])
async def test_bound_cases_still_require_valid_accessible_surviving_targets(monkeypatch, invalid):
    env = merge_loop(monkeypatch, output_format="json", invalid=invalid)
    payload = {"case_sources": [{"case_id": 2, "patch_ids": [1, 2]}]}
    if invalid == "deleted":
        payload["delete_ids"] = [{"delete_page_id": 2, "replacement_page_id": None}]
    env.model.get_completion_async.return_value = json.dumps(payload)
    with pytest.raises(ValueError, match="Selected Case"):
        await env.loop.run()
    env.fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_sources_can_target_a_new_case_without_selecting_existing_candidates(
    monkeypatch, output_format
):
    env = merge_loop(monkeypatch, output_format=output_format)
    fields = {
        "case_name": "new",
        "execution_count": 2,
        "domain": "New domain",
        "description": "context",
        "rubric": '[{"id":"scope","description":"Respect scope."}]',
    }
    if output_format == "python":
        env.model.get_completion_async.return_value = (
            "created = sdk.create_cases("
            + ", ".join(f"{key}={value!r}" for key, value in fields.items())
            + ")\nsdk.commit(case_sources=[(created, [1, 2])])"
        )
    else:
        env.model.get_completion_async.return_value = json.dumps(
            {
                "cases": [{"page_id": 100, **fields}],
                "case_sources": [{"case_id": 100, "patch_ids": [1, 2]}],
            }
        )
    operations, _ = await env.loop.run()
    assert operations.case_references == []
    assert len(operations.upsert_operations) == 1
    operation = operations.upsert_operations[0]
    assert operation.memory_fields["source_extraction_ids"] == ["source-1", "source-2"]
    assert operation.uris == ["viking://user/u/memories/cases/new.md"]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [[], {}, {"case_sources": []}])
async def test_empty_json_merge_cannot_discard_case_associations(monkeypatch, payload):
    env = merge_loop(monkeypatch, output_format="json")
    env.model.get_completion_async.return_value = json.dumps(payload)
    with pytest.raises(RuntimeError, match="could not be parsed"):
        await env.loop.run()
    env.fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_case_merge_source_numbers_do_not_become_model_editable_case_fields(monkeypatch):
    env = merge_loop(monkeypatch)
    env.model.get_completion_async.return_value = response("python", [("first", [1, 2])])
    await env.loop.run()
    context = env.loop._output_context
    contract = env.loop._output_protocol.render_contract(context)
    assert "sdk.commit(case_sources=" in contract
    assert "source_extraction_ids" not in contract
    assert not any(
        "source-1" in str(call.kwargs["messages"])
        for call in env.model.get_completion_async.await_args_list
    )


@pytest.mark.asyncio
async def test_unchanged_case_is_kept_alongside_a_written_case(monkeypatch):
    env = merge_loop(monkeypatch, names=("first", "second"))
    proposed = [request.operations.upsert_operations[0] for request in env.requests]
    operations = ResolvedOperations(upsert_operations=proposed, delete_file_contents=[], errors=[])
    result = MemoryUpdateResult()
    result.add_written(env.memories[0].uri)
    cases = await _canonical_cases_from_update_result(
        operations=operations,
        requested_operations=operations,
        result=result,
        viking_fs=env.fs,
        ctx=env.ctx,
    )
    assert {case.name for case in cases} == {"first", "second"}


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", [True, "1", 1.0])
async def test_json_does_not_coerce_case_source_target_ids(monkeypatch, case_id):
    env = merge_loop(monkeypatch, output_format="json")
    env.model.get_completion_async.return_value = json.dumps(
        {"case_sources": [{"case_id": case_id, "patch_ids": [1, 2]}]}
    )
    with pytest.raises(RuntimeError, match="could not be parsed"):
        await env.loop.run()
    env.fs.write_file.assert_not_awaited()


def test_unchanged_case_patch_keeps_its_identity_but_hides_system_source_ids(monkeypatch):
    from openviking.session.memory.patch_merge_context_provider import _render_field_diff_patches

    env = merge_loop(monkeypatch)
    memory = env.memories[0]
    patch = PatchMergePatch(
        before_file=memory,
        after_file=memory,
        metadata={"source_extraction_ids": ["private-source"]},
    )
    schema = env.loop.context_provider.get_memory_schemas(env.ctx)[0]
    rendered = _render_field_diff_patches([patch], schema)
    assert "(no changes)" in rendered
    assert 'proposed_case: "first"' in rendered
    assert "private-source" not in rendered


@pytest.mark.asyncio
async def test_case_merge_bindings_do_not_implicitly_select_inspected_candidates(monkeypatch):
    env = merge_loop(monkeypatch)
    env.model.get_completion_async.return_value = (
        "inspected = second\nselected = first\nsdk.commit(case_sources=[(selected, [1, 2])])"
    )
    operations, _ = await env.loop.run()
    assert [ref.uris for ref in operations.case_references] == [[env.memories[0].uri]]
    assert not operations.upsert_operations
    env.fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_case_merge_binding_without_source_mapping_is_still_invalid(monkeypatch):
    env = merge_loop(monkeypatch)
    env.model.get_completion_async.return_value = "selected = first"
    with pytest.raises(RuntimeError, match="could not be parsed"):
        await env.loop.run()
    env.fs.write_file.assert_not_awaited()
