from __future__ import annotations

import copy
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v3 import _canonical_cases_from_update_result
from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry, get_default_registry
from openviking.session.memory.memory_updater import MemoryUpdateResult
from openviking.session.memory.merge_op.sum import SumOp
from openviking.session.memory.session_extract_context_provider import SessionExtractContextProvider
from openviking.session.memory.streaming_memory_updater import (
    MemoryUpdateRequest,
    StreamingMemoryUpdater,
)
from openviking.session.memory.tools import add_tool_call_pair_to_messages
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config import get_openviking_config


def case_memory(name, *, user="u", memory_type="cases"):
    return MemoryFile(
        uri=f"viking://user/{user}/memories/{memory_type}/{name}.md",
        memory_type=memory_type,
        content=f"Business domain {name}",
        extra_fields={
            "case_name": name,
            "domain": f"Domain {name}",
            "description": '{"summary":"service domain"}',
            "rubric": "[]",
            # Old on-disk data must not become Case learning context after field removal.
            "evidence": "OBSOLETE_CASE_EVIDENCE",
            "version": 1,
        },
    )


def selection_loop(monkeypatch, *, output_format, invalid=None, execution_count=None):
    monkeypatch.setattr(get_openviking_config().memory, "extraction_output_format", output_format)
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.register(get_default_registry().get("cases"))
    memories = [
        case_memory("first"),
        case_memory(
            "second",
            user="other" if invalid == "foreign" else "u",
            memory_type="entities" if invalid == "wrong_type" else "cases",
        ),
    ]
    if execution_count is not None:
        memories[1].extra_fields["execution_count"] = execution_count
    files = {m.uri: MemoryFileUtils.write(m) for m in memories}
    fs = SimpleNamespace(
        read_file=AsyncMock(side_effect=lambda uri, **kwargs: files[uri]),
        exists=AsyncMock(side_effect=lambda uri, **kwargs: uri in files),
        write_file=AsyncMock(side_effect=AssertionError("selection must not write")),
    )
    ctx = RequestContext(user=UserIdentifier("default", "u"), role=Role.ROOT)
    provider = SessionExtractContextProvider(
        messages=[], ctx=ctx, viking_fs=fs, memory_registry=registry
    )
    extract_context = provider.get_extract_context()
    isolation = MemoryIsolationHandler(
        ctx, extract_context, allowed_memory_types={"cases"}, allow_self=invalid != "disabled"
    )
    isolation.prepare_messages()
    provider._isolation_handler = isolation
    payload = {}
    ids = []
    for name, memory in zip(["first", "second"], memories, strict=True):
        page_id = extract_context.page_id_map.get_page_id(memory.uri)
        ids.append(page_id)
        provider.read_file_contents[memory.uri] = memory
        payload[name] = {**memory.to_metadata(), "page_id": page_id}
    payload["prefetched_search"] = [payload["first"], payload["second"]]
    if invalid == "unread":
        del provider.read_file_contents[memories[1].uri]
    prefetch = []
    add_tool_call_pair_to_messages(
        prefetch,
        "prefetch",
        "eval",
        {
            "code": "\n".join(
                f"{name} = sdk.read(uri={memory.uri!r})"
                for name, memory in zip(["first", "second"], memories, strict=True)
            )
            + "\nprefetched_search = [first, second]"
        },
        payload,
    )
    provider.prefetch = AsyncMock(return_value=prefetch)
    provider.get_tools = lambda: []
    provider.instruction = lambda: "Select the second business domain, not every search candidate."
    if output_format == "python":
        code = "sdk.commit(cases=[second, second])"
    else:
        result = {"selected_case_ids": [ids[1], ids[1]]}
        if invalid == "deleted":
            result["delete_ids"] = [{"delete_page_id": ids[1], "replacement_page_id": None}]
        code = json.dumps(result)
    model = SimpleNamespace(model="offline", get_completion_async=AsyncMock(return_value=code))
    loop = ExtractLoop(
        vlm=model,
        viking_fs=fs,
        ctx=ctx,
        context_provider=provider,
        isolation_handler=isolation,
        max_iterations=1,
    )
    return loop, model, files, fs, ctx, registry, memories


@pytest.mark.asyncio
@pytest.mark.parametrize("execution_count", [None, 7])
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_count_only_case_update_associates_execution_without_explicit_selection(
    monkeypatch, execution_count, output_format
):
    loop, model, files, fs, ctx, _registry, memories = selection_loop(
        monkeypatch, output_format=output_format, execution_count=execution_count
    )
    target = memories[1]
    if output_format == "python":
        # No alias assignment or commit(cases=...): the update itself must associate the Case.
        program = "prefetched_search[1].update(execution_count=1)"
    else:
        page_id = loop.context_provider.get_extract_context().page_id_map.get_page_id(target.uri)
        program = json.dumps(
            {
                "cases": [
                    {
                        "page_id": page_id,
                        **{
                            name: target.extra_fields[name]
                            for name in ("case_name", "domain", "description")
                        },
                        "execution_count": 1,
                    }
                ]
            }
        )
    model.get_completion_async.return_value = program

    operations, _ = await loop.run()

    assert operations.errors == []
    assert operations.case_references == []
    assert operations.delete_file_contents == []
    (update,) = operations.upsert_operations
    assert update.uris == [target.uri]
    assert not update.create_only
    assert update.memory_fields["execution_count"] == 1  # Delta, not old total + 1.
    assert update.memory_fields.get("rubric") is None
    model.get_completion_async.assert_awaited_once()

    # Apply the schema's SUM semantics and resolve the Case passed to trajectory extraction.
    target.extra_fields["execution_count"] = await SumOp().apply(
        execution_count, update.memory_fields["execution_count"]
    )
    assert target.extra_fields["execution_count"] == (execution_count or 0) + 1
    files[target.uri] = MemoryFileUtils.write(target)
    apply_result = MemoryUpdateResult()
    apply_result.add_edited(target.uri)
    cases = await _canonical_cases_from_update_result(
        operations=operations,
        requested_operations=operations,
        result=apply_result,
        viking_fs=fs,
        ctx=ctx,
    )
    assert [case.name for case in cases] == ["second"]
    assert cases[0].metadata["memory_fields"]["execution_count"] == (execution_count or 0) + 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_format, program",
    [
        ("python", None),
        ("json", None),
        ("python", "chosen = prefetched_search[1]"),
        ("python", "chosen = prefetched_search[1]\nsdk.commit()"),
        ("python", "chosen = prefetched_search[1]\nalias = chosen"),
    ],
)
async def test_selected_case_without_writes_reaches_compressor_and_does_not_select_all_candidates(
    monkeypatch, output_format, program
):
    loop, model, files, fs, ctx, registry, memories = selection_loop(
        monkeypatch, output_format=output_format
    )
    if program is not None:
        model.get_completion_async.return_value = program
    before = dict(files)
    operations, _ = await loop.run()
    assert operations.upsert_operations == [] and operations.delete_file_contents == []
    assert [ref.uris for ref in operations.case_references] == [[memories[1].uri]]
    model.get_completion_async.assert_awaited_once()
    updater = StreamingMemoryUpdater(registry=registry)
    monkeypatch.setattr(
        StreamingMemoryUpdater,
        "_apply_operations",
        AsyncMock(side_effect=AssertionError("no write batch")),
    )
    try:
        result = await updater.submit(
            MemoryUpdateRequest(
                operations=operations,
                messages=[],
                ctx=ctx,
                metadata={"source_extraction_id": "this_execution", "session_id": "s"},
            )
        )
    finally:
        await updater.close()
    assert result.apply_result.written_uris == [] and result.apply_result.edited_uris == []
    assert updater._group_batchers == {}
    cases = await _canonical_cases_from_update_result(
        operations=result.operations,
        requested_operations=operations,
        result=result.apply_result,
        viking_fs=fs,
        ctx=ctx,
    )
    assert [case.name for case in cases] == ["second"]
    assert "OBSOLETE_CASE_EVIDENCE" not in json.dumps(asdict(cases[0]))
    assert "evidence" not in cases[0].metadata
    assert files == before
    fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["foreign", "wrong_type", "unread", "deleted"])
async def test_case_selection_rejects_invalid_or_out_of_scope_page_ids(monkeypatch, invalid):
    loop, _model, files, fs, _ctx, _registry, _memories = selection_loop(
        monkeypatch, output_format="json", invalid=invalid
    )
    before = copy.deepcopy(files)
    with pytest.raises(ValueError, match="Selected Case"):
        await loop.run()
    assert files == before
    fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_case_schema_does_not_offer_or_accept_selection(monkeypatch):
    loop, model, files, fs, _ctx, _registry, _memories = selection_loop(
        monkeypatch, output_format="json", invalid="disabled"
    )
    before = dict(files)
    result, _ = await loop.run()
    assert result is None or not result.case_references
    assert "selected_case_ids" not in loop._operations_model.model_fields
    assert "sdk.commit(cases=" not in loop._output_protocol.render_contract(loop._output_context)
    fs.write_file.assert_not_awaited()
    assert files == before


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_reading_candidates_without_selection_does_not_associate_them(
    monkeypatch, output_format
):
    loop, model, files, fs, _ctx, _registry, _memories = selection_loop(
        monkeypatch, output_format=output_format
    )
    model.get_completion_async.return_value = "sdk.commit()" if output_format == "python" else "{}"
    before = dict(files)
    result, _ = await loop.run()
    assert result is None or (not result.case_references and not result.upsert_operations)
    assert files == before
    fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "program",
    [
        "sdk.commit(cases=second)",
        "sdk.commit(cases=[1])",
        "sdk.commit(cases=['viking://user/u/memories/cases/second.md'])",
        "second.delete()\nsdk.commit(cases=[second])",
        "created = sdk.create_cases(case_name='new', domain='New domain', description='{}', rubric='[]', execution_count=1)\nsdk.commit(cases=[created])",
    ],
)
async def test_python_case_selection_rejects_invalid_objects(monkeypatch, program):
    loop, model, files, fs, _ctx, _registry, _memories = selection_loop(
        monkeypatch, output_format="python"
    )
    # Initialize the real output context and binding table with a valid selection.
    await loop.run()
    before = dict(files)
    operations, error = loop._output_protocol.parse(program, loop._output_context)
    assert operations is None
    assert error
    assert files == before
    fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("page_id", [True, "2", 2.5, {"page_id": 2}])
async def test_json_case_selection_does_not_coerce_invalid_page_ids(monkeypatch, page_id):
    loop, model, files, fs, _ctx, _registry, _memories = selection_loop(
        monkeypatch, output_format="json"
    )
    await loop.run()
    before = dict(files)
    operations, error = loop._output_protocol.parse(
        json.dumps({"selected_case_ids": [page_id]}), loop._output_context
    )
    assert operations is None
    assert error
    assert files == before
    fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "program, selected",
    [
        ("existing_case = prefetched_search[0]", [0]),
        ("a = first\nb = second", [0, 1]),
        ("a = first\nb = second\nsdk.commit(cases=[b])", [1]),
        ("chosen = second\nsdk.commit(cases=[])", []),
        ("chosen = second\nchosen.delete()", []),
        ("chosen = second\nchosen.rubric.overwrite('[]')", []),
        ("existing_case = {'memory_type': 'cases', 'page_id': 2}", []),
    ],
)
async def test_python_case_binding_respects_explicit_override_and_write_operations(
    monkeypatch, program, selected
):
    loop, model, files, fs, _ctx, _registry, memories = selection_loop(
        monkeypatch, output_format="python"
    )
    model.get_completion_async.return_value = program
    before = dict(files)
    result, _ = await loop.run()
    assert [ref.uris for ref in result.case_references] == [[memories[i].uri] for i in selected]
    if ".rubric.overwrite" in program:
        assert len(result.upsert_operations) == 1
    if ".delete()" in program:
        assert len(result.delete_file_contents) == 1
    assert files == before
    fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_python_implicit_case_selection_still_enforces_isolation(monkeypatch):
    loop, model, files, fs, _ctx, _registry, _memories = selection_loop(
        monkeypatch, output_format="python", invalid="foreign"
    )
    model.get_completion_async.return_value = "chosen = prefetched_search[1]"
    before = dict(files)
    with pytest.raises(ValueError, match="Selected Case"):
        await loop.run()
    assert files == before
    fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_binding_contract_is_only_for_case_selection(monkeypatch):
    loop, _model, _files, _fs, _ctx, _registry, _memories = selection_loop(
        monkeypatch, output_format="python"
    )
    await loop.run()
    contract = loop._output_protocol.render_contract(loop._output_context)
    assert "Binding a matched, loaded Case to a local variable selects it" in contract
    assert "override" in contract and "cases=[]" in contract


@pytest.mark.parametrize(
    "memory_type, fields, field",
    [
        ("entities", {"category": "people", "name": "alice"}, "content"),
        (
            "events",
            {
                "event_name": "visit",
                "goal": "Meet Alice",
                "summary": "Original",
                "ranges": "0",
            },
            "summary",
        ),
    ],
)
def test_event_and_entity_bindings_still_require_writes_not_case_selection(
    memory_type, fields, field
):
    from test_extraction_output_protocol import _bind, _context

    from openviking.session.memory.extraction_output_protocol import (
        create_extraction_output_protocol,
    )

    memory = MemoryFile(
        uri=f"viking://user/u/memories/{memory_type}/known.md",
        memory_type=memory_type,
        content="Original",
        extra_fields=fields,
    )
    context = _context(
        [get_default_registry().get("cases"), get_default_registry().get(memory_type)],
        files=[case_memory("candidate"), memory],
        template_context={"language": "en"},
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    binding = f"existing_case = {memory_type}_1"
    read_only, error = protocol.parse(binding, context)
    assert error is None and read_only.is_empty()
    assert read_only.selected_case_ids == []
    updated, error = protocol.parse(
        binding + f"\nexisting_case.{field}.overwrite('Changed')", context
    )
    assert error is None and updated.selected_case_ids == []
    assert updated.model_dump()[memory_type][0][field] == "Changed"


@pytest.mark.asyncio
async def test_new_case_binding_still_uses_creation_not_a_reference(monkeypatch):
    loop, model, _files, fs, _ctx, _registry, _memories = selection_loop(
        monkeypatch, output_format="python"
    )
    fs._uri_to_path = lambda uri, **kwargs: uri
    model.get_completion_async.return_value = "created = sdk.create_cases(case_name='new', domain='New domain', description='{}', rubric='[]', execution_count=1)"
    result, _ = await loop.run()
    assert not result.case_references
    assert len(result.upsert_operations) == 1
    assert result.upsert_operations[0].memory_type == "cases"
    fs.write_file.assert_not_awaited()
