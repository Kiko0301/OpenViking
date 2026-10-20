from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_memory_create_guard import CTX, MemoryFS

from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v3 import _canonical_cases_from_update_result
from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation, ResolvedOperations
from openviking.session.memory.experience.rubric import normalize_case_rubric
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry, get_default_registry
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking.session.memory.streaming_memory_updater import (
    MemoryUpdateRequest,
    StreamingMemoryUpdater,
    StreamingMemoryUpdaterConfig,
)
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config import get_openviking_config

ROOT = "viking://user/alice/memories/cases"


def fields(name):
    return {
        "case_name": name,
        "execution_count": 1,
        "domain": (
            "Library lending and returns" if name == "library" else "Airline booking services"
        ),
        "description": "Current service context and supplied user information.",
        "rubric": '[{"id":"scope","description":"Respect the request scope and confirm changes."}]',
    }


def request(name, source, *, ctx=CTX):
    return MemoryUpdateRequest(
        operations=ResolvedOperations(
            upsert_operations=[
                ResolvedOperation(
                    memory_type="cases",
                    memory_fields=fields(name),
                    uris=[f"viking://user/{ctx.user.user_id}/memories/cases/{name}.md"],
                    create_only=True,
                )
            ],
            delete_file_contents=[],
            errors=[],
        ),
        messages=[],
        ctx=ctx,
        metadata={"source_extraction_id": source, "session_id": source},
    )


def create_program(name):
    return (
        "created = sdk.create_cases("
        + ", ".join(f"{k}={v!r}" for k, v in fields(name).items())
        + ")\nsdk.commit(case_sources=[(created, [1])])"
    )


@pytest.fixture
async def setup_merge(monkeypatch):
    monkeypatch.setattr(get_openviking_config().memory, "extraction_output_format", "python")
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.register(get_default_registry().get("cases"))
    fs = MemoryFS()
    fs.search = AsyncMock(return_value=SimpleNamespace(to_dict=lambda: {"memories": []}))
    monkeypatch.setattr(
        "openviking.session.memory.streaming_memory_updater.safe_get_viking_fs", lambda: fs
    )
    monkeypatch.setattr(MemoryUpdater, "_get_viking_fs", lambda self: fs)
    monkeypatch.setattr(MemoryUpdater, "_sync_resource_refs_for_result", AsyncMock())
    monkeypatch.setattr(MemoryUpdater, "_vectorize_memories", AsyncMock())
    monkeypatch.setattr(MemoryUpdater, "generate_overview", AsyncMock())
    model = SimpleNamespace(
        model="offline",
        max_tokens=None,
        get_completion_async=AsyncMock(
            return_value="sdk.commit(case_sources=[(prefetched_memory_1, [1])])"
        ),
    )
    resolver = SimpleNamespace(get_vlm=AsyncMock(return_value=model))
    updater = StreamingMemoryUpdater(
        registry=registry,
        vlm_resolver=resolver,
        config=StreamingMemoryUpdaterConfig(max_operations_per_update=1),
    )
    try:
        yield SimpleNamespace(
            fs=fs, model=model, resolver=resolver, updater=updater, registry=registry
        )
    finally:
        await updater.close()


async def submit(env, req):
    return await asyncio.wait_for(env.updater.submit(req), timeout=3)


@pytest.mark.asyncio
async def test_singletons_reuse_recent_canonical_case_despite_empty_index(setup_merge):
    env = setup_merge
    a = request("airline", "source-a")
    result_a = await submit(env, a)
    assert result_a.apply_result.written_uris == [f"{ROOT}/airline.md"]
    env.model.get_completion_async.assert_not_awaited()
    env.resolver.get_vlm.assert_not_awaited()
    uri = f"{ROOT}/airline.md"
    latest = MemoryFileUtils.read(env.fs.files[uri], uri=uri)
    latest.extra_fields["description"] = "Fresh domain context written after the previous batch."
    env.fs.files[uri] = MemoryFileUtils.write(latest, env.registry.get("cases").content_template)
    original = env.fs.files[uri]
    for name, source in [("airline_service", "source-b"), ("airline_booking", "source-c")]:
        req = request(name, source)
        result = await submit(env, req)
        assert result.apply_result.written_uris == result.apply_result.edited_uris == []
        assert result.operations.upsert_operations == []
        (reference,) = result.operations.case_references
        assert reference.uris == [f"{ROOT}/airline.md"]
        assert reference.memory_fields["source_extraction_ids"] == [source]
        cases = await _canonical_cases_from_update_result(
            operations=result.operations,
            requested_operations=req.operations,
            result=result.apply_result,
            viking_fs=env.fs,
            ctx=CTX,
        )
        assert [case.name for case in cases] == ["airline"]
        assert cases[0].input["summary"] == latest.extra_fields["description"]
        assert env.fs.files == {f"{ROOT}/airline.md": original}
    assert env.fs.search.await_count == 3
    assert env.model.get_completion_async.await_count == 2


@pytest.mark.asyncio
async def test_different_domain_can_create_and_reading_candidates_does_not_select_them(setup_merge):
    env = setup_merge
    await submit(env, request("airline", "a"))
    env.model.get_completion_async.return_value = create_program("library")
    result = await submit(env, request("library", "d"))
    assert result.apply_result.written_uris == [f"{ROOT}/library.md"]
    assert result.operations.case_references == []
    # Both recently committed files are visible, but only the explicitly selected one
    # is associated with this submitter. No other domain is silently attached.
    env.model.get_completion_async.return_value = (
        "sdk.commit(case_sources=[(prefetched_memory_1, [1])])"
    )
    result = await submit(env, request("airline_alias", "b"))
    assert [r.uris for r in result.operations.case_references] == [[f"{ROOT}/airline.md"]]
    assert set(env.fs.files) == {f"{ROOT}/airline.md", f"{ROOT}/library.md"}
    model_input = str(env.model.get_completion_async.await_args.kwargs["messages"])
    assert "airline.md" in model_input and "library.md" in model_input


@pytest.mark.asyncio
async def test_fresh_index_candidates_are_read_without_recent_batch_history(setup_merge):
    env = setup_merge
    uri = f"{ROOT}/airline.md"
    env.fs.files[uri] = MemoryFileUtils.write(
        MemoryFile(uri=uri, memory_type="cases", content="domain", extra_fields=fields("airline"))
    )
    env.fs.search.return_value = SimpleNamespace(
        to_dict=lambda: {"memories": [{"uri": uri, "score": 0.9}]}
    )
    result = await submit(env, request("alias", "b"))
    assert [ref.uris for ref in result.operations.case_references] == [[uri]]
    assert set(env.fs.files) == {uri}
    env.model.get_completion_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_recent_hints_never_cross_user_scope(setup_merge):
    env = setup_merge
    await submit(env, request("airline", "a"))
    # Even if an internal caller reused an updater, URI hints cannot expand its read scope.
    bob = RequestContext(user=UserIdentifier("acme", "bob"), role=Role.USER)
    result = await submit(env, request("different_domain", "b", ctx=bob))
    assert result.apply_result.written_uris == [
        "viking://user/bob/memories/cases/different_domain.md"
    ]
    assert result.operations.case_references == []
    env.model.get_completion_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_recent_candidate_is_not_treated_as_an_existing_case(setup_merge):
    env = setup_merge
    await submit(env, request("airline", "a"))
    del env.fs.files[f"{ROOT}/airline.md"]
    result = await submit(env, request("replacement", "b"))
    assert result.apply_result.written_uris == [f"{ROOT}/replacement.md"]
    env.model.get_completion_async.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["search", "read", "model"])
async def test_candidate_failures_do_not_fall_back_to_creating_duplicate_case(setup_merge, failure):
    env = setup_merge
    await submit(env, request("airline", "a"))
    before = dict(env.fs.files)
    if failure == "search":
        env.fs.search.side_effect = RuntimeError("search unavailable")
    elif failure == "read":
        env.fs.read_file = AsyncMock(side_effect=PermissionError("read denied"))
    else:
        env.model.get_completion_async.side_effect = RuntimeError("model unavailable")
    with pytest.raises((RuntimeError, ValueError)):
        await submit(env, request("alias", "b"))
    assert env.fs.files == before


@pytest.mark.asyncio
async def test_later_aliases_share_a_batch_and_each_keeps_the_canonical_case(setup_merge):
    env = setup_merge
    await submit(env, request("airline", "source-a"))
    batcher = next(iter(env.updater._group_batchers.values()))
    batcher.config.max_items_per_batch = 2
    requests = [request("airline_service", "source-b"), request("airline_booking", "source-c")]
    env.model.get_completion_async.return_value = (
        "sdk.commit(case_sources=[(prefetched_memory_1, [1, 2])])"
    )
    results = await asyncio.wait_for(
        asyncio.gather(*(env.updater.submit(req) for req in requests)), timeout=3
    )
    env.model.get_completion_async.assert_awaited_once()
    for req, result in zip(requests, results, strict=True):
        assert result.metadata["batch_request_count"] == 2
        (reference,) = result.operations.case_references
        assert set(reference.memory_fields["source_extraction_ids"]) == {"source-b", "source-c"}
        cases = await _canonical_cases_from_update_result(
            operations=result.operations,
            requested_operations=req.operations,
            result=result.apply_result,
            viking_fs=env.fs,
            ctx=CTX,
        )
        assert [case.name for case in cases] == ["airline"]
    assert set(env.fs.files) == {f"{ROOT}/airline.md"}


@pytest.mark.asyncio
async def test_canonical_and_alias_inputs_all_receive_the_unchanged_case(setup_merge):
    env = setup_merge
    await submit(env, request("airline", "seed"))
    before = dict(env.fs.files)
    names = ["airline"] * 44 + ["reservation_management"] * 2 + ["reservation_services"]
    requests = [request(name, f"source-{index}") for index, name in enumerate(names)]
    batcher = next(iter(env.updater._group_batchers.values()))
    batcher.config.max_items_per_batch = len(requests)
    env.model.get_completion_async.return_value = (
        f"sdk.commit(case_sources=[(prefetched_memory_1, {list(range(1, 48))!r})])"
    )
    results = await asyncio.wait_for(
        asyncio.gather(*(env.updater.submit(req) for req in requests)), timeout=3
    )
    for req, result in zip(requests, results, strict=True):
        cases = await _canonical_cases_from_update_result(
            operations=result.operations,
            requested_operations=req.operations,
            result=result.apply_result,
            viking_fs=env.fs,
            ctx=CTX,
        )
        assert [case.name for case in cases] == ["airline"]
        assert result.apply_result.written_uris == result.apply_result.edited_uris == []
    env.model.get_completion_async.assert_awaited_once()
    assert env.fs.files == before


@pytest.mark.asyncio
async def test_concurrent_count_only_updates_preserve_deltas_and_case_association(setup_merge):
    env = setup_merge
    uri = f"{ROOT}/airline.md"
    initial = MemoryFile(
        uri=uri,
        memory_type="cases",
        content="",
        extra_fields={**fields("airline"), "execution_count": 7},
    )
    initial.extra_fields["rubric"] = normalize_case_rubric(initial.extra_fields["rubric"])
    env.fs.files[uri] = MemoryFileUtils.write(initial, env.registry.get("cases").content_template)
    initial_content = MemoryFileUtils.read(env.fs.files[uri], uri=uri).content
    requests = []
    for index in range(8):
        req = request("airline", f"execution-{index}")
        operation = req.operations.upsert_operations[0]
        operation.create_only = False
        operation.old_memory_file_content = initial
        operation.memory_fields = {
            key: value for key, value in fields("airline").items() if key != "rubric"
        }
        requests.append(req)

    env.model.get_completion_async.return_value = (
        "prefetched_memory_1.update(execution_count=8)\n"
        "sdk.commit(case_sources=[(prefetched_memory_1, [1, 2, 3, 4, 5, 6, 7, 8])])"
    )
    results = await asyncio.gather(*(submit(env, req) for req in requests))

    stored = MemoryFileUtils.read(env.fs.files[uri], uri=uri)
    assert stored.extra_fields["execution_count"] == 15
    assert stored.content == initial_content
    for name in ("domain", "description"):
        assert stored.extra_fields[name] == initial.extra_fields[name]
    env.model.get_completion_async.assert_awaited_once()
    for req, result in zip(requests, results, strict=True):
        assert not result.apply_result.errors
        cases = await _canonical_cases_from_update_result(
            operations=result.operations,
            requested_operations=req.operations,
            result=result.apply_result,
            viking_fs=env.fs,
            ctx=CTX,
        )
        assert [case.name for case in cases] == ["airline"]
        assert cases[0].task_signature == initial.extra_fields["domain"]
        assert cases[0].input == {"summary": initial.extra_fields["description"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("edit", ["patch", "overwrite"])
async def test_context_updates_preserve_identity_count_and_all_case_sources(setup_merge, edit):
    env = setup_merge
    uri = f"{ROOT}/airline.md"
    original = {**fields("airline"), "execution_count": 7}
    original["rubric"] = normalize_case_rubric(original["rubric"])
    initial = MemoryFile(uri=uri, memory_type="cases", content="", extra_fields=original)
    env.fs.files[uri] = MemoryFileUtils.write(initial, env.registry.get("cases").content_template)
    updates = {
        "domain": "Airline booking and cancellation services",
        "description": "Use current airline service records and preserve user authorization.",
    }
    requests = []
    for index in range(8):
        req = request("airline", f"execution-{index}")
        operation = req.operations.upsert_operations[0]
        operation.create_only = False
        operation.old_memory_file_content = initial
        operation.memory_fields = {**fields("airline"), **updates}
        requests.append(req)
    calls = [
        f"prefetched_memory_1.{name}.edit(search={original[name]!r}, replace={value!r})"
        if edit == "patch"
        else f"prefetched_memory_1.{name}.overwrite({value!r})"
        for name, value in updates.items()
    ]
    env.model.get_completion_async.return_value = "\n".join(
        [
            *calls,
            "prefetched_memory_1.update(execution_count=8)",
            "sdk.commit(case_sources=[(prefetched_memory_1, [1,2,3,4,5,6,7,8])])",
        ]
    )

    results = await asyncio.gather(*(submit(env, req) for req in requests))

    stored = MemoryFileUtils.read(env.fs.files[uri], uri=uri)
    assert stored.extra_fields["case_name"] == "airline"
    assert stored.extra_fields["execution_count"] == 15
    for name, value in updates.items():
        assert stored.extra_fields[name] == value and value in stored.content
    assert stored.extra_fields["rubric"] == original["rubric"]
    env.model.get_completion_async.assert_awaited_once()
    for req, result in zip(requests, results, strict=True):
        assert not result.apply_result.errors
        cases = await _canonical_cases_from_update_result(
            operations=result.operations,
            requested_operations=req.operations,
            result=result.apply_result,
            viking_fs=env.fs,
            ctx=CTX,
        )
        assert [case.name for case in cases] == ["airline"]
        assert cases[0].task_signature == updates["domain"]
        assert cases[0].input == {"summary": updates["description"]}
