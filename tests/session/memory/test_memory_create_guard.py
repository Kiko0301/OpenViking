from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v3 import SessionCompressorV3
from openviking.session.memory.dataclass import (
    MemoryField,
    MemoryFile,
    MemoryTypeSchema,
    ResolvedOperation,
    ResolvedOperations,
)
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.extraction_output_protocol import (
    ExtractionOutputContext,
    create_extraction_output_protocol,
)
from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry, get_default_registry
from openviking.session.memory.memory_updater import (
    ExtractContext,
    MemoryUpdater,
    MemoryUpdateResult,
)
from openviking.session.memory.merge_op import FieldType, MergeOp
from openviking.session.memory.schema_model_generator import SchemaModelGenerator
from openviking.session.memory.streaming_memory_updater import (
    MemoryUpdateRequest,
    StreamingMemoryUpdater,
    StreamingMemoryUpdaterConfig,
    clone_operation_for_uri,
)
from openviking.session.memory.tools import add_tool_call_pair_to_messages
from openviking.session.memory.utils import MemoryFileUtils
from openviking.session.memory.utils.uri import generate_uri
from openviking.session.memory.validation.create import (
    MemoryCreateConflictError,
    validate_create_operations,
)
from openviking_cli.exceptions import ConflictError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier

URI = "viking://user/alice/memories/records/a.md"
CTX = RequestContext(user=UserIdentifier("acme", "alice"), role=Role.USER)


class MemoryFS:
    def __init__(self, files=None):
        self.files = dict(files or {})
        self.lock = asyncio.Lock()
        self._async_agfs = SimpleNamespace(
            pathlock_acquire_exact_batch=self.acquire, pathlock_release=self.release
        )

    def _uri_to_path(self, uri, *, ctx=None):
        return uri.replace("viking://", "/")

    async def exists(self, uri, *, ctx=None):
        return uri in self.files

    async def read_file(self, uri, *, ctx=None):
        if uri not in self.files:
            raise NotFoundError(f"File not found: {uri}")
        return self.files[uri]

    async def write_file(self, uri, content, *, ctx=None, lease_ref=None):
        assert self.lock.locked() and lease_ref == {"lease": "test"}
        self.files[uri] = content

    async def acquire(self, paths, **kwargs):
        await self.lock.acquire()
        return {"lease": "test"}

    async def release(self, lease):
        self.lock.release()


def schema():
    return MemoryTypeSchema(
        memory_type="records",
        directory="viking://user/{{ user_space }}/memories/records",
        filename_template="{{ key }}.md",
        peer_enabled=False,
        fields=[
            MemoryField(name="key", field_type=FieldType.STRING, merge_op=MergeOp.IMMUTABLE),
            MemoryField(name="content", field_type=FieldType.STRING, merge_op=MergeOp.PATCH),
        ],
    )


def create(uri=URI, content="new", *, create_only=True):
    return ResolvedOperation(
        memory_type="records",
        uris=[uri],
        memory_fields={"key": "a", "content": content},
        create_only=create_only,
    )


def operations(*items, deletes=()):
    return ResolvedOperations(
        upsert_operations=list(items), delete_file_contents=list(deletes), errors=[]
    )


def old_file():
    return MemoryFile(uri=URI, memory_type="records", content="- keep", extra_fields={"key": "a"})


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "- keep"])
async def test_existing_including_empty_file_rejects_create(content):
    fs = MemoryFS({URI: content})
    with pytest.raises(MemoryCreateConflictError, match="already_exists"):
        await validate_create_operations(operations(create()), fs, CTX)
    assert fs.files == {URI: content}


@pytest.mark.asyncio
@pytest.mark.parametrize("second", [create(), create(create_only=False)])
async def test_duplicate_target_rejected_before_any_write(second):
    with pytest.raises(MemoryCreateConflictError, match="duplicate_target"):
        await validate_create_operations(operations(create(), second), MemoryFS(), CTX)


@pytest.mark.asyncio
async def test_create_delete_same_target_is_not_a_replacement_escape():
    with pytest.raises(MemoryCreateConflictError, match="duplicate_target"):
        await validate_create_operations(
            operations(create(), deletes=[old_file()]), MemoryFS(), CTX
        )


@pytest.mark.asyncio
async def test_absent_create_allowed_and_plain_upsert_remains_allowed():
    await validate_create_operations(operations(create()), MemoryFS(), CTX)
    await validate_create_operations(
        operations(create(create_only=False)), MemoryFS({URI: "old"}), CTX
    )


@pytest.mark.asyncio
async def test_unexpected_storage_error_is_not_nonexistence():
    fs = MemoryFS()
    fs.exists = AsyncMock(side_effect=PermissionError("denied"))
    with pytest.raises(PermissionError, match="denied"):
        await validate_create_operations(operations(create()), fs, CTX)


def test_create_intent_survives_clone_but_never_enters_memory_fields():
    clone = clone_operation_for_uri(create(), URI)
    assert clone.create_only
    assert "create_only" not in clone.model_dump()
    assert "create_only" not in clone.memory_fields


def test_python_records_create_intent_but_not_set_or_edit():
    record_schema = schema()
    singleton = MemoryTypeSchema(
        memory_type="profile",
        directory="viking://user/alice/memories",
        filename_template="profile.md",
        fields=[record_schema.fields[1]],
    )
    model = SchemaModelGenerator([record_schema, singleton]).create_structured_operations_model()
    context = ExtractionOutputContext(
        model, (record_schema, singleton), ExtractContext([]).page_id_map, {}, False
    )
    protocol = create_extraction_output_protocol("python")
    protocol.render_contract(context)
    parsed, error = protocol.parse(
        "sdk.create_records(key='a', content='new')\nsdk.set_profile(content='profile')", context
    )
    assert error is None
    assert context.create_page_ids == {parsed.records[0].page_id}
    assert parsed.profile[0].page_id not in context.create_page_ids
    assert "create_only" not in parsed.model_dump_json()
    parsed, error = protocol.parse("sdk.commit()", context)
    assert error is None and not context.create_page_ids


def make_loop(fs, outputs, *, max_iterations=1):
    record_schema = schema()
    extract_context = ExtractContext([])
    provider = MagicMock()
    provider.prepare_operation_fields = None
    provider.get_memory_schemas.return_value = [record_schema]
    provider.get_output_language.return_value = "en"
    provider.get_tools.return_value = ["eval"]
    provider.customize_tool_schema.side_effect = lambda name, value: value
    provider.get_extract_context.return_value = extract_context
    provider.read_file_contents = {}
    provider.prefetch = AsyncMock(return_value=[])
    provider.format_prefetch_messages.side_effect = lambda messages: messages
    provider.instruction.return_value = "Maintain user constraints."
    provider.validate_operations = AsyncMock(return_value=[])
    provider.execute_tool = AsyncMock(return_value={"error": "Tool not allowed: read"})
    provider.create_tool_context.return_value = SimpleNamespace(
        viking_fs=fs,
        request_ctx=CTX,
        read_file_contents=provider.read_file_contents,
        page_id_map=extract_context.page_id_map,
    )
    requests = []
    remaining = iter(outputs)

    async def completion(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        return next(remaining)

    loop = ExtractLoop(
        vlm=SimpleNamespace(model="test", get_completion_async=completion),
        viking_fs=fs,
        ctx=CTX,
        context_provider=provider,
        isolation_handler=MemoryIsolationHandler(CTX, extract_context),
        max_iterations=max_iterations,
    )
    return loop, requests


@pytest.mark.asyncio
async def test_existing_create_feedback_uses_native_eval_then_edit():
    raw = MemoryFileUtils.write(old_file())
    fs = MemoryFS({URI: raw})
    loop, requests = make_loop(
        fs,
        [
            "sdk.create_records(key='a', content='- new')",
            "refetched_memory_1.content.edit(search='- keep', replace='- keep\\n- new')",
        ],
    )
    resolved, _ = await loop.run()
    assert len(requests) == 2
    assert any(m["role"] == "user" and "already_exists" in m["content"] for m in requests[1])
    calls = [c for m in requests[1] for c in m.get("tool_calls", [])]
    assert any(
        c["function"]["name"] == "eval" and "sdk.read" in c["function"]["arguments"] for c in calls
    )
    assert any(m["role"] == "tool" and "sdk.existing" in m["content"] for m in requests[1])
    assert resolved.upsert_operations[0].is_edit()
    assert not resolved.upsert_operations[0].create_only
    assert fs.files[URI] == raw


@pytest.mark.asyncio
async def test_repeated_create_conflict_has_bounded_repair():
    fs = MemoryFS({URI: MemoryFileUtils.write(old_file())})
    code = "sdk.create_records(key='a', content='- new')"
    loop, requests = make_loop(fs, [code, code])
    with pytest.raises(MemoryCreateConflictError, match="already_exists"):
        await loop.run()
    assert len(requests) == 2 and "- keep" in fs.files[URI]


@pytest.mark.asyncio
async def test_duplicate_new_create_can_repair_to_one_create():
    code = "sdk.create_records(key='a', content='x')\nsdk.create_records(key='a', content='y')"
    loop, requests = make_loop(MemoryFS(), [code, "sdk.create_records(key='a', content='x\\ny')"])
    loop.max_iterations = 2
    resolved, _ = await loop.run()
    assert len(requests) == 2 and len(resolved.upsert_operations) == 1
    assert resolved.upsert_operations[0].create_only
    assert any(m["role"] == "user" and "duplicate_target" in m["content"] for m in requests[1])


def guarded_updater(fs):
    updater = MemoryUpdater(registry=MemoryTypeRegistry(load_schemas=False))
    updater._get_viking_fs = lambda: fs

    async def apply(ops, ctx, *args, lease_ref=None):
        result = MemoryUpdateResult()
        for op in ops.upsert_operations:
            await fs.write_file(
                op.uris[0], op.memory_fields["content"], ctx=ctx, lease_ref=lease_ref
            )
            result.add_written(op.uris[0])
        return result

    updater._apply_operations = AsyncMock(side_effect=apply)
    return updater


@pytest.mark.asyncio
async def test_locked_apply_rejects_entire_candidate_before_unrelated_writes():
    fs = MemoryFS({URI: "old"})
    updater = guarded_updater(fs)
    with pytest.raises(MemoryCreateConflictError):
        await updater.apply_operations(
            operations(create(URI.replace("a.md", "b.md")), create()), CTX
        )
    updater._apply_operations.assert_not_awaited()
    assert fs.files == {URI: "old"} and not fs.lock.locked()


@pytest.mark.asyncio
async def test_concurrent_creators_have_one_winner_without_overwrite():
    fs = MemoryFS()
    first, second = guarded_updater(fs), guarded_updater(fs)
    outcomes = await asyncio.gather(
        first.apply_operations(operations(create(content="first")), CTX),
        second.apply_operations(operations(create(content="second")), CTX),
        return_exceptions=True,
    )
    assert sum(isinstance(x, MemoryCreateConflictError) for x in outcomes) == 1
    assert fs.files == {URI: "first"} and not fs.lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        MemoryCreateConflictError([{"uri": URI, "code": "already_exists"}]),
        ConflictError("Selected memory cannot be deleted", resource=URI),
    ],
)
async def test_memory_conflict_not_swallowed_when_strict_extract_errors_is_false(error):
    compressor = object.__new__(SessionCompressorV3)
    compressor._extract_user_memories = AsyncMock(side_effect=error)
    with pytest.raises(type(error)) as raised:
        await compressor.extract_long_term_memories([], ctx=CTX, strict_extract_errors=False)
    assert raised.value is error


@pytest.mark.asyncio
async def test_late_create_race_is_rechecked_at_apply():
    fs = MemoryFS()
    candidate = operations(create())
    await validate_create_operations(candidate, fs, CTX)
    fs.files[URI] = "written by another commit"
    updater = guarded_updater(fs)
    with pytest.raises(MemoryCreateConflictError, match="already_exists"):
        await updater.apply_operations(candidate, CTX)
    updater._apply_operations.assert_not_awaited()
    assert fs.files[URI] == "written by another commit"


@pytest.mark.asyncio
async def test_merge_preserves_existing_create_for_locked_apply_guard(monkeypatch):
    fs = MemoryFS({URI: "old"})
    monkeypatch.setattr(
        "openviking.session.memory.streaming_memory_updater.safe_get_viking_fs", lambda: fs
    )
    merge = AsyncMock(side_effect=lambda **kwargs: kwargs["operations"])
    monkeypatch.setattr(
        "openviking.session.memory.streaming_memory_updater.merge_memory_operations", merge
    )
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.register(schema())
    updater = StreamingMemoryUpdater(registry=registry)
    merged = await updater._merge_requests(
        [MemoryUpdateRequest(operations=operations(create()), messages=[], ctx=CTX)]
    )
    merge.assert_awaited_once()
    assert merged.upsert_operations[0].create_only
    writer = guarded_updater(fs)
    with pytest.raises(MemoryCreateConflictError, match="already_exists"):
        await writer.apply_operations(merged, CTX)
    writer._apply_operations.assert_not_awaited()
    assert fs.files[URI] == "old"


@pytest.mark.asyncio
async def test_real_apply_persists_new_file_without_intent_metadata(monkeypatch):
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.register(schema())
    fs = MemoryFS()
    updater = MemoryUpdater(registry=registry)
    updater._get_viking_fs = lambda: fs
    updater._sync_resource_refs_for_result = AsyncMock()
    updater._vectorize_memories = AsyncMock()
    updater.generate_overview = AsyncMock()
    result = await updater.apply_operations(operations(create(content="- new")), CTX)
    assert result.written_uris == [URI] and not result.errors
    stored = MemoryFileUtils.read(fs.files[URI], uri=URI)
    assert stored.plain_content() == "- new"
    assert "create_only" not in fs.files[URI]
    before = fs.files[URI]
    with pytest.raises(MemoryCreateConflictError):
        await updater.apply_operations(operations(create(content="- replacement")), CTX)
    assert fs.files[URI] == before


@pytest.mark.asyncio
@pytest.mark.parametrize("create_only", [True, False])
async def test_merge_preserves_create_intent_for_renamed_outputs(monkeypatch, create_only):
    from openviking.session.memory.streaming_memory_updater import merge_one_memory_type_operations

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.register(schema())
    renamed = create(URI.replace("a.md", "b.md"), create_only=False)
    renamed.memory_fields["key"] = "b"
    provider = MagicMock()
    provider.read_file_contents = {}
    provider.prefetch = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "openviking.session.memory.streaming_memory_updater.PatchMergeContextProvider",
        lambda **kwargs: provider,
    )
    monkeypatch.setattr(
        "openviking.session.memory.streaming_memory_updater.ExtractLoop.run",
        AsyncMock(return_value=(operations(renamed), [])),
    )
    monkeypatch.setattr(
        "openviking.session.memory.streaming_memory_updater.safe_get_viking_fs", lambda: MemoryFS()
    )
    resolver = SimpleNamespace(get_vlm=AsyncMock(return_value=SimpleNamespace(model="test")))
    merged = await merge_one_memory_type_operations(
        memory_type="records",
        operations=[create(create_only=create_only)],
        messages=[],
        ctx=CTX,
        registry=registry,
        vlm_resolver=resolver,
        force_merge=True,
    )
    assert merged.upsert_operations[0].uris == [renamed.uris[0]]
    assert merged.upsert_operations[0].create_only is create_only


@pytest.mark.asyncio
async def test_duplicate_existing_create_reports_both_conflicts_for_one_repair():
    raw = MemoryFileUtils.write(old_file())
    fs = MemoryFS({URI: raw})
    loop, requests = make_loop(
        fs,
        [
            "sdk.create_records(key='a', content='x')\nsdk.create_records(key='a', content='y')",
            "refetched_memory_1.content.edit(search='- keep', replace='- keep\\n- x\\n- y')",
        ],
    )
    resolved, _ = await loop.run()
    errors = [
        m["content"]
        for m in requests[1]
        if m["role"] == "user" and "Memory create failed" in m["content"]
    ]
    assert len(errors) == 1 and "duplicate_target" in errors[0] and "already_exists" in errors[0]
    assert len(requests) == 2 and resolved.upsert_operations[0].is_edit()
    assert fs.files[URI] == raw


@pytest.mark.asyncio
async def test_new_conflicts_keep_refetching_without_one_shot_repair_limit():
    # Match the real failure shape: four targets first, then a fifth target.
    keys = list("abcde")
    original = {
        URI.replace("a.md", f"{key}.md"): MemoryFileUtils.write(
            MemoryFile(
                uri=URI.replace("a.md", f"{key}.md"),
                memory_type="records",
                content=f"- keep {key}",
                extra_fields={"key": key},
            )
        )
        for key in keys
    }
    fs = MemoryFS(original)
    fs.read_file = AsyncMock(wraps=fs.read_file)
    first = "\n".join(f"sdk.create_records(key={key!r}, content='new')" for key in keys[:4])
    edits = [
        f"refetched_memory_{i}.content.overwrite({f'- keep {key} / new {key}'!r})"
        for i, key in enumerate(keys, start=1)
    ]
    second = "\n".join([*edits[:4], "sdk.create_records(key='e', content='new')"])
    loop, requests = make_loop(fs, [first, second, "\n".join(edits)], max_iterations=1)
    resolved, _ = await loop.run()
    assert len(requests) == 3  # One bonus per round of new context, not per file.
    assert fs.read_file.await_count == 5
    assert fs.files == original
    assert {op.uris[0]: op.memory_fields["content"] for op in resolved.upsert_operations} == {
        URI.replace("a.md", f"{key}.md"): f"- keep {key} / new {key}" for key in keys
    }
    assert all(op.is_edit() and not op.create_only for op in resolved.upsert_operations)
    assert "refetched_memory_5" in str(requests[-1])
    for request, count in zip(requests, (0, 4, 5), strict=True):
        indices = [
            message["content"]
            for message in request
            if isinstance(message.get("content"), str)
            and message["content"].startswith("Existing-memory access ONLY via:")
        ]
        expected = ", ".join(f"refetched_memory_{i}" for i in range(1, count + 1)) or "(none)"
        assert len(indices) == 1
        assert f"ONLY via: {expected}." in indices[0]
        assert request[-1]["content"] == indices[0]


@pytest.mark.asyncio
async def test_repeated_known_conflict_uses_remaining_budget_without_refetch_bonus():
    fs = MemoryFS({URI: MemoryFileUtils.write(old_file())})
    fs.read_file = AsyncMock(wraps=fs.read_file)
    code = "sdk.create_records(key='a', content='new')"
    loop, requests = make_loop(fs, [code] * 4, max_iterations=3)
    with pytest.raises(MemoryCreateConflictError, match="already_exists"):
        await loop.run()
    assert len(requests) == 4  # Three normal rounds + the first new-file refetch.
    assert fs.read_file.await_count == 1


@pytest.mark.asyncio
async def test_prefetched_target_gets_only_one_new_conflict_feedback_bonus():
    fs = MemoryFS({URI: MemoryFileUtils.write(old_file())})
    fs.read_file = AsyncMock(wraps=fs.read_file)
    code = "sdk.create_records(key='a', content='new')"
    loop, requests = make_loop(fs, [code] * 4, max_iterations=3)
    loop.context_provider.read_file_contents[URI] = old_file()
    with pytest.raises(MemoryCreateConflictError, match="already_exists"):
        await loop.run()
    assert len(requests) == 4
    fs.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_target_gets_only_one_new_conflict_feedback_bonus():
    fs = MemoryFS()
    fs.read_file = AsyncMock(wraps=fs.read_file)
    code = "sdk.create_records(key='a', content='x')\nsdk.create_records(key='a', content='y')"
    loop, requests = make_loop(fs, [code] * 4, max_iterations=3)
    with pytest.raises(MemoryCreateConflictError, match="duplicate_target"):
        await loop.run()
    assert len(requests) == 4 and fs.files == {}
    fs.read_file.assert_not_awaited()


def prefetch_existing_records(loop, keys):
    provider = loop.context_provider
    page_ids = provider.get_extract_context().page_id_map
    results = {}
    code = []
    for key in keys:
        uri = URI.replace("a.md", f"{key}.md")
        memory = MemoryFile(
            uri=uri, memory_type="records", content=f"- keep {key}", extra_fields={"key": key}
        )
        provider.read_file_contents[uri] = memory
        results[f"existing_{key}"] = {
            "memory_type": "records",
            "key": key,
            "content": memory.content,
            "page_id": page_ids.get_page_id(uri),
        }
        code.append(f"existing_{key} = sdk.read(uri={uri!r})")
    messages = []
    add_tool_call_pair_to_messages(messages, "prefetch", "eval", {"code": "\n".join(code)}, results)
    provider.prefetch = AsyncMock(return_value=messages)


@pytest.mark.asyncio
async def test_last_iteration_known_targets_receive_one_conflict_feedback_round():
    keys = ["a", "b", "c"]
    fs = MemoryFS(
        {
            URI.replace("a.md", f"{key}.md"): MemoryFileUtils.write(
                MemoryFile(
                    uri=URI.replace("a.md", f"{key}.md"),
                    memory_type="records",
                    content=f"- keep {key}",
                    extra_fields={"key": key},
                )
            )
            for key in keys
        }
    )
    original = dict(fs.files)
    fs.read_file = AsyncMock(wraps=fs.read_file)
    first = "\n".join(f"sdk.create_records(key={key!r}, content='new')" for key in keys)
    corrected = "\n".join(
        f"existing_{key}.content.edit(search='- keep {key}', replace='- keep {key} / new')"
        for key in keys
    )
    loop, requests = make_loop(fs, [first, corrected], max_iterations=1)
    prefetch_existing_records(loop, keys)

    resolved, _ = await loop.run()

    assert len(requests) == 2  # One round for all three conflicts, not one per file.
    assert len(resolved.upsert_operations) == 3
    assert all(op.is_edit() and not op.create_only for op in resolved.upsert_operations)
    feedback = [
        m
        for m in requests[1]
        if m["role"] == "user" and "Memory create failed" in m.get("content", "")
    ]
    assert len(feedback) == 1 and feedback[0]["content"].count("already_exists") == 4
    assert any(m["role"] == "assistant" and m.get("content") == first for m in requests[1])
    fs.read_file.assert_not_awaited()
    assert fs.files == original


@pytest.mark.asyncio
async def test_new_known_targets_get_successive_feedback_rounds_without_fixed_cap():
    fs = MemoryFS(
        {
            URI.replace("a.md", f"{key}.md"): MemoryFileUtils.write(
                MemoryFile(
                    uri=URI.replace("a.md", f"{key}.md"),
                    memory_type="records",
                    content=f"- keep {key}",
                    extra_fields={"key": key},
                )
            )
            for key in ["a", "b"]
        }
    )
    fs.read_file = AsyncMock(wraps=fs.read_file)
    edit_a = "existing_a.content.edit(search='- keep a', replace='- keep a / new')"
    edit_b = "existing_b.content.edit(search='- keep b', replace='- keep b / new')"
    loop, requests = make_loop(
        fs,
        [
            "sdk.create_records(key='a', content='new')",
            edit_a + "\nsdk.create_records(key='b', content='new')",
            edit_a + "\n" + edit_b,
        ],
        max_iterations=1,
    )
    prefetch_existing_records(loop, ["a", "b"])

    resolved, _ = await loop.run()

    assert len(requests) == 3 and len(resolved.upsert_operations) == 2
    fs.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_conflict_kind_on_same_target_receives_feedback_round():
    fs = MemoryFS({URI: MemoryFileUtils.write(old_file())})
    fs.read_file = AsyncMock(wraps=fs.read_file)
    first = "sdk.create_records(key='a', content='new')"
    duplicate = first + "\n" + first
    loop, requests = make_loop(fs, [first, duplicate, "sdk.commit()"], max_iterations=1)
    loop.context_provider.read_file_contents[URI] = old_file()

    resolved, _ = await loop.run()

    assert len(requests) == 3 and not resolved.upsert_operations
    assert any(
        "duplicate_target" in m.get("content", "") for m in requests[2] if m["role"] == "user"
    )
    fs.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_case_alias_of_reported_conflict_does_not_refetch_or_extend_budget():
    class CaseInsensitiveFS(MemoryFS):
        async def exists(self, uri, *, ctx=None):
            return await super().exists(uri.lower(), ctx=ctx)

        async def read_file(self, uri, *, ctx=None):
            return await super().read_file(uri.lower(), ctx=ctx)

    fs = CaseInsensitiveFS({URI: MemoryFileUtils.write(old_file())})
    fs.read_file = AsyncMock(wraps=fs.read_file)
    create_lower = "sdk.create_records(key='a', content='new')"
    create_upper = "sdk.create_records(key='A', content='new')"
    loop, requests = make_loop(fs, [create_lower, create_upper, create_upper], max_iterations=1)
    loop.context_provider.read_file_contents[URI] = old_file()

    with pytest.raises(MemoryCreateConflictError, match="already_exists"):
        await loop.run()

    assert len(requests) == 2
    fs.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_reported_create_conflicts_reset_between_runs():
    fs = MemoryFS({URI: MemoryFileUtils.write(old_file())})
    create_code = "sdk.create_records(key='a', content='new')"
    loop, requests = make_loop(
        fs, [create_code, "sdk.commit()", create_code, "sdk.commit()"], max_iterations=1
    )
    loop.context_provider.read_file_contents[URI] = old_file()

    for _ in range(2):
        resolved, _ = await loop.run()
        assert not resolved.upsert_operations
    assert len(requests) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_type", "fields", "text_field"),
    [
        ("entities", {"category": "people", "name": "alice"}, "content"),
        ("preferences", {"user": "alice", "topic": "communication"}, "content"),
        (
            "cases",
            {
                "case_name": "payment_selection",
                "domain": "Choose an eligible payment method",
                "description": '{"summary":"Choose payment"}',
                "rubric": "[]",
            },
            "domain",
        ),
    ],
)
@pytest.mark.parametrize(
    "outcome", ["merged", "existing_at_merge", "duplicate_output", "late_race"]
)
async def test_independent_same_target_creates_merge_before_final_validation(
    monkeypatch, memory_type, fields, text_field, outcome
):
    registry = get_default_registry()
    uri = generate_uri(registry.get(memory_type), fields, user_space="alice")
    fs = MemoryFS()
    monkeypatch.setattr(
        "openviking.session.memory.streaming_memory_updater.safe_get_viking_fs", lambda: fs
    )
    monkeypatch.setattr(MemoryUpdater, "_get_viking_fs", lambda self: fs)
    monkeypatch.setattr(MemoryUpdater, "_sync_resource_refs_for_result", AsyncMock())
    monkeypatch.setattr(MemoryUpdater, "_vectorize_memories", AsyncMock())
    monkeypatch.setattr(MemoryUpdater, "generate_overview", AsyncMock())
    # Only replace model/search work. Batching, merge orchestration, source propagation,
    # create-intent preservation, locks, validation and the memory write remain real.
    monkeypatch.setattr(
        "openviking.session.memory.streaming_memory_updater.PatchMergeContextProvider.prefetch",
        AsyncMock(return_value=[]),
    )
    patches = []

    async def model_output(loop):
        patches.extend(loop.context_provider.patches)
        candidate = ResolvedOperation(
            memory_type=memory_type,
            uris=[uri],
            memory_fields={**fields, text_field: "First fact. Second fact."},
        )
        if outcome == "existing_at_merge":
            # The merger explicitly read an existing object and returned an update.
            candidate.old_memory_file_content = MemoryFileUtils.read(
                await fs.read_file(uri, ctx=CTX), uri=uri
            )
            candidate.memory_fields[text_field] = "Existing fact. First fact. Second fact."
            if memory_type == "cases":
                # Existing domain Cases are selected without any content update.
                return ResolvedOperations(
                    case_references=[candidate],
                    upsert_operations=[],
                    delete_file_contents=[],
                    errors=[],
                ), []
        if outcome == "late_race":
            fs.files[uri] = "Written by another commit after extraction"
        candidates = [candidate]
        if outcome == "duplicate_output":
            candidates.append(candidate.model_copy(deep=True))
        return operations(*candidates), []

    model = AsyncMock(side_effect=model_output)

    # Preserve the instance argument so the test can inspect both merge proposals.
    async def run(loop):
        return await model(loop)

    monkeypatch.setattr(ExtractLoop, "run", run)
    resolver = SimpleNamespace(get_vlm=AsyncMock(return_value=SimpleNamespace(model="test")))
    updater = StreamingMemoryUpdater(
        registry=registry,
        vlm_resolver=resolver,
        config=StreamingMemoryUpdaterConfig(max_operations_per_update=2, max_wait_seconds=60),
    )
    requests = [
        MemoryUpdateRequest(
            operations=operations(
                ResolvedOperation(
                    memory_type=memory_type,
                    uris=[uri],
                    create_only=True,
                    memory_fields={**fields, text_field: fact},
                )
            ),
            messages=[],
            ctx=CTX,
            metadata={"session_id": f"s{index}", "source_extraction_id": f"source{index}"},
        )
        for index, fact in enumerate(["First fact.", "Second fact."])
    ]
    for request in requests:
        await validate_create_operations(request.operations, fs, CTX)
    if outcome == "existing_at_merge":
        # Another batch created the target after extraction but before this merge.
        fs.files[uri] = MemoryFileUtils.write(
            MemoryFile(
                uri=uri,
                memory_type=memory_type,
                content="Existing fact." if text_field == "content" else "",
                extra_fields={**fields, text_field: "Existing fact.", "version": 1},
            )
        )
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(updater.submit(request) for request in requests), return_exceptions=True
            ),
            timeout=5,
        )
    finally:
        await updater.close()
    model.assert_awaited_once()
    assert len(patches) == 2 and {p.target_uri for p in patches} == {uri}
    assert not fs.lock.locked()
    if outcome in {"duplicate_output", "late_race"}:
        assert all(isinstance(result, MemoryCreateConflictError) for result in results)
        code = "duplicate_target" if outcome == "duplicate_output" else "already_exists"
        assert all(result.conflicts == [{"uri": uri, "code": code}] for result in results)
        assert fs.files == (
            {uri: "Written by another commit after extraction"} if outcome == "late_race" else {}
        )
        return
    assert set(fs.files) == {uri}
    memory = MemoryFileUtils.read(fs.files[uri], uri=uri)
    if memory_type == "cases" and outcome == "existing_at_merge":
        assert memory.extra_fields["domain"] == "Existing fact."
        for result in results:
            assert result.apply_result.written_uris == [] and result.apply_result.edited_uris == []
            assert result.operations.upsert_operations == []
            assert len(result.operations.case_references) == 1
            assert set(
                result.operations.case_references[0].memory_fields["source_extraction_ids"]
            ) == {"source0", "source1"}
        return
    assert set(memory.extra_fields["source_extraction_ids"]) == {"source0", "source1"}
    assert "First fact. Second fact." in memory.plain_content()
    if outcome == "existing_at_merge":
        assert "Existing fact." in memory.plain_content()
    for result in results:
        assert result.apply_result.written_uris == ([] if outcome == "existing_at_merge" else [uri])
        assert result.apply_result.edited_uris == ([uri] if outcome == "existing_at_merge" else [])
        assert result.apply_result.errors == []
        assert len(result.operations.upsert_operations) == 1
        assert result.operations.upsert_operations[0].create_only is (
            outcome != "existing_at_merge"
        )
