# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.session.memory.dataclass import MemoryField, MemoryFile, MemoryTypeSchema
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry, get_default_registry
from openviking.session.memory.merge_op import FieldType, MergeOp
from openviking.session.memory.patch_merge_context_provider import (
    PatchMergeContextProvider,
    PatchMergePatch,
)
from openviking.session.skill.session_skill_context_provider import load_skill_extract_registry


def _memory_file(
    *,
    name: str,
    uri: str | None,
    content: str,
    memory_type: str = "experiences",
) -> MemoryFile:
    return MemoryFile(
        uri=uri,
        content=content,
        memory_type=memory_type,
        extra_fields={
            "memory_type": memory_type,
            "experience_name": name,
            "status": "production",
        },
    )


@pytest.mark.asyncio
async def test_patch_merge_context_provider_prefetch_reads_originals_and_renders_patch():
    uri = "viking://user/u/memories/experiences/booking.md"
    provider = PatchMergeContextProvider(
        memory_type="experiences",
        required_file_uris=[uri],
        patches=[
            PatchMergePatch(
                before_file=_memory_file(name="booking", uri=uri, content="old line\nkeep line"),
                after_file=_memory_file(name="booking", uri=uri, content="new line\nkeep line"),
            )
        ],
    )
    provider._eval_memory = AsyncMock(
        return_value={
            "prefetched_memory_1": {
                "memory_type": "experiences",
                "experience_name": "booking",
                "content": "1\told line\n2\tkeep line",
            }
        }
    )

    messages = await provider.prefetch()

    assert provider.get_tools() == []
    provider._eval_memory.assert_awaited_once()
    call = messages[0]["tool_calls"][0]
    assert call["function"]["name"] == "eval"
    assert "prefetched_memory_1 = sdk.read(" in json.loads(call["function"]["arguments"])["code"]
    assert messages[1]["tool_call_id"] == call["id"]
    assert json.loads(messages[1]["content"])["prefetched_memory_1"]["experience_name"] == "booking"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"].startswith("# Memory File Patches")
    assert "Patch 1" in messages[2]["content"]
    # Patch headers should not include target_uri/target_name/memory_type
    assert "target_uri:" not in messages[2]["content"]
    assert "target_name:" not in messages[2]["content"]
    assert "  content:" in messages[2]["content"]
    assert "--- content.before" not in messages[2]["content"]
    assert "+++ content.after" not in messages[2]["content"]
    assert "    -old line" in messages[2]["content"]
    assert "    +new line" in messages[2]["content"]
    assert "     keep line" in messages[2]["content"]  # n=1 context line
    assert "  status:" not in messages[2]["content"]


@pytest.mark.asyncio
async def test_patch_merge_context_provider_prefetch_searches_and_reads_extra_candidates():
    schema = MemoryTypeSchema(
        memory_type="experiences",
        description="Experiences",
        directory="viking://user/{{ user_space }}/memories/experiences",
        filename_template="{{ experience_name }}.md",
        fields=[],
    )
    provider = PatchMergeContextProvider(
        memory_type="experiences",
        required_file_uris=["viking://user/u/memories/experiences/book.md"],
        patches=[
            PatchMergePatch(
                before_file=None,
                after_file=_memory_file(
                    name="books",
                    uri="viking://user/u/memories/experiences/books.md",
                    content="用户喜欢阅读科幻书籍，尤其是太空歌剧。",
                ),
            )
        ],
    )
    provider._registry = SimpleNamespace(get=lambda name: schema if name == "experiences" else None)
    provider._ctx = SimpleNamespace(user=SimpleNamespace(user_id="u"))
    provider.search_files = AsyncMock(
        return_value=[
            "viking://user/u/memories/experiences/book.md",
            *[f"viking://user/u/memories/experiences/candidate_{idx}.md" for idx in range(10)],
        ]
    )
    provider._eval_memory = AsyncMock(return_value={})

    messages = await provider.prefetch()

    provider.search_files.assert_awaited_once()
    _, search_kwargs = provider.search_files.await_args
    assert search_kwargs["search_uris"] == ["viking://user/u/memories/experiences"]
    assert search_kwargs["limit"] == 10
    provider._eval_memory.assert_awaited_once()
    code = provider._eval_memory.await_args.args[0]
    assert "viking://user/u/memories/experiences/book.md" in code
    assert "viking://user/u/memories/experiences/candidate_0.md" in code
    assert "viking://user/u/memories/experiences/candidate_4.md" in code
    assert "viking://user/u/memories/experiences/candidate_5.md" not in code
    assert messages[-1]["content"].startswith("# Memory File Patches")


@pytest.mark.asyncio
async def test_patch_merge_context_provider_skips_extra_candidates_for_existing_files():
    uri = "viking://user/u/memories/experiences/booking.md"
    provider = PatchMergeContextProvider(
        memory_type="experiences",
        required_file_uris=[uri],
        patches=[
            PatchMergePatch(
                before_file=_memory_file(name="booking", uri=uri, content="old line"),
                after_file=_memory_file(name="booking", uri=uri, content="new line"),
            )
        ],
    )
    provider.search_files = AsyncMock(
        return_value=["viking://user/u/memories/experiences/other.md"]
    )
    provider._eval_memory = AsyncMock(return_value={})

    messages = await provider.prefetch()

    provider.search_files.assert_not_awaited()
    provider._eval_memory.assert_awaited_once()
    assert uri in provider._eval_memory.await_args.args[0]
    assert len(messages) == 3


@pytest.mark.asyncio
async def test_patch_merge_context_provider_caps_extra_candidate_reads_at_ten():
    schema = MemoryTypeSchema(
        memory_type="experiences",
        description="Experiences",
        directory="viking://user/{{ user_space }}/memories/experiences",
        filename_template="{{ experience_name }}.md",
        fields=[],
    )
    required_uris = [f"viking://user/u/memories/experiences/required_{idx}.md" for idx in range(12)]
    provider = PatchMergeContextProvider(
        memory_type="experiences",
        required_file_uris=required_uris,
        patches=[
            PatchMergePatch(
                before_file=None,
                after_file=_memory_file(
                    name="books",
                    uri="viking://user/u/memories/experiences/books.md",
                    content="用户喜欢阅读科幻书籍，尤其是太空歌剧。",
                ),
            )
        ],
    )
    provider._registry = SimpleNamespace(get=lambda name: schema if name == "experiences" else None)
    provider._ctx = SimpleNamespace(user=SimpleNamespace(user_id="u"))
    provider.search_files = AsyncMock(
        return_value=[
            *required_uris,
            *[f"viking://user/u/memories/experiences/candidate_{idx}.md" for idx in range(20)],
        ]
    )
    provider._eval_memory = AsyncMock(return_value={})

    await provider.prefetch()

    _, search_kwargs = provider.search_files.await_args
    assert search_kwargs["limit"] == 20
    assert provider._eval_memory.await_count == 2
    code = "\n".join(call.args[0] for call in provider._eval_memory.await_args_list)
    assert required_uris[-1] in code
    assert "viking://user/u/memories/experiences/candidate_9.md" in code
    assert "viking://user/u/memories/experiences/candidate_10.md" not in code


@pytest.mark.asyncio
async def test_patch_merge_context_provider_renders_compact_patch_metadata():
    provider = PatchMergeContextProvider(
        memory_type="experiences",
        required_file_uris=[],
        patches=[
            PatchMergePatch(
                before_file=None,
                after_file=_memory_file(
                    name="new_booking",
                    uri=None,
                    content="created line",
                ),
                metadata={
                    "base_version": 3,
                    "rationale": "useful reason",
                    "confidence": 0.9,
                    "links": [{"to_uri": "viking://user/u/memories/trajectories/t.md"}],
                    "memory_fields": {"content": "created line"},
                    "uris": ["viking://user/u/memories/experiences/new_booking.md"],
                    "gradient_metadata": {
                        "trajectory_outcome": "success",
                        "rubric_passed": True,
                        "training_category": "tau2:airline:train:1",
                        "memory_fields": {"content": "duplicated"},
                    },
                },
            )
        ],
    )

    messages = await provider.prefetch()
    content = messages[0]["content"]

    assert 'meta: {"confidence": 0.9}' in content
    assert "base_version" not in content
    assert "rationale" not in content
    assert "trajectory_outcome" not in content
    assert "gradient_metadata" not in content
    assert "links" not in content
    assert "memory_fields" not in content
    assert "duplicated" not in content


@pytest.mark.asyncio
async def test_patch_merge_context_provider_hides_last_update_trace_id_from_patch_diff():
    provider = PatchMergeContextProvider(
        memory_type="experiences",
        required_file_uris=[],
        patches=[
            PatchMergePatch(
                before_file=MemoryFile(
                    uri="viking://user/u/memories/experiences/booking.md",
                    content="same content",
                    memory_type="experiences",
                    extra_fields={
                        "memory_type": "experiences",
                        "experience_name": "booking",
                        "last_update_trace_id": "trace_old",
                    },
                ),
                after_file=MemoryFile(
                    uri="viking://user/u/memories/experiences/booking.md",
                    content="same content",
                    memory_type="experiences",
                    extra_fields={
                        "memory_type": "experiences",
                        "experience_name": "booking",
                        "last_update_trace_id": "trace_new",
                    },
                ),
            )
        ],
    )

    messages = await provider.prefetch()
    content = messages[0]["content"]

    assert "last_update_trace_id" not in content
    assert "trace_old" not in content
    assert "trace_new" not in content
    assert "(no changes)" in content


@pytest.mark.asyncio
async def test_patch_merge_context_provider_renders_create_patch_from_dev_null():
    provider = PatchMergeContextProvider(
        memory_type="experiences",
        required_file_uris=[],
        patches=[
            PatchMergePatch(
                before_file=None,
                after_file=_memory_file(name="new_booking", uri=None, content="created line"),
            )
        ],
    )

    messages = await provider.prefetch()

    assert len(messages) == 1
    assert "Patch 1" in messages[0]["content"]
    # Patch headers should not include target_name/target_uri/memory_type
    assert "target_name:" not in messages[0]["content"]
    assert "target_uri:" not in messages[0]["content"]
    # Routing metadata is not a schema business-field diff.
    assert "  memory_type:" not in messages[0]["content"]
    assert "  content:" in messages[0]["content"]
    assert "--- content.before" not in messages[0]["content"]
    assert "+++ content.after" not in messages[0]["content"]
    assert "    +created line" in messages[0]["content"]


def test_patch_merge_context_provider_get_memory_schema_single_type(monkeypatch):
    schema = MemoryTypeSchema(
        memory_type="experiences",
        description="Experiences",
        directory="viking://user/{{ user_space }}/memories/experiences",
        filename_template="{{ experience_name }}.md",
        fields=[],
    )
    provider = PatchMergeContextProvider(
        memory_type="experiences",
        required_file_uris=[],
        patches=[],
    )
    provider._registry = SimpleNamespace(get=lambda name: schema if name == "experiences" else None)

    assert provider.get_memory_schemas(ctx=None) == [schema]


def test_patch_merge_context_provider_get_memory_schema_raises_for_missing_type():
    provider = PatchMergeContextProvider(
        memory_type="missing",
        required_file_uris=[],
        patches=[],
    )
    provider._registry = SimpleNamespace(get=lambda name: None)

    with pytest.raises(ValueError, match="Memory schema not found or disabled: missing"):
        provider.get_memory_schemas(ctx=None)


def test_patch_merge_context_provider_uses_registry_override_for_session_skills():
    provider = PatchMergeContextProvider(
        memory_type="session_skills",
        memory_registry=load_skill_extract_registry(),
        required_file_uris=[],
        patches=[],
    )

    schemas = provider.get_memory_schemas(ctx=None)

    assert len(schemas) == 1
    assert schemas[0].memory_type == "session_skills"
    assert schemas[0].enabled is True


def test_patch_merge_context_provider_instruction_mentions_path_field_normalization():
    provider = PatchMergeContextProvider(
        memory_type="entities",
        required_file_uris=[],
        patches=[],
    )

    instruction = provider.instruction()

    assert "additions, edits, and deletion proposals together" in instruction
    assert "same schema-defined identity" in instruction
    assert "same entity/event" in instruction
    assert "same owner and behavioral choice dimension" in instruction
    assert "Similarity or shared topics are insufficient" in instruction
    assert "preserve each still-valid distinct fact" in instruction
    assert "surviving schema-valid destination" in instruction
    assert "If identity, validity, or destination is uncertain, keep the source" in instruction
    assert "Split only into distinct schema-valid identities" in instruction
    assert "never invent identities" in instruction
    assert "Keep one identity in one object" in instruction
    assert "directory/filename fields" in instruction
    assert "schema identifiers" in instruction
    assert "book not books" in instruction
    assert "Chinese" in instruction
    assert "书籍 not 书/图书" in instruction
    assert "verified duplicate existing file" in instruction
    assert "put it in delete_ids" in instruction


def test_patch_merge_context_provider_detects_language_from_patch_content(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    monkeypatch.setattr(
        "openviking.session.memory.utils.language.get_openviking_config",
        lambda: SimpleNamespace(output_language_override=""),
    )
    provider = PatchMergeContextProvider(
        memory_type="preferences",
        required_file_uris=[],
        patches=[
            PatchMergePatch(
                before_file=None,
                after_file=MemoryFile(
                    uri=None,
                    content="User prefers concise implementation and minimal fallback logic.",
                    memory_type="preferences",
                    extra_fields={
                        "memory_type": "preferences",
                        "user": "alice",
                        "topic": "code_style",
                    },
                ),
            )
        ],
    )

    assert provider.get_output_language() == "en"
    assert "All memory content must be written in en." in provider.instruction()


def test_patch_merge_context_provider_empty_patches_fallback_to_english(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    monkeypatch.setattr(
        "openviking.session.memory.utils.language.get_openviking_config",
        lambda: SimpleNamespace(output_language_override=""),
    )
    provider = PatchMergeContextProvider(
        memory_type="preferences",
        required_file_uris=[],
        patches=[],
    )

    assert provider.get_output_language() == "en"


def test_patch_merge_context_provider_ignores_before_file_language(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    monkeypatch.setattr(
        "openviking.session.memory.utils.language.get_openviking_config",
        lambda: SimpleNamespace(output_language_override=""),
    )
    provider = PatchMergeContextProvider(
        memory_type="preferences",
        required_file_uris=[],
        patches=[
            PatchMergePatch(
                before_file=MemoryFile(
                    uri="viking://user/u/memories/preferences/old.md",
                    content="用户偏好简洁实现。",
                    memory_type="preferences",
                    extra_fields={"memory_type": "preferences", "topic": "代码风格"},
                ),
                after_file=MemoryFile(
                    uri="viking://user/u/memories/preferences/old.md",
                    content="User prefers concise implementation.",
                    memory_type="preferences",
                    extra_fields={"memory_type": "preferences", "topic": "code_style"},
                ),
            )
        ],
    )

    assert provider.get_output_language() == "en"


@pytest.mark.asyncio
async def test_fixed_case_merge_never_searches_or_renames_other_experiences():
    from unittest.mock import AsyncMock

    uri = "viking://user/u/memories/experiences/case_a.md"
    provider = PatchMergeContextProvider(
        memory_type="experiences",
        fixed_target=True,
        required_file_uris=[uri],
        patches=[
            PatchMergePatch(
                before_file=None,
                after_file=MemoryFile(
                    uri=uri,
                    content="dag",
                    memory_type="experiences",
                    extra_fields={"experience_name": "case_a"},
                ),
            )
        ],
    )
    provider.search_files = AsyncMock(side_effect=AssertionError("fixed target must not search"))
    assert await provider._resolve_prefetch_file_uris() == [uri]
    provider.search_files.assert_not_called()
    assert "Never rename" in provider.instruction()
    assert uri in provider.instruction()


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_type", ["cases", "entities", "preferences"])
@pytest.mark.parametrize("relation_field", ["links", "backlinks"])
@pytest.mark.parametrize("placement", ["attribute", "extra_fields", "both"])
async def test_merge_prompt_omits_relation_indexes_without_mutating_memories(
    memory_type, relation_field, placement
):
    from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils

    uri = f"viking://user/u/memories/{memory_type}/shared.md"
    hidden_uri = "viking://user/u/memories/trajectories/relation_only.md"
    relation = {
        "from_uri": uri if relation_field == "links" else hidden_uri,
        "to_uri": hidden_uri if relation_field == "links" else uri,
        "link_type": "related_to",
        "weight": 1.0,
    }
    before = _memory_file(
        name="shared",
        uri=uri,
        memory_type=memory_type,
        content="Old fact.\nKeep [policy](./policy.md).",
    )
    if placement in {"attribute", "both"}:
        setattr(before, relation_field, [relation])
    if placement in {"extra_fields", "both"}:
        before.extra_fields[relation_field] = [relation]
    after = _memory_file(
        name="shared",
        uri=uri,
        memory_type=memory_type,
        content="New fact.\nKeep [policy](./policy.md).",
    )
    if memory_type == "cases":
        before.extra_fields["domain"] = before.content
        after.extra_fields["domain"] = after.content
    snapshots = copy.deepcopy((before.model_dump(), after.model_dump()))
    stored_before = MemoryFileUtils.write(before)
    provider = PatchMergeContextProvider(
        memory_type=memory_type,
        patches=[PatchMergePatch(before_file=before, after_file=after)],
    )
    messages = await provider.prefetch()
    text = messages[0]["content"]
    assert hidden_uri not in text
    assert f"  {relation_field}:" not in text
    assert "Old fact." in text and "New fact." in text
    assert "Keep [policy](./policy.md)." in text
    assert (before.model_dump(), after.model_dump()) == snapshots
    assert MemoryFileUtils.write(before) == stored_before


@pytest.mark.asyncio
async def test_relation_only_changes_do_not_generate_business_patch_or_change_language_input():
    from openviking.session.memory.patch_merge_context_provider import _patch_language_text

    uri = "viking://user/u/memories/entities/shared.md"
    before = _memory_file(
        name="shared", uri=uri, content="Unchanged business facts.", memory_type="entities"
    )
    after = before.model_copy(deep=True)
    relation = {
        "from_uri": uri,
        "to_uri": "viking://user/u/memories/entities/relation_only.md",
        "link_type": "related_to",
        "weight": 1.0,
        "description": "关系索引，不是正文",
    }
    after.links = [relation]
    after.backlinks = [relation]
    after.extra_fields.update(links=[relation], backlinks=[relation])
    patch = PatchMergePatch(before_file=before, after_file=after)
    provider = PatchMergeContextProvider(memory_type="entities", patches=[patch])
    text = (await provider.prefetch())[0]["content"]
    assert "(no changes)" in text
    assert "relation_only" not in _patch_language_text(
        [patch], get_default_registry().get("entities")
    )
    assert after.links == [relation] and after.backlinks == [relation]


@pytest.mark.asyncio
async def test_update_from_filtered_patch_keeps_stored_outgoing_and_incoming_relations():
    from openviking.server.identity import RequestContext, Role
    from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
    from openviking.session.memory.memory_type_registry import get_default_registry
    from openviking.session.memory.memory_updater import MemoryUpdater
    from openviking.session.memory.patch_merge_context_provider import _memory_file_fields
    from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
    from openviking_cli.session.user_id import UserIdentifier

    uri = "viking://user/u/memories/entities/shared.md"
    fields = {"category": "concept", "name": "shared"}
    before = MemoryFile(
        uri=uri,
        memory_type="entities",
        content="Old fact.",
        extra_fields=fields,
        links=[
            {
                "from_uri": uri,
                "to_uri": "viking://user/u/memories/experiences/shared_domain.md",
                "link_type": "related_to",
                "weight": 1.0,
            }
        ],
        backlinks=[
            {
                "from_uri": "viking://user/u/memories/trajectories/source.md",
                "to_uri": uri,
                "link_type": "related_to",
                "weight": 1.0,
            }
        ],
    )
    files = {uri: MemoryFileUtils.write(before)}

    async def write_file(uri, content, **kwargs):
        files[uri] = content

    fs = SimpleNamespace(
        read_file=AsyncMock(side_effect=lambda uri, **kwargs: files[uri]),
        write_file=AsyncMock(side_effect=write_file),
    )
    after = before.model_copy(deep=True)
    after.content = "New fact."
    model_fields = _memory_file_fields(after, get_default_registry().get("entities"))
    assert "links" not in model_fields and "backlinks" not in model_fields
    updater = MemoryUpdater(registry=get_default_registry())
    updater._get_viking_fs = lambda: fs
    updater._sync_resource_refs_for_result = AsyncMock()
    updater._vectorize_memories = AsyncMock()
    updater.generate_overview = AsyncMock()
    result = await updater.apply_operations(
        ResolvedOperations(
            upsert_operations=[
                ResolvedOperation(
                    memory_type="entities",
                    uris=[uri],
                    old_memory_file_content=before,
                    memory_fields=model_fields,
                )
            ],
            delete_file_contents=[],
            errors=[],
        ),
        RequestContext(user=UserIdentifier("default", "u"), role=Role.ROOT),
    )
    assert result.errors == [] and result.edited_uris == [uri]
    stored = MemoryFileUtils.read(files[uri], uri=uri)
    assert stored.content == "New fact."
    assert stored.links == before.links and stored.backlinks == before.backlinks


@pytest.fixture
def structured_template():
    schema = MemoryTypeSchema(
        memory_type="summaries",
        directory="viking://user/{{ user_space }}/memories/summaries",
        filename_template="{{ name }}.md",
        content_template="TEMPLATE_WRAPPER\n# {{ name }}\n{{ evidence }}\nCount: {{ sample_count }}",
        fields=[
            MemoryField(name="name", field_type=FieldType.STRING, merge_op=MergeOp.IMMUTABLE),
            MemoryField(name="evidence", field_type=FieldType.STRING, merge_op=MergeOp.PATCH),
            MemoryField(name="sample_count", field_type=FieldType.INT64, merge_op=MergeOp.SUM),
        ],
    )
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.register(schema)
    return schema, registry


def _rendered_record(schema, fields):
    from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils

    uri = "viking://user/u/memories/summaries/domain.md"
    return MemoryFileUtils.read(
        MemoryFileUtils.write(
            MemoryFile(uri=uri, memory_type=schema.memory_type, content="", extra_fields=fields),
            content_template=schema.content_template,
        ),
        uri=uri,
    )


@pytest.mark.asyncio
async def test_template_output_is_not_a_patch_field_or_language_input(structured_template):
    from openviking.session.memory.patch_merge_context_provider import (
        _memory_file_fields,
        _patch_language_text,
    )

    schema, registry = structured_template
    before = _rendered_record(schema, {"name": "domain", "evidence": "OLD_FACT", "sample_count": 1})
    after = _rendered_record(schema, {"name": "domain", "evidence": "NEW_FACT", "sample_count": 0})
    before.extra_fields.update(version=9, server_only="NOT_A_BUSINESS_FIELD")
    after.extra_fields.update(version=10, server_only="CHANGED_SYSTEM_VALUE")
    snapshots = copy.deepcopy((before.model_dump(), after.model_dump()))
    patch = PatchMergePatch(before_file=before, after_file=after)
    provider = PatchMergeContextProvider(
        memory_type=schema.memory_type, memory_registry=registry, patches=[patch]
    )
    text = (await provider.prefetch())[0]["content"]
    assert text.count("NEW_FACT") == 1
    assert "  evidence:" in text and "  sample_count:" in text
    assert "  content:" not in text and "TEMPLATE_WRAPPER" not in text
    assert "server_only" not in text and "version" not in text
    assert _memory_file_fields(after, schema) == {
        "name": "domain",
        "evidence": "NEW_FACT",
        "sample_count": 0,
    }
    assert "TEMPLATE_WRAPPER" not in _patch_language_text([patch], schema)
    assert (before.model_dump(), after.model_dump()) == snapshots


@pytest.mark.asyncio
async def test_template_only_change_is_not_a_business_change(structured_template):
    schema, registry = structured_template
    before = _rendered_record(
        schema, {"name": "domain", "evidence": "SAME_FACT", "sample_count": 1}
    )
    after = before.model_copy(deep=True)
    after.content = "A different presentation of SAME_FACT"
    provider = PatchMergeContextProvider(
        memory_type=schema.memory_type,
        memory_registry=registry,
        patches=[PatchMergePatch(before_file=before, after_file=after)],
    )
    assert "(no changes)" in (await provider.prefetch())[0]["content"]


@pytest.mark.asyncio
async def test_prefetch_omits_rendered_copy_but_keeps_read_cache_and_errors(structured_template):
    schema, registry = structured_template
    memory = _rendered_record(
        schema, {"name": "domain", "evidence": "STORED_FACT", "sample_count": 1}
    )
    result = {
        "prefetched_memory_1": {**memory.to_metadata(), "page_id": 1},
        "prefetched_memory_2": {
            "memory_type": schema.memory_type,
            "error": "not found",
            "content": "error detail",
        },
    }
    original = copy.deepcopy(result)
    provider = PatchMergeContextProvider(
        memory_type=schema.memory_type,
        memory_registry=registry,
        required_file_uris=[memory.uri],
        patches=[],
    )
    provider.read_file_contents[memory.uri] = memory
    provider._eval_memory = AsyncMock(return_value=result)
    messages = await provider.prefetch()
    displayed = json.loads(messages[1]["content"])
    assert "content" not in displayed["prefetched_memory_1"]
    assert displayed["prefetched_memory_1"]["evidence"] == "STORED_FACT"
    assert displayed["prefetched_memory_1"]["page_id"] == 1
    assert displayed["prefetched_memory_2"] == original["prefetched_memory_2"]
    assert "TEMPLATE_WRAPPER" not in json.dumps(messages)
    assert result == original
    assert provider.read_file_contents[memory.uri] is memory
    assert memory.content.startswith("TEMPLATE_WRAPPER")


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_type", ["entities", "preferences", "experiences", "profile"])
async def test_declared_content_remains_business_data_in_patch_and_prefetch(memory_type):
    from openviking.session.memory.patch_merge_context_provider import _memory_file_fields

    schema = get_default_registry().get(memory_type)
    before = _memory_file(
        name="original",
        uri=f"viking://user/u/memories/{memory_type}/original.md",
        memory_type=memory_type,
        content="Old fact. [reference](./original.md)",
    )
    after = before.model_copy(deep=True)
    after.content = "New fact. [reference](./original.md)"
    provider = PatchMergeContextProvider(
        memory_type=memory_type,
        required_file_uris=[before.uri],
        patches=[PatchMergePatch(before_file=before, after_file=after)],
    )
    provider._eval_memory = AsyncMock(
        return_value={"prefetched_memory_1": {**before.to_metadata(), "page_id": 1}}
    )
    messages = await provider.prefetch()
    assert json.loads(messages[1]["content"])["prefetched_memory_1"]["content"] == before.content
    assert "  content:" in messages[-1]["content"]
    assert before.content in messages[-1]["content"] and after.content in messages[-1]["content"]
    assert _memory_file_fields(before, schema)["content"] == before.content


@pytest.mark.asyncio
async def test_merged_fields_are_applied_then_rendered_by_existing_writer(structured_template):
    from openviking.server.identity import RequestContext, Role
    from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
    from openviking.session.memory.memory_updater import MemoryUpdater
    from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
    from openviking_cli.session.user_id import UserIdentifier

    schema, registry = structured_template
    before = _rendered_record(
        schema, {"name": "domain", "evidence": "Old fact.", "sample_count": 2}
    )
    files = {before.uri: MemoryFileUtils.write(before)}

    async def write_file(uri, content, **kwargs):
        files[uri] = content

    fs = SimpleNamespace(
        read_file=AsyncMock(side_effect=lambda uri, **kwargs: files[uri]),
        write_file=AsyncMock(side_effect=write_file),
    )
    updater = MemoryUpdater(registry=registry)
    updater._get_viking_fs = lambda: fs
    updater._sync_resource_refs_for_result = AsyncMock()
    updater._vectorize_memories = AsyncMock()
    updater.generate_overview = AsyncMock()
    result = await updater.apply_operations(
        ResolvedOperations(
            upsert_operations=[
                ResolvedOperation(
                    memory_type=schema.memory_type,
                    uris=[before.uri],
                    old_memory_file_content=before,
                    memory_fields={
                        "name": "domain",
                        "evidence": "Old fact. New fact.",
                        "sample_count": 1,
                    },
                )
            ],
            delete_file_contents=[],
            errors=[],
        ),
        RequestContext(user=UserIdentifier("default", "u"), role=Role.ROOT),
    )
    assert result.errors == [] and result.edited_uris == [before.uri]
    stored = MemoryFileUtils.read(files[before.uri], uri=before.uri)
    assert stored.extra_fields["sample_count"] == 3
    assert stored.extra_fields["evidence"] == "Old fact. New fact."
    assert stored.content == "TEMPLATE_WRAPPER\n# domain\nOld fact. New fact.\nCount: 3"
    assert "content" not in stored.extra_fields


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_type", ["cases", "entities"])
async def test_delete_patch_is_explicit_not_an_empty_content_edit(memory_type):
    """Template-rendered content must not hide deletions as '(no changes)'."""
    uri = f"viking://user/u/memories/{memory_type}/domain.md"
    memory = MemoryFile(
        uri=uri,
        memory_type=memory_type,
        content="Rendered document",
        extra_fields={"case_name": "domain", "domain": "Domain service", "name": "domain"},
    )
    patch = PatchMergePatch(before_file=memory, after_file=None)
    provider = PatchMergeContextProvider(memory_type=memory_type, patches=[patch])
    text = (await provider.prefetch())[0]["content"]
    assert f"delete: {uri}" in text
    assert "(no changes)" not in text
    assert patch.target_uri == uri
    assert patch.target_name == "domain"
    assert patch.memory_type == memory_type


@pytest.mark.asyncio
async def test_empty_business_content_is_an_edit_not_a_file_delete():
    uri = "viking://user/u/memories/entities/domain.md"
    before = _memory_file(name="domain", uri=uri, memory_type="entities", content="Old facts")
    after = before.model_copy(update={"content": ""})
    provider = PatchMergeContextProvider(
        memory_type="entities", patches=[PatchMergePatch(before_file=before, after_file=after)]
    )
    text = (await provider.prefetch())[0]["content"]
    assert "delete:" not in text
    assert "-Old facts" in text
