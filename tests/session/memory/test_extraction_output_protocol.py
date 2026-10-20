# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for JSON and restricted-Python memory extraction output protocols."""

import ast
import copy
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openviking.models.vlm.base import ToolCall, VLMResponse
from openviking.session.memory.dataclass import (
    MemoryField,
    MemoryFile,
    MemoryTypeSchema,
    ResolvedOperations,
)
from openviking.session.memory.experience.trajectory_context import TrajectoryTarget
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.extraction_output_protocol import (
    ExtractionOutputContext,
    create_extraction_output_protocol,
)
from openviking.session.memory.extraction_output_protocol.python_protocol import (
    _NESTED_STRING_GUIDANCE,
    _identifier_alias,
)
from openviking.session.memory.memory_isolation_handler import RoleScope
from openviking.session.memory.memory_type_registry import (
    MemoryTypeRegistry,
    resolve_memory_templates_dir,
)
from openviking.session.memory.merge_op import FieldType, MergeOp, PatchOp
from openviking.session.memory.merge_policy import MEMORY_MERGE_POLICY
from openviking.session.memory.page_id_map import PageIdMap
from openviking.session.memory.schema_model_generator import SchemaModelGenerator
from openviking.session.memory.tools import add_tool_call_pair_to_messages
from tests.session.train.test_fakes import render_trajectory_template


def _assert_native_pair(messages, *, call_id, tool_name, params, result, python_result=False):
    assert messages[0]["role"] == "assistant"
    call = messages[0]["tool_calls"][0]
    assert call == {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(params, ensure_ascii=False, default=str),
        },
    }
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == call_id
    if python_result:
        assert _literal_tool_result(messages[1]["content"], tool_name) == result
    else:
        assert json.loads(messages[1]["content"]) == result


def _literal_tool_result(code, tool_name="eval"):
    def decode(node):
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Attribute) and node.func.attr == "error"
            assert isinstance(node.func.value, ast.Name) and node.func.value.id == "sdk"
            assert len(node.args) == 1 and not node.keywords
            return {"error": ast.literal_eval(node.args[0])}
        if isinstance(node, ast.List):
            return [decode(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {
                ast.literal_eval(key): decode(value)
                for key, value in zip(node.keys, node.values, strict=True)
            }
        return ast.literal_eval(node)

    statements = ast.parse(code).body
    if len(statements) == 1 and isinstance(statements[0], ast.Expr):
        return decode(statements[0].value)
    values = {node.targets[0].id: decode(node.value) for node in statements}
    return values if tool_name == "eval" else values[tool_name]


def _preference_schema(*, operation_mode: str = "upsert") -> MemoryTypeSchema:
    return MemoryTypeSchema(
        memory_type="preferences",
        description="User preferences",
        directory="viking://user/{{ user_space }}/memories/preferences",
        filename_template="{{ topic }}.md",
        operation_mode=operation_mode,
        fields=[
            MemoryField(
                name="topic",
                field_type=FieldType.STRING,
                merge_op=MergeOp.IMMUTABLE,
            ),
            MemoryField(
                name="content",
                field_type=FieldType.STRING,
                merge_op=MergeOp.PATCH,
            ),
            MemoryField(
                name="score",
                field_type=FieldType.INT64,
                merge_op=MergeOp.SUM,
            ),
        ],
    )


def _project_schema() -> MemoryTypeSchema:
    return MemoryTypeSchema(
        memory_type="projects",
        description="Projects",
        directory="viking://user/{{ user_space }}/memories/projects",
        filename_template="{{ name }}.md",
        fields=[
            MemoryField(
                name="name",
                field_type=FieldType.STRING,
                merge_op=MergeOp.IMMUTABLE,
            ),
            MemoryField(
                name="content",
                field_type=FieldType.STRING,
                merge_op=MergeOp.REPLACE,
            ),
        ],
    )


def _profile_schema() -> MemoryTypeSchema:
    return MemoryTypeSchema(
        memory_type="profile",
        description="User profile",
        directory="viking://user/{{ user_space }}/memories",
        filename_template="profile.md",
        fields=[
            MemoryField(
                name="content",
                field_type=FieldType.STRING,
                merge_op=MergeOp.PATCH,
            )
        ],
    )


def _context(
    schemas: list[MemoryTypeSchema],
    *,
    files: list[MemoryFile] | None = None,
    link_enabled: bool = False,
    role_scope: RoleScope | None = None,
    available_tools: tuple[str, ...] = ("read",),
    template_context: dict[str, str] | None = None,
) -> ExtractionOutputContext:
    config = SimpleNamespace(memory=SimpleNamespace(link_enabled=link_enabled))
    with patch("openviking_cli.utils.config.get_openviking_config", return_value=config):
        operations_model = SchemaModelGenerator(
            schemas, template_context=template_context
        ).create_structured_operations_model(role_scope)
    page_id_map = PageIdMap()
    read_file_contents = {}
    for memory_file in files or []:
        read_file_contents[memory_file.uri] = memory_file
        page_id_map.get_page_id(memory_file.uri)
    return ExtractionOutputContext(
        operations_model=operations_model,
        schemas=tuple(schemas),
        page_id_map=page_id_map,
        read_file_contents=read_file_contents,
        link_enabled=link_enabled,
        role_scope=role_scope,
        available_tools=available_tools,
        template_context=dict(template_context or {}),
    )


def _existing_preference(uri: str, topic: str, content: str, score: int = 0) -> MemoryFile:
    return MemoryFile(
        uri=uri,
        memory_type="preferences",
        content=content,
        extra_fields={
            "topic": topic,
            "score": score,
            "version": 7,
            "_uri": uri,
        },
    )


def _bind(protocol, context: ExtractionOutputContext) -> str:
    protocol.render_contract(context)
    counts, result, requests = {}, {}, []
    for uri, memory in context.read_file_contents.items():
        kind = memory.memory_type
        counts[kind] = counts.get(kind, 0) + 1
        name = f"{_identifier_alias(kind)}_{counts[kind]}"
        requests.append(f"{name} = sdk.read(uri={uri!r})")
        result[name] = {
            **memory.extra_fields,
            "memory_type": kind,
            "content": memory.plain_content(),
            "page_id": context.page_id_map.get_page_id(uri),
        }
    return protocol.render_tool_result_messages(
        context,
        call_id="bind",
        tool_name="eval",
        params={"code": "\n".join(requests)},
        result=result,
        source="test read",
    )[1]["content"]


def test_json_protocol_preserves_stable_parser_and_empty_contract():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("json")

    operations, error = protocol.parse(
        '{"preferences": [], "delete_ids": [], "ignored": true}', context
    )

    assert error is None
    assert operations.model_dump() == {"preferences": [], "delete_ids": []}
    assert '"preferences": []' in protocol.render_final_instruction(context)
    assert '"delete_ids": []' in protocol.render_final_instruction(context)


@pytest.mark.parametrize(
    ("raw_content", "expected_content"),
    [
        ("1\t- Use Vim\n2\t- Avoid tabs\n", "- Use Vim\n- Avoid tabs\n"),
        ("1\t- Use Vim\n- Avoid tabs", "- Use Vim\n- Avoid tabs"),
        ("2026\tbudget", "2026\tbudget"),
    ],
)
def test_json_protocol_strips_read_line_numbers_from_complete_content(
    raw_content: str, expected_content: str
):
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("json")

    operations, error = protocol.parse(
        json.dumps(
            {
                "preferences": [
                    {
                        "page_id": 1,
                        "topic": "editor",
                        "content": raw_content,
                    }
                ],
                "delete_ids": [],
            }
        ),
        context,
    )

    assert error is None
    assert operations.preferences[0].content == expected_content


def test_python_contract_and_bindings_expose_only_selected_schema_fields():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim", 2)],
    )
    protocol = create_extraction_output_protocol("python")

    contract = protocol.render_contract(context)
    bindings = _bind(protocol, context)

    assert "sdk.create_preferences" in contract
    assert "sdk.create_projects" not in contract
    assert "preferences_1 = sdk.existing" in bindings
    assert "topic='editor'" in bindings
    assert "content='Use Vim'" in bindings
    assert "version" not in bindings.split("\n# Metadata", 1)[0]
    assert uri not in bindings
    assert _bind(protocol, context) == bindings


@pytest.mark.parametrize("include_link_example", [True, False])
def test_python_contract_includes_link_rules_when_enabled(include_link_example):
    context = _context([_preference_schema()], link_enabled=True)
    assert context.include_link_example is True
    context.include_link_example = include_link_example
    protocol = create_extraction_output_protocol("python")

    contract = protocol.render_contract(context)

    assert "## Link Rules" in contract
    assert "obj.link(target" in contract
    assert "match_text" in contract
    assert ("obj_a.link(" in contract) is include_link_example
    assert "assign the create/set call to a variable first" in contract


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize(
    ("memory_type", "field_name"),
    [("preferences", "topic"), ("entities", "category"), ("events", "event_name")],
)
def test_python_contract_renders_builtin_field_descriptions_like_json(
    language, memory_type, field_name
):
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / f"{memory_type}.yaml"))
    schema = registry.get(memory_type)
    original = next(field.description for field in schema.fields if field.name == field_name)
    assert "{{ language }}" in original
    context = _context([schema], template_context={"language": language})

    python_contract = create_extraction_output_protocol("python").render_contract(context)
    json_contract = create_extraction_output_protocol("json").render_contract(context)
    json_schema = json.loads(json_contract.split("```json\n", 1)[1].split("```", 1)[0])
    model_ref = json_schema["properties"][memory_type]["items"]["$ref"].rsplit("/", 1)[1]
    rendered = json_schema["$defs"][model_ref]["properties"][field_name]["description"]

    assert " ".join(rendered.split()) in python_contract
    assert language in rendered
    assert "{{ language }}" not in python_contract
    assert "{% if language" not in python_contract
    assert ("Use lowercase with underscores" in rendered) == (language == "en")
    assert (
        next(field.description for field in schema.fields if field.name == field_name) == original
    )


def test_python_field_description_keeps_dsl_patch_instructions():
    schema = _preference_schema()
    schema.fields[1].description = "Write content in {{ language.upper() }}."
    context = _context([schema], template_context={"language": "en"})

    contract = create_extraction_output_protocol("python").render_contract(context)

    assert (
        "content [editable string: obj.field.edit/drop/overwrite]: Write content in EN." in contract
    )
    assert "obj.field.edit(search=..., replace=...)" in contract
    assert "PATCH operation for" not in contract
    assert "Use a DELETE block" not in contract


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh-CN"])
async def test_default_python_extract_loop_passes_language_to_field_descriptions(language):
    schema = _preference_schema()
    schema.description = "Preferences in {{ language }}."
    schema.fields[1].description = "Write content in {{ language }}."
    provider = MagicMock()
    provider.format_prefetch_messages.side_effect = lambda messages: messages
    provider.get_memory_schemas.return_value = [schema]
    provider.get_output_language.return_value = language
    provider.get_tools.return_value = []
    provider.get_extract_context.return_value = SimpleNamespace(page_id_map=PageIdMap())
    provider.read_file_contents = {}
    provider.instruction.return_value = "Extract memory operations."
    provider.prefetch = AsyncMock(return_value=[])
    vlm = SimpleNamespace(
        model="test-model", get_completion_async=AsyncMock(return_value="sdk.commit()")
    )
    loop = ExtractLoop(vlm=vlm, viking_fs=MagicMock(), context_provider=provider, max_iterations=1)
    empty = ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[])
    loop.resolve_operations = AsyncMock(return_value=(empty, []))
    loop._check_unread_existing_files = AsyncMock(return_value={})
    # Exercise the default protocol selection rather than explicitly choosing Python.
    config = SimpleNamespace(memory=SimpleNamespace(link_enabled=False))
    with (
        patch("openviking.session.memory.extract_loop.get_openviking_config", return_value=config),
        patch("openviking_cli.utils.config.get_openviking_config", return_value=config),
    ):
        operations, _ = await loop.run()

    assert operations is empty
    prompt = vlm.get_completion_async.call_args.kwargs["messages"][0]["content"]
    assert "restricted Python memory SDK" in prompt
    assert f"Preferences in {language}." in prompt
    assert f"Write content in {language}." in prompt
    assert "{{ language }}" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
@pytest.mark.parametrize("provider_kind", ["session", "patch_merge"])
async def test_shared_write_decision_reaches_model_once_without_extra_calls(
    output_format, provider_kind
):
    from openviking.session.memory.patch_merge_context_provider import PatchMergeContextProvider
    from openviking.session.memory.session_extract_context_provider import (
        SessionExtractContextProvider,
    )

    real_provider = (
        SessionExtractContextProvider(messages=[])
        if provider_kind == "session"
        else PatchMergeContextProvider(memory_type="preferences", patches=[])
    )
    real_provider.get_tools = lambda: []
    provider = MagicMock()
    provider.instruction.side_effect = real_provider.instruction
    provider.format_prefetch_messages.side_effect = lambda messages: messages
    provider.get_memory_schemas.return_value = [_preference_schema()]
    provider.get_output_language.return_value = "en"
    provider.get_tools.return_value = []
    provider.get_extract_context.return_value = SimpleNamespace(page_id_map=PageIdMap())
    provider.read_file_contents = {}
    provider.prefetch = AsyncMock(return_value=[])
    response = "sdk.commit()" if output_format == "python" else '{"preferences": []}'
    completion = AsyncMock(return_value=response)
    loop = ExtractLoop(
        vlm=SimpleNamespace(model="test", get_completion_async=completion),
        viking_fs=MagicMock(),
        context_provider=provider,
        max_iterations=1,
    )
    empty = ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[])
    loop.resolve_operations = AsyncMock(return_value=(empty, []))
    loop._check_unread_existing_files = AsyncMock(return_value={})
    config = SimpleNamespace(
        memory=SimpleNamespace(link_enabled=False, extraction_output_format=output_format)
    )
    with (
        patch("openviking.session.memory.extract_loop.get_openviking_config", return_value=config),
        patch("openviking_cli.utils.config.get_openviking_config", return_value=config),
    ):
        operations, _ = await loop.run()

    assert operations is empty
    completion.assert_awaited_once()
    prompt = completion.call_args.kwargs["messages"][0]["content"]
    assert prompt.count(MEMORY_MERGE_POLICY) == 1
    assert prompt.count("## Memory Write Decision") == 1
    assert prompt.count("## Memory Reorganization Safety") == 1
    assert len(MEMORY_MERGE_POLICY) < 1000  # Bound prompt size without tying it to a tokenizer.
    policy = " ".join(MEMORY_MERGE_POLICY.split())
    assert "Per target, write only for schema-admissible additions, corrections" in policy
    assert "explicit revocations, or schema-required cleanup; otherwise no-op" in policy
    assert "Mentions or action requests alone do not justify writes" in policy
    assert "Merge only the same schema-defined identity" in policy
    assert "same entity/event" in policy
    assert "same owner and behavioral choice dimension" in policy
    assert "Similarity or shared topics are insufficient" in policy
    assert "preserve each still-valid distinct fact exactly once" in policy
    assert "surviving schema-valid destination" in policy
    assert "If identity, validity, or destination is uncertain, keep the source" in policy
    assert "Split only into distinct schema-valid identities" in policy
    assert "never invent identities to meet length targets" in policy
    assert "Keep one identity in one object, using sections when needed" in policy
    assert "Compact duplicate wording only" in policy
    assert "never drop, weaken, or generalize valid facts to shorten content" in policy
    assert "When a memory is too large" not in policy
    assert all(term not in policy for term in ("ss_project", "project_id", "plot_id", "clip_id"))


@pytest.mark.asyncio
@pytest.mark.parametrize("call_count", [1, 2])
async def test_python_loop_preserves_native_eval_calls_and_aggregate_results(call_count):
    code = 'editor = sdk.read_preferences(topic="editor")\nformatting = sdk.read_preferences(topic="formatting")'
    result = {
        "editor": {"error": "File not found: editor.md"},
        "formatting": {"error": "File not found: formatting.md"},
    }
    calls = [
        ToolCall(id=f"eval-{i}", name="eval", arguments={"code": code}) for i in range(call_count)
    ]
    requests = []

    async def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return VLMResponse(tool_calls=calls) if len(requests) == 1 else "sdk.commit()"

    provider = MagicMock()
    provider.format_prefetch_messages.side_effect = lambda messages: messages
    provider.get_memory_schemas.return_value = [_preference_schema()]
    provider.get_output_language.return_value = "en"
    provider.get_tools.return_value = ["eval"]
    provider.get_extract_context.return_value = SimpleNamespace(page_id_map=PageIdMap())
    provider.read_file_contents = {}
    provider.instruction.return_value = "Extract memory operations."
    provider.prefetch = AsyncMock(return_value=[])
    provider.customize_tool_schema.side_effect = lambda name, schema: schema
    provider.execute_tool = AsyncMock(return_value=result)
    loop = ExtractLoop(
        vlm=SimpleNamespace(model="test", get_completion_async=complete),
        viking_fs=MagicMock(),
        context_provider=provider,
        max_iterations=3,
    )
    empty = ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[])
    loop.resolve_operations = AsyncMock(return_value=(empty, []))
    loop._check_unread_existing_files = AsyncMock(return_value={})
    config = SimpleNamespace(memory=SimpleNamespace(link_enabled=False))
    with (
        patch("openviking.session.memory.extract_loop.get_openviking_config", return_value=config),
        patch("openviking_cli.utils.config.get_openviking_config", return_value=config),
    ):
        operations, tools_used = await loop.run()

    assert operations is empty
    assert provider.execute_tool.await_count == call_count
    assert len(tools_used) == call_count
    assert len(requests) == 2
    assert [tool["function"]["name"] for tool in requests[1]["tools"]] == ["eval"]
    assert all("ONLY via: (none)." in request["messages"][-1]["content"] for request in requests)
    history_before = requests[0]["messages"][:-1]
    history_after = requests[1]["messages"][:-1]
    added = history_after[len(history_before) :]
    assert [m["role"] for m in added] == ["assistant"] + ["tool"] * call_count
    assert len(added[0]["tool_calls"]) == call_count
    for call, request_call, response in zip(calls, added[0]["tool_calls"], added[1:], strict=True):
        assert request_call["id"] == response["tool_call_id"] == call.id
        assert json.loads(request_call["function"]["arguments"]) == {"code": code}
        assert _literal_tool_result(response["content"]) == result
    assert "eval_result" not in json.dumps(added)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefetch", [False, True])
async def test_python_loop_edits_objects_bound_in_tool_code_without_user_bindings(prefetch):
    uri = "viking://user/alice/memories/preferences/editor.md"
    memory_file = _existing_preference(uri, "editor", "Use Vim")
    page_ids = PageIdMap()
    requests = []
    raw_result = {
        "memory_type": "preferences",
        "page_id": page_ids.get_page_id(uri),
        "content": "1\tUse Vim",
    }
    code = (
        "matches = sdk.search_by_session(limit=5)"
        if prefetch
        else f"memory = sdk.read(uri={uri!r})"
    )
    result = {"matches": [raw_result]} if prefetch else {"memory": raw_result}
    final = (
        "matches[0].content.overwrite('Use Emacs')"
        if prefetch
        else "memory.content.overwrite('Use Emacs')"
    )
    call = ToolCall(id="eval-read", name="eval", arguments={"code": code})
    provider = MagicMock()
    provider.format_prefetch_messages.side_effect = lambda messages: messages
    provider.get_memory_schemas.return_value = [_preference_schema()]
    provider.get_output_language.return_value = "en"
    provider.get_tools.return_value = ["eval"]
    provider.get_extract_context.return_value = SimpleNamespace(page_id_map=page_ids)
    provider.read_file_contents = {}
    provider.instruction.return_value = "Extract memory operations."
    provider.customize_tool_schema.side_effect = lambda name, schema: schema

    async def load_prefetch():
        messages = [{"role": "user", "content": "Prefer Emacs instead."}]
        if prefetch:
            provider.read_file_contents[uri] = memory_file
            add_tool_call_pair_to_messages(messages, "prefetch", "eval", {"code": code}, result)
        return messages

    async def execute(_call):
        provider.read_file_contents[uri] = memory_file
        return result

    async def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        if not prefetch and len(requests) == 1:
            return VLMResponse(tool_calls=[call])
        return final

    provider.prefetch = load_prefetch
    provider.execute_tool = execute
    loop = ExtractLoop(
        vlm=SimpleNamespace(model="test", get_completion_async=complete),
        viking_fs=MagicMock(),
        context_provider=provider,
        max_iterations=3,
    )
    empty = ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[])
    loop.resolve_operations = AsyncMock(return_value=(empty, []))
    loop._check_unread_existing_files = AsyncMock(return_value={})
    config = SimpleNamespace(memory=SimpleNamespace(link_enabled=False))
    with (
        patch("openviking.session.memory.extract_loop.get_openviking_config", return_value=config),
        patch("openviking_cli.utils.config.get_openviking_config", return_value=config),
    ):
        operations, _ = await loop.run()
    assert operations is empty
    assert len(requests) == (1 if prefetch else 2)
    messages = requests[-1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assert "sdk.existing(" in messages[-2]["content"]
    assert messages[-2]["tool_call_id"] == messages[-3]["tool_calls"][0]["id"]
    expression = "matches[0]" if prefetch else "memory"
    assert f"ONLY via: {expression}." in messages[-1]["content"]
    assert "sdk.existing(" not in messages[-1]["content"]
    if not prefetch:
        assert "ONLY via: (none)." in requests[0]["messages"][-1]["content"]
    compiled = loop.resolve_operations.call_args.args[0]
    assert len(compiled.preferences) == 1
    assert compiled.preferences[0].page_id == 1
    assert compiled.preferences[0].content == "Use Emacs"


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_eval_loop_pairs_invalid_arguments_without_extra_user_context(output_format):
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim")],
        available_tools=("eval",),
    )
    # A valid eval variable may itself be named error; it is not a tool failure.
    result = {"error": {"memory_type": "preferences", "content": "1\tUse Vim", "page_id": 1}}
    invalid = {"error": "Invalid tool arguments for eval: expected an object"}
    calls = [
        ToolCall(
            id="valid",
            name="eval",
            arguments={"code": 'error = sdk.read_preferences(topic="editor")'},
        ),
        ToolCall(id="invalid", name="eval", arguments=None),
    ]
    provider = SimpleNamespace(execute_tool=AsyncMock(side_effect=[result, invalid]))
    loop = ExtractLoop(
        vlm=SimpleNamespace(model="test"), viking_fs=MagicMock(), context_provider=provider
    )
    loop._output_context = context
    loop._output_protocol = create_extraction_output_protocol(output_format)
    loop._output_protocol.render_contract(context)
    messages, tools_used = [], []

    assert await loop._execute_tool_calls(messages, calls, tools_used) is False

    roles = ["assistant", "tool", "tool"]
    assert [m["role"] for m in messages] == roles
    assert [c["id"] for c in messages[0]["tool_calls"]] == ["valid", "invalid"]
    assert [m["tool_call_id"] for m in messages[1:3]] == ["valid", "invalid"]
    assert json.loads(messages[0]["tool_calls"][1]["function"]["arguments"]) is None
    assert len(tools_used) == provider.execute_tool.await_count == 2
    if output_format == "python":
        assert _literal_tool_result(messages[2]["content"]) == invalid
        assert messages[1]["content"].startswith("error = sdk.existing(")
        assert messages[1]["content"].count("Use Vim") == 1
    else:
        assert json.loads(messages[2]["content"]) == invalid
        assert json.loads(messages[1]["content"]) == result


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["Unknown tool: missing", "Tool not allowed: missing"])
async def test_eval_loop_preserves_unknown_tool_json_reply(error):
    provider = SimpleNamespace(execute_tool=AsyncMock(return_value={"error": error}))
    loop = ExtractLoop(
        vlm=SimpleNamespace(model="test"), viking_fs=MagicMock(), context_provider=provider
    )
    loop._output_context = _context([_preference_schema()])
    loop._output_protocol = create_extraction_output_protocol("python")
    messages = []

    assert (
        await loop._execute_tool_calls(
            messages, [ToolCall(id="missing", name="missing", arguments={})], []
        )
        is True
    )
    _assert_native_pair(
        messages,
        python_result=True,
        call_id="missing",
        tool_name="missing",
        params={},
        result={"error": error},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_automatic_refetch_uses_unique_native_eval_pairs(output_format):
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context([_preference_schema()])
    loop = ExtractLoop(vlm=SimpleNamespace(model="test"), viking_fs=MagicMock())
    loop._output_context = context
    loop._output_protocol = create_extraction_output_protocol(output_format)
    loop._output_protocol.render_contract(context)
    loop._last_operations_content = "original draft"
    result = {"error": "File not found"}
    messages = []
    for _ in range(2):
        await loop._add_refetch_results_to_messages(messages, {uri: result})
    assert [m["role"] for m in messages] == ["assistant", "assistant", "tool", "user"] * 2
    ids = []
    for index, start in enumerate((0, 4), start=1):
        variable = f"refetched_memory_{index}"
        assert messages[start]["content"] == "original draft"
        pair = messages[start + 1 : start + 3]
        call_id = pair[0]["tool_calls"][0]["id"]
        ids.append(call_id)
        _assert_native_pair(
            pair,
            python_result=output_format == "python",
            call_id=call_id,
            tool_name="eval",
            params={"code": f"{variable} = sdk.read(uri={uri!r})"},
            result={variable: result},
        )
    assert ids[0] != ids[1]
    assert "eval_result" not in json.dumps(messages)


def test_python_contract_omits_link_rules_when_disabled():
    context = _context([_preference_schema()], link_enabled=False)
    protocol = create_extraction_output_protocol("python")

    contract = protocol.render_contract(context)

    assert "## Link Rules" not in contract
    assert "obj.link(...) is unavailable" in contract


def test_python_protocol_compiles_extracted_code_from_fence():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")
    compiled = []

    def record_compile(self, code):
        compiled.append(code)
        return context.operations_model(preferences=[], delete_ids=[])

    with patch(
        "openviking.session.memory.extraction_output_protocol.python_protocol._PythonProgramCompiler.compile",
        autospec=True,
        side_effect=record_compile,
    ):
        operations, error = protocol.parse("```python\nsdk.commit()\n```", context)

    assert error is None
    assert operations is not None
    # The fenced code (not the raw fence wrapper) is what gets compiled.
    assert compiled == ["sdk.commit()"]


def test_python_create_uses_fixed_case_experience_schema():
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "experiences.yaml"))
    schema = registry.get("experiences")
    context = _context([schema], template_context={"language": "en"})
    protocol = create_extraction_output_protocol("python")

    contract = protocol.render_contract(context)
    operations, error = protocol.parse(
        "sdk.create_experiences(experience_name='cancel_booking', content='dag source')\n"
        "sdk.commit()",
        context,
    )

    assert error is None
    assert operations is not None
    assert "supersedes" not in contract
    assert operations.model_dump()["experiences"] == [
        {
            "page_id": 100,
            "experience_name": "cancel_booking",
            "content": "dag source",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_name", ["case_a", '案例_"fixed"'])
@pytest.mark.parametrize(
    ("existing", "method"),
    [(False, "create"), (True, "edit"), (True, "overwrite"), (False, "noop"), (True, "noop")],
)
async def test_experience_prompt_templates_parse_with_real_prefetch_bindings(
    case_name, existing, method
):
    from openviking.session.memory.experience.compiler import compile_dag
    from openviking.session.memory.experience.experience_context import (
        _EXPERIENCE_DAG_EXAMPLE,
        _EXPERIENCE_STAGED_DAG_EXAMPLE,
        AgentExperienceContextProvider,
        _trajectory_model_content,
    )
    from openviking.session.train import Case, Rubric, Trajectory
    from openviking.session.train.components.trajectory_observation import (
        TRAJECTORY_FORMAT,
    )

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "experiences.yaml"))
    schema = registry.get("experiences")
    uri = f"viking://user/u/memories/experiences/{case_name}.md"
    old = MemoryFile(
        uri=uri,
        memory_type="experiences",
        content=_EXPERIENCE_DAG_EXAMPLE,
        extra_fields={"experience_name": case_name},
    )
    context = _context(
        [schema],
        files=[old] if existing else [],
        available_tools=(),
        template_context={"language": "en"},
    )
    protocol = create_extraction_output_protocol("python")
    provider = AgentExperienceContextProvider(
        messages=[],
        case=Case(case_name, "signature", {}, Rubric("reward", "", [])),
        target_uri=uri,
        source_session_uri="viking://session/s/archives/1",
        trajectories=[
            Trajectory(
                name="sample",
                uri="viking://user/u/memories/trajectories/sample.md",
                outcome="failure",
                content="# Session Context\n\n# Trajectory\n\n1. `user` Requests the total\n",
                metadata={
                    "case_name": case_name,
                    "source_session_uri": "viking://session/s/archives/1",
                },
            )
        ],
    )
    for trajectory in provider.trajectories:
        trajectory.metadata["trajectory_format"] = TRAJECTORY_FORMAT
        trajectory.content = render_trajectory_template(trajectory)
    provider._eval_memory = AsyncMock(
        return_value={
            "experience": (
                {
                    "uri": uri,
                    "memory_type": "experiences",
                    "experience_name": case_name,
                    "content": old.content,
                }
                if existing
                else {"error": f"File not found: {uri}"}
            )
        }
    )
    instruction = provider.instruction()
    messages = protocol.render_prefetch_messages(await provider.prefetch(), context)
    tool_reply = next(message["content"] for message in messages if message["role"] == "tool")
    assert ("experience = sdk.existing(" if existing else "experience = sdk.error(") in tool_reply
    messages = provider.format_prefetch_messages(messages)
    assert all(message["role"] == "user" for message in messages)
    assert not any("tool_calls" in message or "tool_call_id" in message for message in messages)
    experience_message = next(
        message["content"]
        for message in messages
        if message["content"].startswith("# Existing Experience")
    )
    assert (old.content if existing else "File not found:") in experience_message
    assert messages[0]["content"].startswith("# Case and Trajectory Batch\n")
    assert messages[1] == {"role": "user", "content": experience_message}
    assert messages[2] == {
        "role": "user",
        "content": _trajectory_model_content(provider.trajectories[0].content),
    }
    assert messages[3]["content"].startswith("Use the Create/Update rules")
    assert len(messages) == 4
    # Graph examples do not choose a write method; Edit usage stays in the shared contract.
    dag_examples = re.findall(r"```text\n(.*?)\n```", instruction, re.DOTALL)
    assert dag_examples == [_EXPERIENCE_STAGED_DAG_EXAMPLE]
    assert "DAG structure example (content only, not the SDK reply)" in instruction
    assert instruction.index(dag_examples[0]) < instruction.index("## Final response:")
    assert "experience.content.overwrite(" not in instruction
    assert "experience.content.edit(" not in instruction
    assert instruction.count(_EXPERIENCE_DAG_EXAMPLE) == 1
    assert "read-only context, not the SDK reply" not in instruction
    assert "shared Edit instructions in the output contract" in instruction
    examples = re.findall(r"```python\n(.*?)\n```", instruction, re.DOTALL)
    assert len(examples) == 1
    assert "<complete DAG source>" not in instruction
    assert "sdk.create_experiences(" in examples[0]
    assert "content='''" in examples[0]
    assert 'dag = workflow("""' in examples[0]
    assert "For no change, return only `sdk.commit()`." in instruction
    programs = {
        "create": examples[0],
        "edit": (
            'experience.content.edit(search="whether the available records cover that scope.", '
            'replace="whether the available records cover that scope, and which records are missing.")'
            "\nsdk.commit()"
        ),
        "overwrite": f"experience.content.overwrite({dag_examples[0]!r})\nsdk.commit()",
        "noop": "sdk.commit()",
    }
    program = programs[method]
    outer = ast.parse(program)
    assert all(isinstance(statement, ast.Expr) for statement in outer.body)
    assert ast.unparse(outer.body[-1].value) == "sdk.commit()"
    assert "sdk.read(" not in program and "sdk.existing(" not in program
    assert "sdk.error(" not in program
    operations, error = protocol.parse(program, context)
    assert error is None
    assert operations is not None
    changes = operations.model_dump()["experiences"]
    if method == "noop":
        assert changes == []
    else:
        assert len(changes) == 1
        assert changes[0]["experience_name"] == case_name
        dag_source = await PatchOp(FieldType.STRING).apply_checked(
            old.content if existing else None, changes[0]["content"]
        )
        assert dag_source.startswith("dag = workflow(")
        compiled = json.loads(compile_dag(dag_source))
        if method == "create":
            assert "total_requested.then(remind_total_scope)" in dag_source
            assert "reservations_known.then(remind_total_scope)" not in dag_source
            assert len(compiled["nodes"]) == 2
            assert "re-check which reservations" in dag_source
            assert "before or after proposed changes" in dag_source
            assert "current rules and evidence before deciding" in dag_source
        elif method == "edit":
            assert dag_source == old.content.replace(
                "whether the available records cover that scope.",
                "whether the available records cover that scope, and which records are missing.",
            )
        else:
            assert len(compiled["nodes"]) == 6
            calls = {
                node.func.id
                for node in ast.walk(ast.parse(dag_source))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            assert calls == {"workflow", "check", "ask", "call", "choose", "tell"}
            assert 'call("get_record_details"' in dag_source
            assert 'classify_state.case("already_matches", review_existing_state)' in dag_source
            assert 'classify_state.case("needs_change", review_change_scope)' in dag_source
        assert "Before any mutation, retrieve every reservation" not in dag_source
        assert "include reservations that the user also plans to modify" not in dag_source
    assert "Identify the trigger, minimal rule change, changed action or output" in instruction
    assert "Either graph shape can be used for creation or updates" in instruction
    assert "Do not add node types just to match an example" in instruction
    assert "Do not repeat questions or tool calls already satisfied" not in instruction
    assert "do not pack a multi-step procedure" not in instruction
    assert "hypothetical Session" in instruction
    assert "as separate check -> tell subgraphs" not in instruction
    assert "workflow/check/ask/call/choose/tell/any_of" in instruction
    assert "Do not assign its result to" in instruction
    assert "even when it is an error" in instruction
    assert "permission, network or other read error" in instruction
    assert "Do not return a bare DAG" in messages[-1]["content"]
    assert "experience.content.overwrite(...)" not in messages[-1]["content"]
    assert "sdk.create_experiences(...)" not in messages[-1]["content"]


def test_experience_schema_allows_focused_node_types_and_tell_successors():
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "experiences.yaml"))
    schema = registry.get("experiences")
    context = _context([schema], template_context={"language": "en"})

    contract = create_extraction_output_protocol("python").render_contract(context)

    assert "focused conditional guidance, not a full task plan" in contract
    assert (
        next(field for field in schema.fields if field.name == "content").merge_op == MergeOp.PATCH
    )
    # SDK rules still come from the common protocol, not a second copy in the schema.
    assert "SDK content editing rules" not in contract
    assert "Full-replacement SDK syntax is mandatory" not in contract
    assert "Only the system may emit sdk.existing() or sdk.error()." in contract
    assert "Never overwrite an existing binding." in contract
    assert "existing_experience" not in contract
    assert "obj.field.edit(search=..., replace=...)" in contract
    assert "obj.field.drop(text=...)" in contract
    assert (
        "Every search must match a unique, non-overlapping region of the original field" in contract
    )
    assert "The final SDK program is write-only." in contract
    assert "the stored result must still be a complete valid DAG" in contract
    assert "Do not map every Case requirement or every observed tool call into a node" in contract
    assert 'ask("""one specific question for information needed from the user""")' in contract
    assert 'call("tool_name", """guidance for arguments using the current context""")' in contract
    assert 'choose("""one classification question about the current context""")' in contract
    assert "Generated Experiences must not use `choose(...)`" not in contract
    assert 'selector.case("label", target)' in contract
    assert "selector.default(target)" in contract
    assert "mutually exclusive cases" in contract
    assert "independent needs that can coexist" in contract
    assert "never for missing or uncertain evidence" in contract
    assert "not use `ask(...)`, `call(...)`" not in contract
    assert "do not turn that action into a DAG step" not in contract
    assert "replace obsolete action nodes" not in contract
    assert "Preserve supported `ask(...)` and `call(...)` nodes" in contract
    assert "A simple check -> tell graph is sufficient" in contract
    assert "Missing user information belongs in ask" in contract
    assert "Do not repeat questions or tool calls already satisfied" in contract
    assert "do not pack a multi-step procedure" in contract
    assert "Every root-to-leaf path must end in a `tell(...)` reminder" in contract
    assert "A `tell(...)` may have successors" in contract
    assert "without waiting for the advice to be followed" in contract
    assert "reminder nodes must not have outgoing edges" not in contract
    assert "does not establish user confirmation or successful tool execution" in contract
    assert "Trigger the reminder as soon as its situation is recognizable" in contract
    assert '"all records have been retrieved"' in contract
    assert "DAG source belongs only inside the content string" in contract
    assert "real bindings for create/update/no-change" in contract
    assert "Example content:" not in contract
    assert "reservations_known.then(remind_total_scope)" not in contract
    assert "confirm_cancel.then(cancel)" not in contract


@pytest.mark.parametrize("phase", ["synthesis", "merge"])
def test_experience_evidence_rules_are_shared_by_synthesis_and_merge(phase):
    from openviking.session.memory.experience.experience_context import (
        AgentExperienceContextProvider,
    )
    from openviking.session.memory.patch_merge_context_provider import (
        PatchMergeContextProvider,
        PatchMergePatch,
    )
    from openviking.session.train import Case, Rubric

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "experiences.yaml"))
    uri = "viking://user/u/memories/experiences/case_a.md"
    if phase == "synthesis":
        provider = AgentExperienceContextProvider(
            messages=[],
            case=Case("case_a", "signature", {}, Rubric("reward", "", [])),
            target_uri=uri,
            source_session_uri="viking://session/s/archives/1",
            memory_registry=registry,
        )
    else:
        provider = PatchMergeContextProvider(
            memory_type="experiences",
            fixed_target=True,
            patches=[
                PatchMergePatch(
                    before_file=None,
                    after_file=MemoryFile(
                        uri=uri,
                        memory_type="experiences",
                        content='dag = workflow("review")\nr = tell("Check the requested scope")',
                        extra_fields={"experience_name": "case_a"},
                    ),
                )
            ],
            memory_registry=registry,
        )
    context = _context(
        provider.get_memory_schemas(None), link_enabled=True, template_context={"language": "en"}
    )
    provider.configure_output_context(context)

    contract = create_extraction_output_protocol("python").render_contract(context)

    assert context.link_enabled is True
    assert "## Link Rules" in contract
    assert "obj.link(target" in contract
    assert "Example — linking" not in contract
    assert "obj_a.link(" not in contract
    assert _NESTED_STRING_GUIDANCE in contract

    instruction = provider.instruction()
    assert "shared Edit instructions in the output contract" in instruction
    assert "prefer edit()/drop()" not in instruction
    assert "one complete valid DAG" in instruction
    assert "Replace the whole content field" not in instruction
    assert "not a snippet patch" not in instruction
    assert "Return one full standalone Python DAG" not in instruction
    assert (
        "Every search must match a unique, non-overlapping region of the original field" in contract
    )
    assert "SDK content editing rules" not in contract
    assert "Only the system may emit sdk.existing() or sdk.error()." in contract
    assert "Never overwrite an existing binding." in contract
    assert "The final SDK program is write-only." in contract
    assert "Keep search as small as possible" in contract
    assert "Keep all still-valid existing constraints." in contract
    assert "Missing user information belongs in ask" in contract
    assert "Do not repeat questions or tool calls already satisfied" in contract
    assert "do not pack a multi-step procedure" in contract
    assert "Never copy concrete user, reservation, payment, flight" in contract
    assert "even if the same ID appears in every Trajectory" in contract
    assert "Refer to the entity by its role in the current request" in contract
    assert "preserve unaffected rules, node names, and relationships" in contract
    assert contract.count("AND semantics") == 1
    assert (
        "multiple `.then(reminder)` edges require all their predecessors, not just one" in contract
    )
    assert "Split independent conditions into separate check nodes" in contract
    assert "any_of(node_a, node_b, ...)" in contract
    assert "either = any_of(a, b)" in contract
    assert "(a OR b) AND confirmed" in contract
    assert "without a model call" in contract
    assert "it does not add an alternative trigger" in contract
    for duplicated_rule in (
        "Missing user information belongs in ask",
        "Do not repeat questions or tool calls already satisfied",
        "do not pack a multi-step procedure",
        "Never copy a concrete user",
        "Preserve unaffected node",
        "with unaffected paths preserved",
    ):
        assert duplicated_rule not in instruction
    rule_text = " ".join(contract.split())
    for rule in (
        "External evaluation is authoritative for outcome and requirement compliance",
        "Do not reinterpret an evaluated failure as success",
        "evaluator-backed successful comparison behavior takes precedence over the failed "
        "trajectory's interpretation",
        "Derive a narrow runtime-observable correction or exception from differing inputs and actions",
        "A failed trajectory need not independently prove the successful comparison action",
        "delivered before the relevant decision boundary",
        "the corresponding failure still occurred",
        "require a concrete decision-rule or action delta",
        "Reject paraphrases, stronger wording or extra checklists that preserve the same decision logic",
        "preserve nearby correct cases",
        "Do not invent hidden causes, unsupported policy rules or tool capabilities",
        "A successful comparison alone does not authorize overriding an explicit source rule",
    ):
        assert rule_text.count(rule) == 1
        assert rule not in " ".join(instruction.split())
    for obsolete in (
        "rather than prescribing the final action",
        "Current authoritative policy and current user/tool evidence determine the decision",
        "Leave the final choice to the agent after review",
        "Apply this review-oriented wording",
        "retain the conflict rather than inventing a rule to output a grader literal",
        "the decision belongs to the agent using current policy and facts",
    ):
        assert obsolete not in " ".join((contract + instruction).split())
    assert "one evidence-backed, actionable correction or verification" in contract
    assert "applicability conditions, exceptions, scope and timing" in contract
    assert "Do not expand a restriction on one operation to the whole goal" in contract
    assert "check the restriction's scope and other allowed approaches" in contract
    assert "task requirements and constraints" in contract
    assert 'not just "be careful" or "review the policy"' in contract
    assert "Do not request a full policy recap, lengthy reasoning" in contract
    assert 'ask("""one specific question for information needed from the user""")' in contract
    assert 'call("tool_name", """guidance for arguments using the current context""")' in contract
    assert "A `tell(...)` may have successors" in contract
    assert "without waiting for the advice to be followed" in contract
    assert 'choose("""one classification question about the current context""")' in contract
    assert "Generated Experiences must not use `choose(...)`" not in contract
    assert 'selector.case("label", target)' in contract
    assert "selector.default(target)" in contract
    assert "mutually exclusive cases" in contract
    assert "independent needs that can coexist" in contract
    assert "never for missing or uncertain evidence" in contract


def test_python_trajectory_output_exposes_only_extracted_fields():
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "trajectories.yaml"))
    schema = registry.get("trajectories")
    context = _context([schema], template_context={"language": "en"})
    protocol = create_extraction_output_protocol("python")

    contract = protocol.render_contract(context)
    operations, error = protocol.parse(
        "sdk.create_trajectories(case_name='case_a', context_rules='', steps='Observed steps')\nsdk.commit()",
        context,
    )

    assert error is None
    assert operations is not None
    assert {field.name for field in schema.fields} == {
        "case_name",
        "context_rules",
        "steps",
        "evaluation_criteria",
        "action_scores",
    }
    for name in [
        "trajectory_name",
        "task_spec",
        "outcome",
        "trajectory_format",
        "evaluation",
        "experience_execution",
        "experience_update",
    ]:
        assert f"{name}: str" not in contract
    item = operations.model_dump()["trajectories"][0]
    assert set(item) == {
        "page_id",
        "case_name",
        "context_rules",
        "steps",
        "evaluation_criteria",
        "action_scores",
    }
    assert item["steps"] == "Observed steps"


def test_json_protocol_uses_native_tool_messages():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("json")
    result = {"memory_type": "preferences", "content": "1\tUse Vim"}
    params = {"uri": "viking://user/alice/memories/preferences/editor.md"}
    messages = protocol.render_tool_result_messages(
        context,
        call_id="call-1",
        tool_name="read",
        params=params,
        result=result,
        source="tool call",
    )
    assert len(messages) == 2
    _assert_native_pair(messages, call_id="call-1", tool_name="read", params=params, result=result)


@pytest.mark.parametrize("available_tools", [("read",), ()])
@pytest.mark.parametrize(
    "result", [[], [{"uri": "viking://user/alice/memories/preferences/editor.md", "score": 0.9}]]
)
def test_python_protocol_renders_search_as_native_python_result(available_tools, result):
    context = _context([_preference_schema()], available_tools=available_tools)
    protocol = create_extraction_output_protocol("python")
    params = {"query": "editor"}
    messages = protocol.render_tool_result_messages(
        context,
        call_id="call-1",
        tool_name="search",
        params=params,
        result=result,
        source="prefetch",
    )
    assert len(messages) == 2
    _assert_native_pair(
        messages,
        python_result=True,
        call_id="call-1",
        tool_name="search",
        params=params,
        result=result,
    )
    assert "sdk.existing(" not in json.dumps(messages)


def test_python_read_reply_is_binding_code_with_partial_content_and_metadata():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim\nKeep plugins small", 2)],
    )
    protocol = create_extraction_output_protocol("python")
    result = {
        "memory_type": "preferences",
        "topic": "editor",
        "score": 2,
        "content": "2\tKeep plugins small",
        "page_id": 1,
        "memory_maintenance_notice": {
            "maintenance_required": True,
            "guidance": "choose split or compact",
        },
    }
    original = copy.deepcopy(result)
    messages = protocol.render_tool_result_messages(
        context,
        call_id="call-1",
        tool_name="read",
        params={"uri": uri, "offset": 1, "limit": 1},
        result=result,
        source="tool call",
    )
    assert result == original
    assert [m["role"] for m in messages] == ["assistant", "tool"]
    assert messages[1]["tool_call_id"] == messages[0]["tool_calls"][0]["id"]
    code = messages[1]["content"]
    assert code.startswith("read = sdk.existing(")
    assert "content='Keep plugins small'" in code
    assert "Use Vim" not in code and "2\\tKeep plugins small" not in code
    assert "choose split or compact" in code and "'maintenance_required': True" in code
    assert "page_id" not in code
    assert "content_loaded_as_existing_memory" not in code
    ast.parse(code)
    operations, error = protocol.parse(
        "read.content.edit(search='Keep plugins small', replace='Keep plugins minimal')", context
    )
    assert error is None
    assert operations.preferences[0].page_id == 1


def test_python_eval_reply_uses_requested_variable_for_existing_memory():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim", 2)]
    )
    protocol = create_extraction_output_protocol("python")
    result = {
        "memory": {
            "memory_type": "preferences",
            "topic": "editor",
            "score": 2,
            "content": "1\tUse Vim",
            "page_id": 1,
        }
    }
    messages = protocol.render_tool_result_messages(
        context,
        call_id="call-1",
        tool_name="eval",
        params={"code": f"memory = sdk.read(uri={uri!r})"},
        result=result,
        source="tool call",
    )
    assert [m["role"] for m in messages] == ["assistant", "tool"]
    code = messages[1]["content"]
    assert code.startswith("memory = sdk.existing(")
    assert "content='Use Vim'" in code
    assert code.count("Use Vim") == 1
    assert "preferences_1" not in code and "eval_result" not in code
    assert "content_loaded_as_existing_memory" not in code
    operations, error = protocol.parse("memory.content.overwrite('Use Emacs')", context)
    assert error is None
    assert operations.preferences[0].page_id == 1


@pytest.mark.parametrize("source", ["prefetch", "tool call", "automatic read"])
@pytest.mark.parametrize("failed", [False, True])
def test_python_eval_context_uses_native_request_and_python_result(source, failed):
    context = _context([_preference_schema()], available_tools=("eval",))
    protocol = create_extraction_output_protocol("python")
    code = '\nfirst = sdk.read_preferences(topic="editor")\nsecond = sdk.read_preferences(topic="formatting")\n'
    result = (
        {"error": "invalid program"}
        if failed
        else {"first": {"error": "File not found"}, "second": {"error": "File not found"}}
    )
    messages = protocol.render_tool_result_messages(
        context,
        call_id="call-1",
        tool_name="eval",
        params={"code": code},
        result=result,
        source=source,
    )
    assert len(messages) == 2
    _assert_native_pair(
        messages,
        python_result=True,
        call_id="call-1",
        tool_name="eval",
        params={"code": code},
        result=result,
    )
    assert "eval_result" not in json.dumps(messages) and "eval_error" not in json.dumps(messages)


@pytest.mark.parametrize("params", [None, {}, {"code": None}, {"code": 123}])
def test_python_eval_context_preserves_invalid_request_arguments(params):
    context = _context([_preference_schema()], available_tools=("eval",))
    protocol = create_extraction_output_protocol("python")
    result = {"error": "code must be a non-empty string"}
    messages = protocol.render_tool_result_messages(
        context,
        call_id="call-1",
        tool_name="eval",
        params=params,
        result=result,
        source="tool call",
    )
    _assert_native_pair(
        messages,
        python_result=True,
        call_id="call-1",
        tool_name="eval",
        params=params,
        result=result,
    )


@pytest.mark.parametrize("source", ["prefetch", "tool call", "automatic read"])
def test_python_eval_code_preserves_full_unbound_nested_results_without_compaction(source):
    context = _context([_preference_schema()], available_tools=("eval",))
    protocol = create_extraction_output_protocol("python")
    result = {
        "search": [
            {"memory_type": "preferences", "content": "1\tUse Vim", "page_id": 1},
            {"memory_type": "preferences", "content": "", "page_id": 2},
        ],
        "read": {"memory_type": "preferences", "content": None},
        "missing": {"error": "File not found"},
        "enabled": True,
    }
    original = copy.deepcopy(result)
    params = {"code": "search = sdk.search_by_session(limit=5)"}
    messages = protocol.render_tool_result_messages(
        context, call_id="full-json", tool_name="eval", params=params, result=result, source=source
    )
    assert result == original
    _assert_native_pair(
        messages,
        python_result=True,
        call_id="full-json",
        tool_name="eval",
        params=params,
        result=original,
    )
    assert "content_loaded_as_existing_memory" not in json.dumps(messages)


def test_python_prefetch_keeps_short_native_eval_request_without_repeating_query():
    context = _context([_preference_schema()], available_tools=("eval",))
    protocol = create_extraction_output_protocol("python")
    conversation = "SESSION_QUERY_MARKER " + "detail " * 1000
    code = "prefetched_search = sdk.search_by_session(limit=5)"
    result = {"prefetched_search": []}
    prefetched = [{"role": "user", "content": conversation}]
    add_tool_call_pair_to_messages(prefetched, "prefetch-1", "eval", {"code": code}, result)
    messages = protocol.render_prefetch_messages(prefetched, context)
    assert len(messages) == 3 and messages[0]["content"] == conversation
    _assert_native_pair(
        messages[1:],
        python_result=True,
        call_id="prefetch-1",
        tool_name="eval",
        params={"code": code},
        result=result,
    )
    assert "SESSION_QUERY_MARKER" not in json.dumps(messages[1:])
    assert "query=" not in json.dumps(messages[1:])


def test_python_prefetch_keeps_all_result_code_inside_tool_replies():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim", 2)]
    )
    protocol = create_extraction_output_protocol("python")
    protocol.render_contract(context)
    prefetched = [{"role": "user", "content": "Conversation History"}]
    add_tool_call_pair_to_messages(
        prefetched, "search-1", "search", {"query": "editor"}, [{"uri": uri, "score": 0.9}]
    )
    add_tool_call_pair_to_messages(
        prefetched,
        "read-1",
        "read",
        {"uri": uri},
        {"memory_type": "preferences", "topic": "editor", "score": 2, "content": "1\tUse Vim"},
    )
    messages = protocol.render_prefetch_messages(prefetched, context)
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert messages[1] == prefetched[1] and messages[3] == prefetched[3]
    assert messages[2]["tool_call_id"] == "search-1" and messages[4]["tool_call_id"] == "read-1"
    combined = "\n".join(m.get("content") or "" for m in messages)
    assert _literal_tool_result(messages[2]["content"], "search") == [{"uri": uri, "score": 0.9}]
    assert messages[4]["content"].startswith("read = sdk.existing(")
    assert combined.count("Use Vim") == 1
    assert "eval_result" not in combined and "read_result" not in combined


@pytest.mark.parametrize(
    ("order", "error"),
    [
        ([1], "no matching request"),
        ([0], "missing results"),
        ([0, 2], "interrupted pending"),
        ([0, 0], "before answering prior calls"),
        ([0, 1, 0, 1], "Duplicate prefetch tool call ID"),
        ([0, 1, 1], "no matching request"),
    ],
)
def test_python_prefetch_rejects_unbalanced_native_pairs(order, error):
    protocol = create_extraction_output_protocol("python")
    pair = []
    add_tool_call_pair_to_messages(pair, "prefetch-1", "eval", {"code": "x = sdk.read()"}, {})
    pair.append({"role": "user", "content": "interruption"})
    with pytest.raises(ValueError, match=error):
        protocol.render_prefetch_messages(
            [pair[i] for i in order], _context([_preference_schema()])
        )


def test_python_prefetch_preserves_parallel_calls_and_reply_order():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim")]
    )
    protocol = create_extraction_output_protocol("python")
    protocol.render_contract(context)
    pairs = []
    for i in range(2):
        add_tool_call_pair_to_messages(
            pairs,
            f"prefetch-{i}",
            "eval",
            {"code": 'editor = sdk.read_preferences(topic="editor")'},
            {"editor": {"memory_type": "preferences", "content": "1\tUse Vim", "page_id": 1}},
        )
    request = {**pairs[0], "tool_calls": pairs[0]["tool_calls"] + pairs[2]["tool_calls"]}
    messages = protocol.render_prefetch_messages([request, pairs[3], pairs[1]], context)
    assert [m["role"] for m in messages] == ["assistant", "tool", "tool"]
    assert messages[0] == request
    assert [m["tool_call_id"] for m in messages[1:3]] == ["prefetch-1", "prefetch-0"]
    assert messages[1]["content"].startswith("editor = sdk.existing(")
    assert messages[2]["content"] == messages[1]["content"]
    operations, error = protocol.parse("editor.content.overwrite('Use Emacs')", context)
    assert error is None
    assert len(operations.preferences) == 1


def _render_eval_results(protocol, context, result, *, call_id="result"):
    code = "\n".join(f"{name} = sdk.search_by_session(limit=5)" for name in result)
    return protocol.render_tool_result_messages(
        context,
        call_id=call_id,
        tool_name="eval",
        params={"code": code},
        result=result,
        source="tool call",
    )


def test_python_search_results_bind_nested_objects_for_edit_delete_and_replacement():
    first = "viking://user/alice/memories/preferences/editor.md"
    second = "viking://user/alice/memories/preferences/duplicate.md"
    context = _context(
        [_preference_schema()],
        files=[
            _existing_preference(first, "editor", "Use Vim"),
            _existing_preference(second, "duplicate", "Duplicate Vim preference"),
        ],
    )
    protocol = create_extraction_output_protocol("python")
    result = {
        "matches": [
            {"memory_type": "preferences", "page_id": 1, "content": "1\tUse Vim"},
            {"memory_type": "preferences", "page_id": 2, "content": "1\tDuplicate Vim preference"},
        ]
    }
    original = copy.deepcopy(result)
    messages = _render_eval_results(protocol, context, result)
    assert [m["role"] for m in messages] == ["assistant", "tool"]
    code = messages[1]["content"]
    assert code.startswith("matches = [sdk.existing(")
    assert code.count("sdk.existing(") == 2
    assert code.count("content='Use Vim'") == 1
    ast.parse(code)
    assert result == original
    operations, error = protocol.parse(
        "matches[0].content.edit(search='Use Vim', replace='Use Emacs')\n"
        "matches[1].delete(replacement=matches[0])",
        context,
    )
    assert error is None, error
    assert len(operations.preferences) == 1
    assert operations.preferences[0].page_id == 1
    assert len(operations.delete_ids) == 1
    assert operations.delete_ids[0].delete_page_id == 2
    assert operations.delete_ids[0].replacement_page_id == 1


def test_python_eval_aliases_share_one_object_and_reset_for_each_parse():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim")]
    )
    protocol = create_extraction_output_protocol("python")
    value = {"memory_type": "preferences", "page_id": 1, "content": "1\tUse Vim"}
    _render_eval_results(protocol, context, {"first": value}, call_id="one")
    _render_eval_results(protocol, context, {"second": value}, call_id="two")
    operations, error = protocol.parse(
        "first.content.overwrite('Use Emacs')\nsecond.content.overwrite('Use Neovim')", context
    )
    assert error is None, error
    assert len(operations.preferences) == 1
    assert operations.preferences[0].content == "Use Neovim"
    unchanged, error = protocol.parse("sdk.commit()", context)
    assert error is None
    assert unchanged.preferences == []
    assert context.read_file_contents[uri].plain_content() == "Use Vim"


def test_python_eval_reassignment_uses_latest_result_and_invalidates_failed_reference():
    first = "viking://user/alice/memories/preferences/editor.md"
    second = "viking://user/alice/memories/preferences/formatting.md"
    context = _context(
        [_preference_schema()],
        files=[
            _existing_preference(first, "editor", "Use Vim"),
            _existing_preference(second, "formatting", "Use spaces"),
        ],
    )
    protocol = create_extraction_output_protocol("python")
    _render_eval_results(
        protocol, context, {"memory": {"memory_type": "preferences", "page_id": 1}}
    )
    _render_eval_results(
        protocol, context, {"memory": {"memory_type": "preferences", "page_id": 2}}
    )
    operations, error = protocol.parse("memory.content.overwrite('Use tabs')", context)
    assert error is None
    assert [op.page_id for op in operations.preferences] == [2]
    _render_eval_results(protocol, context, {"memory": {"error": "File not found"}})
    operations, error = protocol.parse("memory.delete()", context)
    assert operations is None
    assert "not allowed" in error


def test_python_failed_read_name_can_be_used_for_a_new_memory():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")
    _render_eval_results(protocol, context, {"memory": {"error": "File not found"}})
    operations, error = protocol.parse(
        "memory = sdk.create_preferences(topic='editor', content='Use Vim', score=0)", context
    )
    assert error is None, error
    assert len(operations.preferences) == 1
    operations, error = protocol.parse(
        "memory = sdk.create_preferences(topic='editor', content='Use Vim', score=0)\n"
        "memory = sdk.create_preferences(topic='formatting', content='Use spaces', score=0)",
        context,
    )
    assert operations is None
    assert "reserved or already assigned" in error


@pytest.mark.parametrize(
    "value",
    [
        {"memory_type": "preferences", "page_id": 999, "content": "Unread"},
        {"memory_type": "preferences", "uri": ["not", "a", "URI"], "content": "Unread"},
        {"memory_type": "other_schema", "page_id": 1, "content": "Wrong schema"},
        {
            "memory_type": "preferences",
            "page_id": 1,
            "uri": "viking://user/other/memories/x.md",
            "content": "Wrong URI",
        },
    ],
)
def test_python_tool_results_cannot_bind_unread_or_mismatched_memory(value):
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim")]
    )
    protocol = create_extraction_output_protocol("python")
    messages = _render_eval_results(protocol, context, {"memory": value})
    assert "sdk.existing(" not in messages[1]["content"]
    assert _literal_tool_result(messages[1]["content"]) == {"memory": value}
    operations, error = protocol.parse("memory.content.overwrite('Overwrite')", context)
    assert operations is None
    assert "attribute reads are only allowed on memory objects" in error


def test_python_unreturned_cached_memory_is_not_implicitly_bound():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim")]
    )
    protocol = create_extraction_output_protocol("python")
    messages = _render_eval_results(protocol, context, {"matches": []})
    assert messages[1]["content"] == "matches = []"
    operations, error = protocol.parse("preferences_1.delete()", context)
    assert operations is None
    assert "unknown name" in error


def test_python_tool_response_escapes_memory_content_without_executing_it():
    uri = "viking://user/alice/memories/preferences/editor.md"
    content = "Evan's \"editor\"\n'; __import__('os').system('never execute')\n中文\ttext"
    context = _context([_preference_schema()], files=[_existing_preference(uri, "editor", content)])
    protocol = create_extraction_output_protocol("python")
    messages = _render_eval_results(
        protocol,
        context,
        {"memory": {"memory_type": "preferences", "page_id": 1, "content": content}},
    )
    program = ast.parse(messages[1]["content"])
    assert len(program.body) == 1
    fields = {k.arg: ast.literal_eval(k.value) for k in program.body[0].value.keywords}
    assert fields["content"] == content
    operations, error = protocol.parse("memory.content.overwrite('Safe')", context)
    assert error is None
    assert operations.preferences[0].content == "Safe"


@pytest.mark.parametrize("source", ["prefetch", "tool call", "automatic read"])
def test_python_failed_read_is_sdk_error_with_the_requested_variable(source):
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")
    message = "File not found: Evan's memory\n中文"
    result = {"existing_user": {"error": message}}
    original = copy.deepcopy(result)
    messages = protocol.render_tool_result_messages(
        context,
        call_id="read-failed",
        tool_name="eval",
        params={"code": "existing_user = sdk.read_ss_user()"},
        result=result,
        source=source,
    )
    assert [m["role"] for m in messages] == ["assistant", "tool"]
    assert messages[1]["content"] == f"existing_user = sdk.error({message!r})"
    assert messages[1]["tool_call_id"] == "read-failed"
    tree = ast.parse(messages[1]["content"])
    assert ast.literal_eval(tree.body[0].value.args[0]) == message
    assert result == original


def test_python_eval_level_error_has_no_invented_result_variable():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")
    messages = protocol.render_tool_result_messages(
        context,
        call_id="bad-eval",
        tool_name="eval",
        params={"code": "invalid("},
        result={"error": "invalid program"},
        source="tool call",
    )
    assert messages[1]["content"] == "sdk.error('invalid program')"
    assert "eval_error" not in messages[1]["content"]


def test_python_error_hides_internal_page_id_but_keeps_diagnostics():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")
    result = {"memory": {"error": "Permission denied", "page_id": 7, "retryable": False}}
    messages = _render_eval_results(protocol, context, result)
    code = messages[1]["content"]
    assert code.startswith("memory = sdk.error('Permission denied')")
    assert "page_id" not in code
    assert "'retryable': False" in code
    assert result["memory"]["page_id"] == 7


def test_python_existing_result_hides_page_id_and_preserves_internal_mapping():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim")]
    )
    protocol = create_extraction_output_protocol("python")
    result = {"memory": {"memory_type": "preferences", "page_id": 1, "content": "1\tUse Vim"}}
    messages = _render_eval_results(protocol, context, result)
    code = messages[1]["content"]
    assert code.startswith("memory = sdk.existing(")
    assert "page_id" not in code and "# Metadata" not in code
    assert context.page_id_map.resolve(1) == uri
    operations, error = protocol.parse("memory.content.overwrite('Use Emacs')", context)
    assert error is None
    assert operations.preferences[0].page_id == 1


@pytest.mark.parametrize("program", ["sdk.error('failed')", "failed = sdk.error('failed')"])
def test_python_sdk_error_is_not_a_final_write_operation(program):
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")
    operations, error = protocol.parse(program, context)
    assert operations is None
    assert "sdk.error() is reserved for system-provided tool results" in error
    assert "sdk.commit()" in error


def test_python_contract_uses_set_for_single_file_schema_and_create_for_collection():
    context = _context(
        [_profile_schema(), _preference_schema()],
        role_scope=RoleScope(user_ids=["alice"], peer_ids=["bob"]),
    )
    protocol = create_extraction_output_protocol("python")

    contract = protocol.render_contract(context)

    assert "sdk.set_profile" in contract
    assert "- sdk.create_profile(*," not in contract
    assert (
        "profile is a SINGLETON per identity: sdk.create_profile() does not exist, and "
        "sdk.set_profile() must appear AT MOST ONCE for each identity"
    ) in contract
    assert (
        "Current identity peer_id='bob': exactly one "
        "sdk.set_profile(peer_id='bob', ...) call; combine every person and section"
    ) in contract
    assert "Memory type rules:" in contract
    assert "User profile" in contract
    assert "sdk.create_preferences" in contract
    assert "sdk.set_preferences" not in contract
    assert "canonical = sdk.create_<type>(...)" in contract
    assert "duplicate_1.delete(replacement=canonical)" in contract
    assert "Identity fields (primary key): peer_id." in contract
    assert "Identity fields (primary key): peer_id, topic." in contract
    assert "Current self identity: exactly one sdk.set_profile() call without peer_id" in contract


def test_python_reserved_existing_retry_explains_new_replacement_binding():
    protocol = create_extraction_output_protocol("python")

    retry = protocol.render_format_retry(
        "Line 3: sdk.existing() is reserved for system-provided bindings"
    )

    assert "existing-object variable names already supplied by the system" in retry
    assert "canonical = sdk.create_<type>(...)" in retry
    assert "duplicate_1.delete(replacement=canonical)" in retry


def test_python_string_literal_retry_pushes_outer_triple_single_quotes():
    protocol = create_extraction_output_protocol("python")

    retry = protocol.render_format_retry("Line 33: invalid syntax. Perhaps you forgot a comma?")

    assert "offending line is shown above" in retry
    assert "triple-single-quoted string ('''...''')" in retry


def test_python_non_string_syntax_error_omits_quote_guidance():
    protocol = create_extraction_output_protocol("python")

    retry = protocol.render_format_retry("Line 3: unexpected indent")

    assert "offending line is shown above" not in retry


def test_resolution_repair_matches_protocol_output_shape():
    issues = [
        {
            "memory_type": "events",
            "page_id": 100,
            "reason_code": "no_writable_target",
            "reason": "range out of bounds",
            "operation": {"event_name": "yoga_class_started", "ranges": "99"},
        }
    ]

    json_repair = create_extraction_output_protocol("json").render_resolution_repair(issues)
    assert "ONLY one JSON object" in json_repair
    assert "yoga_class_started" in json_repair

    python_repair = create_extraction_output_protocol("python").render_resolution_repair(issues)
    assert "sdk.create_events" in python_repair
    assert "Output only Python code." in python_repair
    assert "JSON object" not in python_repair
    assert "triple-single-quoted strings ('''...''')" in python_repair


def _kebab_schema() -> MemoryTypeSchema:
    return MemoryTypeSchema(
        memory_type="project-notes",
        description="Project notes",
        directory="viking://user/{{ user_space }}/memories/project-notes",
        filename_template="{{ project_name }}.md",
        fields=[
            MemoryField(
                name="project_name",
                field_type=FieldType.STRING,
                merge_op=MergeOp.IMMUTABLE,
            ),
            MemoryField(
                name="note-body",
                field_type=FieldType.STRING,
                merge_op=MergeOp.PATCH,
            ),
        ],
    )


def test_python_contract_aliases_non_identifier_schema_names():
    context = _context([_kebab_schema()])
    protocol = create_extraction_output_protocol("python")

    contract = protocol.render_contract(context)

    # Method/parameter names are folded to valid identifiers on the DSL surface.
    assert "sdk.create_project_notes(" in contract
    assert "note_body" in contract
    assert "sdk.create_project-notes" not in contract


def test_python_compiles_aliased_names_back_to_real_schema():
    context = _context([_kebab_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        """
sdk.create_project_notes(
    project_name="atlas",
    note_body=\"\"\"Kickoff on 2023-06-09.\"\"\",
)
sdk.commit()
""",
        context,
    )

    assert error is None, error
    dumped = operations.model_dump()
    # Real schema name and real field name are preserved in the operations.
    assert dumped["project-notes"] == [
        {"page_id": 100, "project_name": "atlas", "note-body": "Kickoff on 2023-06-09."}
    ]


def test_python_edits_aliased_field_on_existing_object():
    existing = MemoryFile(
        uri="viking://user/alice/memories/project-notes/atlas.md",
        memory_type="project-notes",
        content="",
        extra_fields={
            "project_name": "atlas",
            "note-body": "Kickoff on 2023-06-09.",
            "version": 3,
            "_uri": "x",
        },
    )
    context = _context([_kebab_schema()], files=[existing])
    protocol = create_extraction_output_protocol("python")

    bindings = _bind(protocol, context)
    # Existing-object binding exposes the field under its identifier alias.
    assert "note_body=" in bindings

    var = bindings.split(" = ", 1)[0].strip().splitlines()[-1]
    operations, error = protocol.parse(
        f"""
{var}.note_body.edit(search=\"\"\"Kickoff on 2023-06-09.\"\"\", replace=\"\"\"Kickoff moved to 2023-07-01.\"\"\")
sdk.commit()
""",
        context,
    )

    assert error is None, error
    edited = operations.model_dump()["project-notes"]
    assert edited and edited[0]["page_id"] == 1


def _schema_named(memory_type: str, field_names: list[str]) -> MemoryTypeSchema:
    return MemoryTypeSchema(
        memory_type=memory_type,
        description=memory_type,
        directory=f"viking://user/{{{{ user_space }}}}/memories/{memory_type}",
        filename_template="{{ name }}.md",
        fields=[
            MemoryField(name=name, field_type=FieldType.STRING, merge_op=MergeOp.PATCH)
            for name in field_names
        ],
    )


def test_python_rejects_colliding_memory_type_aliases():
    # 'project-notes' and 'project_notes' both fold to the same DSL alias.
    context = _context(
        [_schema_named("project-notes", ["name"]), _schema_named("project_notes", ["name"])]
    )
    protocol = create_extraction_output_protocol("python")

    with pytest.raises(ValueError, match="same Python DSL alias"):
        protocol.render_contract(context)


def test_python_rejects_colliding_field_aliases():
    # 'note-body' and 'note_body' collide within one schema.
    context = _context([_schema_named("notes", ["note-body", "note_body"])])
    protocol = create_extraction_output_protocol("python")

    with pytest.raises(ValueError, match="same Python DSL alias"):
        protocol.render_contract(context)


def test_python_parse_rejects_colliding_aliases_without_render():
    # Collision must be caught even when parse() runs without render_contract().
    context = _context(
        [_schema_named("project-notes", ["name"]), _schema_named("project_notes", ["name"])]
    )
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse("sdk.commit()", context)

    assert operations is None
    assert "same Python DSL alias" in error


def test_python_syntax_error_includes_offending_source_line():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")

    # A double-quoted summary that itself embeds a double quote ends the string
    # early; the compiler should surface the real offending line, not just the
    # misleading "forgot a comma" hint.
    _operations, error = protocol.parse(
        'sdk.create_preferences(topic="x", content="he saw "Little Women" today", score=1)',
        context,
    )

    assert error is not None
    assert "invalid Python syntax" in error
    assert "Little Women" in error
    assert "^" in error


def test_memory_schema_identity_fields_follow_scope_and_uri_template():
    entities = MemoryTypeSchema(
        memory_type="entities",
        directory="viking://user/{{ user_space }}/memories/entities",
        filename_template="{{ category|lower }}/{{ name|lower }}.md",
        fields=[
            MemoryField(name="category", field_type=FieldType.STRING),
            MemoryField(name="name", field_type=FieldType.STRING),
            MemoryField(name="content", field_type=FieldType.STRING),
        ],
    )
    preferences = MemoryTypeSchema(
        memory_type="preferences",
        directory="viking://user/{{ user_space }}/memories/preferences",
        filename_template="{{ user }}/{{ topic }}.md",
        fields=[
            MemoryField(name="user", field_type=FieldType.STRING),
            MemoryField(name="topic", field_type=FieldType.STRING),
            MemoryField(name="content", field_type=FieldType.STRING),
        ],
    )

    assert _profile_schema().identity_fields() == ("peer_id",)
    assert preferences.identity_fields() == ("peer_id", "user", "topic")
    assert entities.identity_fields() == ("peer_id", "category", "name")


def test_memory_schema_identity_fields_ignore_jinja_token_collisions():
    # `year` and `context` appear only inside a filter/function/attribute
    # expression, never as referenced variables, so they must not be treated
    # as identity fields. Only `ranges` and `event_name` are real references.
    events = MemoryTypeSchema(
        memory_type="events",
        directory="viking://user/{{ user_space }}/memories/events",
        filename_template="{{ extract_context.get_year(ranges) }}/{{ event_name }}.md",
        fields=[
            MemoryField(name="event_name", field_type=FieldType.STRING),
            MemoryField(name="ranges", field_type=FieldType.STRING),
            MemoryField(name="year", field_type=FieldType.STRING),
            MemoryField(name="context", field_type=FieldType.STRING),
            MemoryField(name="content", field_type=FieldType.STRING),
        ],
    )

    assert events.identity_fields() == ("peer_id", "event_name", "ranges")


def test_python_set_single_file_memory_compiles_to_same_operations_as_json():
    context = _context([_profile_schema()])
    python_protocol = create_extraction_output_protocol("python")
    json_protocol = create_extraction_output_protocol("json")

    python_operations, python_error = python_protocol.parse(
        "sdk.set_profile(content='Engineer')\nsdk.commit()",
        context,
    )
    json_operations, json_error = json_protocol.parse(
        '{"profile":[{"page_id":100,"content":"Engineer"}],"delete_ids":[]}',
        context,
    )

    assert python_error is None
    assert json_error is None
    assert python_operations.model_dump() == json_operations.model_dump()


def test_python_commit_is_optional():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    # A program without a trailing sdk.commit() is accepted as-is.
    without_commit, error = protocol.parse("sdk.set_profile(content='Engineer')", context)
    assert error is None
    assert without_commit.model_dump()["profile"][0]["content"] == "Engineer"

    # An empty program means "no changes".
    empty_operations, empty_error = protocol.parse("", context)
    assert empty_error is None
    assert empty_operations.model_dump() == {"profile": [], "delete_ids": []}


def test_python_rejects_create_for_single_file_and_set_for_collection():
    context = _context([_profile_schema(), _preference_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        "sdk.create_profile(content='Engineer')\nsdk.commit()", context
    )
    assert operations is None
    assert "use sdk.set_profile()" in error

    operations, error = protocol.parse(
        "sdk.set_preferences(topic='editor', content='Use Vim', score=1)\nsdk.commit()",
        context,
    )
    assert operations is None
    assert "use sdk.create_preferences()" in error


def test_python_rejects_repeated_single_file_set_for_same_peer_scope():
    context = _context(
        [_profile_schema()],
        role_scope=RoleScope(user_ids=["alice"], peer_ids=["bob"]),
    )
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        """
sdk.set_profile(peer_id="bob", content="Engineer")
sdk.set_profile(peer_id="bob", content="Musician")
sdk.commit()
""",
        context,
    )

    assert operations is None
    assert "duplicate profile identity (peer_id='bob')" in error
    assert "may be called only once for peer 'bob'" in error
    assert "combine all people and content sections" in error


def test_python_rejects_repeated_single_file_set_for_self_scope():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        """
sdk.set_profile(content="Engineer")
sdk.set_profile(content="Musician")
sdk.commit()
""",
        context,
    )

    assert operations is None
    assert "may be called only once for self" in error


def test_python_allows_one_single_file_set_per_peer_scope():
    context = _context(
        [_profile_schema()],
        role_scope=RoleScope(user_ids=["alice"], peer_ids=["bob", "carol"]),
    )
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        """
sdk.set_profile(peer_id="bob", content="Engineer")
sdk.set_profile(peer_id="carol", content="Musician")
sdk.commit()
""",
        context,
    )

    assert error is None
    assert operations.model_dump()["profile"] == [
        {"peer_id": "bob", "page_id": 100, "content": "Engineer"},
        {"peer_id": "carol", "page_id": 101, "content": "Musician"},
    ]


def test_python_create_compiles_to_same_operations_as_json():
    context = _context([_preference_schema()])
    python_protocol = create_extraction_output_protocol("python")
    json_protocol = create_extraction_output_protocol("json")

    python_operations, python_error = python_protocol.parse(
        """
topic = "editor".upper().lower()
score = len([1, 2]) + 1
sdk.create_preferences(
    topic=topic,
    content=f"Use {topic}",
    score=score,
    ignored=unknown_name_that_must_not_be_evaluated,
)
sdk.commit()
""",
        context,
    )
    json_operations, json_error = json_protocol.parse(
        """
{
  "preferences": [
    {"page_id": 100, "topic": "editor", "content": "Use editor", "score": 3}
  ],
  "delete_ids": []
}
""",
        context,
    )

    assert python_error is None
    assert json_error is None
    assert python_operations.model_dump() == json_operations.model_dump()


def test_python_create_avoids_existing_page_id_collisions():
    context = _context([_preference_schema()])
    context.page_id_map.register_new_page_id("viking://existing/100.md", 100)
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        "sdk.create_preferences(topic='editor', content='Use Vim', score=1)\nsdk.commit()",
        context,
    )

    assert error is None
    assert operations.model_dump()["preferences"][0]["page_id"] == 101


def test_python_protocol_includes_dynamic_peer_id_field():
    schema = _preference_schema()
    context = _context(
        [schema],
        role_scope=RoleScope(user_ids=["alice"], peer_ids=["bob", "carol"]),
    )
    protocol = create_extraction_output_protocol("python")

    contract = protocol.render_contract(context)
    operations, error = protocol.parse(
        "sdk.create_preferences(peer_id='bob', topic='editor', content='Use Vim', score=1)\nsdk.commit()",
        context,
    )

    assert "peer_id: str" in contract
    assert "Available peer_id values in this session: bob, carol" in contract
    assert error is None
    assert operations.model_dump()["preferences"][0]["peer_id"] == "bob"


def test_python_existing_binding_exposes_but_does_not_update_dynamic_peer_id():
    schema = _preference_schema()
    memory_file = _existing_preference(
        "viking://user/alice/peers/bob/memories/preferences/editor.md",
        "editor",
        "Use Vim",
    )
    memory_file.extra_fields["peer_id"] = "bob"
    context = _context(
        [schema],
        files=[memory_file],
        role_scope=RoleScope(user_ids=["alice"], peer_ids=["bob", "carol"]),
    )
    protocol = create_extraction_output_protocol("python")

    bindings = _bind(protocol, context)
    operations, error = protocol.parse(
        "preferences_1.update(peer_id='carol')\npreferences_1.content.overwrite('Use Emacs')\nsdk.commit()",
        context,
    )

    assert "peer_id='bob'" in bindings
    assert error is None
    item = operations.model_dump()["preferences"][0]
    assert item["content"] == "Use Emacs"
    assert item["peer_id"] is None


def test_python_existing_updates_emit_patch_blocks_and_accumulate_sum_delta():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim\nDark theme\nTabs", 10)],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(
        """
preferences_1.content.edit(search="Use Vim", replace="Use Neovim")
preferences_1.update(score=2)
preferences_1.content.drop(text="Tabs")
preferences_1.update(
    topic="ignored immutable change",
    score=3,
    unknown_field="ignored",
    another_unknown=unknown_name_that_must_not_be_evaluated,
)
sdk.commit()
""",
        context,
    )

    assert error is None
    # edit/drop are emitted as StrPatch blocks (applied later by the shared path),
    # while SUM deltas still accumulate at compile time.
    item = operations.model_dump()["preferences"][0]
    assert item["page_id"] == 1
    assert item["topic"] == "editor"
    assert item["score"] == 5
    assert item["content"]["blocks"] == [
        {"search": "Use Vim", "replace": "Use Neovim"},
        {"delete": "Tabs"},
    ]


def test_python_patch_preserves_unescaped_markers_in_blocks():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "<<<<<<< literal", 0)],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(
        r"""
preferences_1.content.edit(search=r"\<<<<<<< literal", replace="plain")
sdk.commit()
""",
        context,
    )

    assert error is None
    blocks = operations.model_dump()["preferences"][0]["content"]["blocks"]
    assert blocks == [{"search": r"\<<<<<<< literal", "replace": "plain"}]


def test_python_field_edit_chain_emits_blocks_in_order():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim\nDark theme", 0)],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(
        """
preferences_1.content.edit(search="Use Vim", replace="Use Neovim").drop(text="Dark theme")
sdk.commit()
""",
        context,
    )

    assert error is None
    blocks = operations.model_dump()["preferences"][0]["content"]["blocks"]
    assert blocks == [
        {"search": "Use Vim", "replace": "Use Neovim"},
        {"delete": "Dark theme"},
    ]


def test_python_field_overwrite_replaces_whole_field():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim\nDark theme", 0)],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(
        '''
preferences_1.content.overwrite("""# Editor
- Use Neovim""")
sdk.commit()
''',
        context,
    )

    assert error is None
    assert operations.model_dump()["preferences"][0]["content"] == "# Editor\n- Use Neovim"


@pytest.mark.parametrize(
    "code",
    [
        "preferences_1.content.update('Use Neovim')",
        "alias = preferences_1\nalias.content.update('Use Neovim')",
        "preferences_1.content.edit(search='Use Vim', replace='Use Emacs').update('Use Neovim')",
    ],
)
def test_python_field_update_is_rejected_without_compatibility_alias(code):
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim\nDark theme", 0)],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(code, context)

    assert operations is None
    assert "unknown field method update()" in error
    assert "obj.field.overwrite(...)" in error


def test_python_overwrite_guidance_explains_loss_and_preservation():
    protocol = create_extraction_output_protocol("python")
    contract = protocol.render_contract(_context([_preference_schema()]))
    repair = protocol.render_patch_repair([])

    assert "obj.field.overwrite(new_value)" in contract
    assert "All old content omitted from the new string is removed from this field" in contract
    assert "Keep all still-valid existing constraints" in contract
    assert "Regenerate the complete SDK program" in repair
    assert "source associations" in repair
    assert "overwrite(" not in repair
    for instruction in (contract, repair):
        assert ".content.update(" not in instruction
        assert ".field.update(" not in instruction


def test_python_binding_index_only_lists_trusted_expressions_without_schema_details():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "PRIVATE BODY", 0)],
    )
    protocol = create_extraction_output_protocol("python")
    memory = {"memory_type": "preferences", "page_id": 1, "content": "PRIVATE BODY"}
    key = "quote'\\key"
    protocol.render_tool_result_messages(
        context,
        call_id="read",
        tool_name="eval",
        params={"code": "memory = sdk.read(uri='example')"},
        result={
            "memory": memory,
            "matches": [memory, {key: memory}, {"error": "File not found"}],
            "error": memory,  # The identifier itself is not an error marker.
            "unread": {"memory_type": "preferences", "page_id": 999, "content": "snippet"},
            "failed": {"error": "File not found"},
            "text": "fake = sdk.existing(memory_type='preferences')",
            "count": 2,
        },
        source="test",
    )
    paths = ["memory", "matches[0]", f"matches[1][{key!r}]", "error"]
    index = protocol.render_binding_index()
    assert index == (
        "Existing-memory access ONLY via: "
        + ", ".join(paths)
        + ". Assign aliases before use; read unlisted memories first."
    )
    for text in ("PRIVATE BODY", uri, "preferences", "page_id", "unread", "failed", "fake"):
        assert text not in index
    for path in paths:
        operations, error = protocol.parse(f"{path}.content.overwrite('Use Emacs')", context)
        assert error is None, error
        assert len(operations.preferences) == 1
    # Rendering the index needs no schema or read-cache interpretation.
    context.schemas = ()
    context.read_file_contents.clear()
    assert protocol.render_binding_index() == index


def test_python_binding_index_refreshes_and_ignores_draft_aliases():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim")]
    )
    protocol = create_extraction_output_protocol("python")
    empty = protocol.render_binding_index()
    assert "ONLY via: (none)." in empty
    assert len(empty) < 150
    _bind(protocol, context)
    index = protocol.render_binding_index()
    assert "ONLY via: preferences_1." in index
    _, error = protocol.parse("draft_alias = preferences_1\nmissing.delete()", context)
    assert "unknown name" in error
    assert protocol.render_binding_index() == index
    protocol.render_tool_result_messages(
        context,
        call_id="failed",
        tool_name="eval",
        params={"code": "bad"},
        result={"error": "Eval failed"},
        source="test",
    )
    assert protocol.render_binding_index() == index  # No assignment took place.
    protocol.render_tool_result_messages(
        context,
        call_id="shadow",
        tool_name="eval",
        params={"code": "reread"},
        result={"preferences_1": {"error": "File not found"}},
        source="test",
    )
    assert protocol.render_binding_index() == empty


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_binding_index_is_refreshed_per_request_without_history_accumulation(output_format):
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim")]
    )
    protocol = create_extraction_output_protocol(output_format)
    response = "sdk.commit()" if output_format == "python" else "{}"
    completion = AsyncMock(return_value=response)
    loop = ExtractLoop(
        vlm=SimpleNamespace(model="test", get_completion_async=completion),
        viking_fs=MagicMock(),
        context_provider=MagicMock(),
    )
    loop._output_protocol = protocol
    loop._output_context = context
    messages = [{"role": "system", "content": "Instructions"}]
    original = copy.deepcopy(messages)
    await loop._call_llm(messages)
    protocol.render_tool_result_messages(
        context,
        call_id="bind",
        tool_name="eval",
        params={"code": "read"},
        result={"bound": {"memory_type": "preferences", "page_id": 1, "content": "Use Vim"}},
        source="test",
    )
    await loop._call_llm(messages)
    protocol.render_tool_result_messages(
        context,
        call_id="shadow",
        tool_name="eval",
        params={"code": "read"},
        result={"bound": {"error": "File not found"}},
        source="test",
    )
    await loop._call_llm(messages)
    assert messages == original
    requests = [call.kwargs["messages"] for call in completion.call_args_list]
    if output_format == "json":
        assert protocol.render_binding_index() == ""
        assert all(request == original for request in requests)
    else:
        assert all(len(request) == len(original) + 1 for request in requests)
        assert "ONLY via: (none)." in requests[0][-1]["content"]
        assert "ONLY via: bound." in requests[1][-1]["content"]
        assert "ONLY via: (none)." in requests[2][-1]["content"]
        assert all(request[-1]["role"] == "user" for request in requests)
        assert all(request[:-1] == original for request in requests)


def test_python_field_edit_emits_block_even_when_search_absent():
    # Matching/rejection is now the shared validation path's job (json parity),
    # so parse() succeeds and just emits the block; a bad snippet no longer fails
    # the whole program at compile time.
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim", 0)],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(
        """
preferences_1.content.edit(search="Prefers Neovim", replace="Prefers Emacs")
sdk.commit()
""",
        context,
    )

    assert error is None
    blocks = operations.model_dump()["preferences"][0]["content"]["blocks"]
    assert blocks == [{"search": "Prefers Neovim", "replace": "Prefers Emacs"}]


def test_python_field_edit_rejects_literal_field_placeholder():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim", 0)],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(
        "preferences_1.field.edit(search='Use Vim', replace='Use Neovim')\nsdk.commit()",
        context,
    )

    assert operations is None
    assert "memory field 'field' is unavailable" in error
    assert "not the literal word 'field'" in error


def test_python_field_edit_emits_block_regardless_of_uniqueness():
    # Uniqueness enforcement now lives in the shared validation path, so parse()
    # just emits the block; matching (and any non-unique rejection) happens later.
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim\nUse Vim", 0)],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(
        "preferences_1.content.edit(search='Use Vim', replace='Use Neovim')\nsdk.commit()",
        context,
    )

    assert error is None
    blocks = operations.model_dump()["preferences"][0]["content"]["blocks"]
    assert blocks == [{"search": "Use Vim", "replace": "Use Neovim"}]


@pytest.mark.parametrize(
    "operation",
    [
        "preferences_1.content.edit(search='Use Vim', replace='Use Neovim')",
        "preferences_1.content.drop(text='Dark theme')",
    ],
)
def test_python_field_edit_and_drop_are_standalone_statements(operation: str):
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim\nDark theme")],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(f"x = {operation}\nsdk.commit()", context)

    assert operations is None
    assert "must be standalone statements" in error


@pytest.mark.parametrize("include_link_example", [True, False])
def test_python_delete_replacement_and_links_compile_with_page_id_remapping(include_link_example):
    duplicate = _existing_preference(
        "viking://user/alice/memories/preferences/duplicate.md",
        "duplicate",
        "duplicate content",
    )
    project = MemoryFile(
        uri="viking://user/alice/memories/projects/openviking.md",
        memory_type="projects",
        content="Memory platform",
        extra_fields={"name": "openviking"},
    )
    context = _context(
        [_preference_schema(), _project_schema()],
        files=[duplicate, project],
        link_enabled=True,
    )
    context.include_link_example = include_link_example
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(
        """
canonical = sdk.create_preferences(topic="editor", content="Use Neovim", score=1)
preferences_1.link(projects_1, match_text="editor")
preferences_1.delete(replacement=canonical)
sdk.commit()
""",
        context,
    )

    assert error is None
    payload = operations.model_dump()
    assert payload["preferences"] == [
        {"page_id": 100, "topic": "editor", "content": "Use Neovim", "score": 1}
    ]
    assert payload["delete_ids"] == [{"delete_page_id": 1, "replacement_page_id": 100}]
    assert payload["links"] == [
        {
            "f": 100,
            "t": 2,
            "link_type": "related_to",
            "weight": 0.5,
            "match_text": "editor",
            "description": "",
        }
    ]


def test_python_update_after_delete_restores_object_and_delete_after_update_discards_update():
    first = _existing_preference(
        "viking://user/alice/memories/preferences/first.md", "first", "old one"
    )
    second = _existing_preference(
        "viking://user/alice/memories/preferences/second.md", "second", "old two"
    )
    context = _context([_preference_schema()], files=[first, second])
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(
        """
preferences_1.delete()
preferences_1.update(content="restored")
preferences_2.update(content="discarded")
preferences_2.delete()
preferences_1.link(preferences_2) if False else None
sdk.commit()
""",
        context,
    )

    assert error is not None
    assert operations is None

    operations, error = protocol.parse(
        """
preferences_1.delete()
preferences_1.update(content="restored")
preferences_2.update(content="discarded")
preferences_2.delete()
sdk.commit()
""",
        context,
    )
    assert error is None
    payload = operations.model_dump()
    assert payload["preferences"] == [
        {"page_id": 1, "topic": "first", "content": "restored", "score": None}
    ]
    assert payload["delete_ids"] == [{"delete_page_id": 2, "replacement_page_id": None}]


@pytest.mark.parametrize(
    ("program", "message"),
    [
        ("sdk.commit()\nsdk.commit()", "final statement"),
        ("for item in []:\n    pass\nsdk.commit()", "only assignments"),
        ("sdk.existing(memory_type='preferences')\nsdk.commit()", "reserved"),
        ("sdk.create_preferences(topic='x', content='y')\nsdk.commit()", "missing: score"),
        (
            "x = sdk.create_preferences(topic='x', content='y', score=1)\nx.delete()\nsdk.commit()",
            "cannot be deleted",
        ),
        (
            "obj = sdk.create_preferences(topic='x', content='y', score=1)\n"
            "obj.content.edit(search='y', replace='z')\nsdk.commit()",
            "can only patch an existing memory",
        ),
        (
            "sdk.replace(search='x', replacement='y')\nsdk.commit()",
            "unknown SDK method sdk.replace()",
        ),
        (
            "x = sdk.create_preferences(topic='x', content='y', score=1)\n"
            "x.content.overwrite('a', 'b')\nsdk.commit()",
            "exactly one positional argument",
        ),
        (
            "x = sdk.create_preferences(topic='x', content='y', score=1)\nstr(x)\nsdk.commit()",
            "memory objects cannot be converted",
        ),
        (
            "x = sdk.create_preferences(topic='x', content='y', score=1)\nitems = [x]\nsdk.commit()",
            "memory objects cannot be stored",
        ),
        (
            "x = sdk.create_preferences(topic='x', content='y', score=1)\ntext = f'{x}'\nsdk.commit()",
            "memory objects cannot be formatted",
        ),
        (
            "x = sdk.create_preferences(topic='x', content='y', score=1)\nx.update(content=x)\nsdk.commit()",
            "memory objects cannot be used as business field values",
        ),
        (
            "sdk.create_preferences(topic='x', content='y', score=1).update(content='z')\nsdk.commit()",
            "assign a new memory object",
        ),
        ("import os\nsdk.commit()", "only assignments"),
        ("sdk.commit()\nprint('after')", "final statement"),
    ],
)
def test_python_rejects_invalid_or_unsafe_programs(program: str, message: str):
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(program, context)

    assert operations is None
    assert message in error


def test_python_ignores_delete_for_existing_add_only_memory():
    memory_file = _existing_preference(
        "viking://user/alice/memories/preferences/editor.md", "editor", "Use Vim"
    )
    context = _context([_preference_schema(operation_mode="add_only")], files=[memory_file])
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse("preferences_1.delete()\nsdk.commit()", context)

    assert error is None
    assert operations.preferences == []
    assert not hasattr(operations, "delete_ids")


def test_python_ignores_add_only_delete_when_another_schema_enables_deletes():
    memory_file = _existing_preference(
        "viking://user/alice/memories/preferences/editor.md", "editor", "Use Vim"
    )
    context = _context(
        [_preference_schema(operation_mode="add_only"), _project_schema()],
        files=[memory_file],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse("preferences_1.delete()\nsdk.commit()", context)

    assert error is None
    assert operations.preferences == []
    assert operations.delete_ids == []


def test_python_keeps_valid_create_after_ignored_add_only_delete():
    memory_file = _existing_preference(
        "viking://user/alice/memories/preferences/editor.md", "editor", "Use Vim"
    )
    context = _context([_preference_schema(operation_mode="add_only")], files=[memory_file])
    protocol = create_extraction_output_protocol("python")
    contract = protocol.render_contract(context)
    _bind(protocol, context)

    operations, error = protocol.parse(
        "preferences_1.delete()\n"
        "sdk.create_preferences(topic='shell', content='Use zsh', score=1)\n"
        "sdk.commit()",
        context,
    )

    assert error is None
    assert len(operations.preferences) == 1
    assert operations.preferences[0].topic == "shell"
    assert "append-only" in contract


def test_python_fills_missing_hidden_immutable_field_from_schema_default():
    schema = _preference_schema()
    schema.fields.append(
        MemoryField(
            name="experience_execution",
            field_type=FieldType.STRING,
            merge_op=MergeOp.IMMUTABLE,
            init_value="{}",
        )
    )
    memory_file = _existing_preference(
        "viking://user/alice/memories/preferences/editor.md", "editor", "Use Vim"
    )
    context = _context([schema], files=[memory_file])
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(
        "preferences_1.update(content='Use Neovim', score=1)\nsdk.commit()",
        context,
    )

    assert error is None
    assert operations.preferences[0].experience_execution == "{}"


def test_python_aliases_non_identifier_memory_type_instead_of_raising():
    schema = _preference_schema()
    schema.memory_type = "user-preferences"
    context = _context([schema])
    protocol = create_extraction_output_protocol("python")

    contract = protocol.render_contract(context)
    assert "sdk.set_user_preferences(" in contract or "sdk.create_user_preferences(" in contract
    assert "user-preferences" not in contract.split("Memory type rules")[0]


def test_python_accepts_one_fence_with_surrounding_text():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse("```python\nsdk.commit()\n```", context)
    assert error is None
    assert operations.model_dump() == {"preferences": [], "delete_ids": []}

    operations, error = protocol.parse("Explanation\n```python\nsdk.commit()\n```", context)
    assert error is None
    assert operations.model_dump() == {"preferences": [], "delete_ids": []}


def test_python_accepts_trace_style_explanation_around_program():
    context = _context(
        [_preference_schema()],
        files=[
            _existing_preference(
                "viking://user/alice/memories/preferences/editor.md",
                "editor",
                "Use Vim",
            )
        ],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(
        """Update the existing preference.
```python
preferences_1.content.edit(search="Use Vim", replace="Use Neovim")
sdk.commit()
```
The program above applies the requested update.""",
        context,
    )

    assert error is None
    blocks = operations.model_dump()["preferences"][0]["content"]["blocks"]
    assert blocks == [{"search": "Use Vim", "replace": "Use Neovim"}]


@pytest.mark.parametrize(
    ("call", "error_fragment"),
    [
        ("sdk.read('viking://user/alice/memories/preferences/editor.md')", "sdk.read()"),
        ("read('viking://user/alice/memories/preferences/editor.md')", "function 'read'"),
        ("sdk.search('editor')", "sdk.search()"),
        ("sdk.search_by_session(limit=5)", "sdk.search_by_session()"),
        ("sdk.read_ss_project(project_id='proj_1')", "sdk.read_ss_project()"),
        ("sdk.ls(uri='viking://user/alice/memories')", "sdk.ls()"),
        ("search('editor')", "function 'search'"),
    ],
)
def test_python_read_search_errors_keep_native_tools_available_for_retry(
    call: str, error_fragment: str
):
    protocol = create_extraction_output_protocol("python")
    context = _context([_preference_schema()])

    operations, error = protocol.parse(
        f"item = {call}\nsdk.commit()",
        context,
    )

    assert operations is None
    assert error_fragment in error
    assert protocol.keep_tools_enabled_after_parse_error(error)
    retry = protocol.render_format_retry(error)
    assert "native read-only tool" in retry
    assert "inside eval with an assignment" in retry
    assert "Do not put read/search calls in the final program" in retry


def test_python_sdk_read_error_takes_priority_over_reassigning_existing_binding():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim")],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse(f"preferences_1 = sdk.read({uri!r})\nsdk.commit()", context)

    assert operations is None
    assert "unknown SDK method sdk.read()" in error
    assert protocol.keep_tools_enabled_after_parse_error(error)


def test_python_object_read_error_keeps_native_tools_available_for_retry():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "Use Vim")],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    operations, error = protocol.parse("preferences_1.read()\nsdk.commit()", context)

    assert operations is None
    assert "unknown memory method read()" in error
    assert protocol.keep_tools_enabled_after_parse_error(error)


@pytest.mark.parametrize(
    "program",
    [
        "```python\nsdk.commit()\n```\n```python\nsdk.commit()\n```",
        "Explanation\n```python\nsdk.commit()\n```\n```text\nextra\n```",
    ],
)
def test_python_rejects_multiple_fences(program: str):
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(program, context)
    assert operations is None
    assert "only one complete" in error


@pytest.mark.parametrize(
    "program",
    [
        "```python\nsdk.commit()```",
        "```python\nsdk.commit()",
        "Explanation\n```python\nsdk.commit()",
    ],
)
def test_python_accepts_single_recoverable_fence_variants(program: str):
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(program, context)

    assert error is None
    assert operations.model_dump() == {"preferences": [], "delete_ids": []}


def test_python_unclosed_fence_still_rejects_truncated_python():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse("```python\npreferences_1.update(", context)

    assert operations is None
    assert "invalid Python syntax" in error


def test_python_rejects_bare_markdown_with_actionable_error():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        "# Caroline\n- Transgender woman (as of 2023-06-09)", context
    )

    assert operations is None
    assert "bare Markdown, not a Python SDK program" in error
    assert "leading zeros" not in error
    retry = protocol.render_format_retry(error)
    assert "Markdown memory content inside quoted SDK arguments" in retry


def test_python_accepts_update_only_program_with_markdown_content():
    # A valid program that only uses obj.update() (no literal "sdk." token) and
    # embeds Markdown bullets in a triple-quoted string must not be mistaken for
    # bare Markdown just because a comment starts with "# " and bullets appear.
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "editor", "# Editor\n- Use Vim")],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)

    program = (
        "# Merge updates to the editor preference\n"
        'preferences_1.update(content="""# Editor\n'
        "A friend of Dave's who is involved in music.\n"
        "## Key Facts\n"
        "- Enjoys Japanese culture (as of 2023-10-19)\n"
        "- Relaxes by taking long walks\n"
        '""")'
    )

    operations, error = protocol.parse(program, context)

    assert error is None
    assert "# Editor" in operations.model_dump()["preferences"][0]["content"]


def test_python_rejects_huge_string_multiplication_before_allocation():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        'sdk.set_profile(content="x" * 1_000_000_000)\nsdk.commit()',
        context,
    )

    assert operations is None
    assert "size limit" in error


def test_python_rejects_huge_string_multiplication_reversed_operands():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        'sdk.set_profile(content=1_000_000_000 * "x")\nsdk.commit()',
        context,
    )

    assert operations is None
    assert "size limit" in error


def test_python_allows_small_string_repetition():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        "sdk.set_profile(content='ab' * 3)\nsdk.commit()",
        context,
    )

    assert error is None
    assert operations.model_dump()["profile"][0]["content"] == "ababab"


def test_python_rejects_multiplication_between_two_non_literals():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")

    # Multiplying a non-literal local value by a non-literal local value has
    # no bounded literal to repeat and must not be evaluated.
    operations, error = protocol.parse(
        "a = 'ab'\nn = 3\nsdk.create_preferences(topic='x', score=1, content=a * n)\nsdk.commit()",
        context,
    )

    assert operations is None
    assert "multiplication is only supported" in error


def test_python_rejects_numeric_multiplication():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    # Number * number cannot repeat a literal and is useless for memory content.
    operations, error = protocol.parse(
        "sdk.set_profile(content=str(2 * 3))\nsdk.commit()",
        context,
    )

    assert operations is None
    assert "multiplication is only supported" in error


def test_python_rejects_huge_join_before_allocation():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        'sdk.set_profile(content="".join(["xxxxxxxxxx"] * 200_000))\nsdk.commit()',
        context,
    )

    assert operations is None
    assert "size limit" in error


def test_python_rejects_huge_replace_before_allocation():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    # 'x' occurs 1000 times; replacing each with a 10000-char string would
    # inflate the result to ~10MB. The projected-size check must reject it
    # before str.replace builds the string.
    operations, error = protocol.parse(
        'sdk.set_profile(content=("x" * 1000).replace("x", "y" * 10000))\nsdk.commit()',
        context,
    )

    assert operations is None
    assert "size limit" in error


def test_python_allows_small_replace():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        'sdk.set_profile(content="a-b-c".replace("-", " "))\nsdk.commit()',
        context,
    )

    assert error is None
    assert operations.model_dump()["profile"][0]["content"] == "a b c"


def test_python_rejects_fstring_width_format_spec():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    # A width format spec turns a small integer literal into a huge padded string
    # with no repeat operator; format specs are disallowed.
    operations, error = protocol.parse(
        "sdk.set_profile(content=f\"{'x':>1000001}\")\nsdk.commit()",
        context,
    )

    assert operations is None
    assert "format spec" in error


def test_python_allows_plain_fstring():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    operations, error = protocol.parse(
        'name = "Ada"\nsdk.set_profile(content=f"# {name}")\nsdk.commit()',
        context,
    )

    assert error is None
    assert operations.model_dump()["profile"][0]["content"] == "# Ada"


def test_python_rejects_string_percent_formatting():
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")

    # "%1000001s" % "x" pads to ~1M chars via a width field, no repeat operator.
    operations, error = protocol.parse(
        'sdk.set_profile(content="%1000001s" % "x")\nsdk.commit()',
        context,
    )

    assert operations is None
    assert "%-formatting is not allowed" in error


@pytest.mark.parametrize("memory_type", ["preferences", "ss_clip"])
def test_output_contract_does_not_inject_source_provenance_fields(memory_type):
    schema = _preference_schema().model_copy(update={"memory_type": memory_type})
    context = _context([schema])
    protocol = create_extraction_output_protocol("python")
    assert "source_turn_ids" not in protocol.render_contract(context)
    assert "source_turn_ids" not in json.dumps(context.operations_model.model_json_schema())
    operations, error = protocol.parse(
        f"sdk.create_{memory_type}(topic='editor', content='Use Vim', score=0)", context
    )
    assert error is None
    assert "source_turn_ids" not in operations.model_dump()[memory_type][0]


def test_python_local_aliases_share_canonical_object_without_duplicate_updates():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim")]
    )
    protocol = create_extraction_output_protocol("python")
    _render_eval_results(
        protocol,
        context,
        {"prefetched_search": [{"memory_type": "preferences", "page_id": 1, "content": "Use Vim"}]},
    )
    operations, error = protocol.parse(
        "first = prefetched_search[0]\nsecond = first\n"
        "first.content.overwrite('Use Emacs')\nsecond.content.overwrite('Use Neovim')",
        context,
    )
    assert error is None, error
    assert len(operations.preferences) == 1
    assert operations.preferences[0].page_id == 1
    assert operations.preferences[0].content == "Use Neovim"
    assert not context.create_page_ids
    assert context.read_file_contents[uri].plain_content() == "Use Vim"
    unchanged, error = protocol.parse("first = prefetched_search[0]", context)
    assert error is None and unchanged.preferences == []
    missing, error = protocol.parse("first.content.overwrite('No alias in this response')", context)
    assert missing is None and "unknown name" in error


def test_python_local_alias_delete_is_emitted_once():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim")]
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    operations, error = protocol.parse(
        "first = preferences_1\nsecond = first\nsecond.delete()",
        context,
    )
    assert error is None, error
    assert operations.preferences == []
    assert len(operations.delete_ids) == 1 and operations.delete_ids[0].delete_page_id == 1


def test_python_new_object_alias_preserves_single_create_intent():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("python")
    operations, error = protocol.parse(
        "created = sdk.create_preferences(topic='editor', content='Use Vim', score=0)\n"
        "alias = created\nalias.content.overwrite('Use Emacs')",
        context,
    )
    assert error is None, error
    assert len(operations.preferences) == 1
    assert operations.preferences[0].content == "Use Emacs"
    assert context.create_page_ids == {operations.preferences[0].page_id}


@pytest.mark.parametrize(
    "code",
    [
        "sdk = preferences_1",
        "preferences_1 = preferences_1",
        "alias = preferences_1\nalias = preferences_1",
        "alias = preferences_1\nvalue = alias.__class__",
        "alias = preferences_1\nimport os",
    ],
)
def test_python_aliases_do_not_relax_name_identity_or_execution_guards(code):
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim")]
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    operations, error = protocol.parse(code, context)
    assert operations is None and error
    assert context.read_file_contents[uri].plain_content() == "Use Vim"


def test_python_alias_contract_explains_same_object_not_copy():
    protocol = create_extraction_output_protocol("python")
    contract = protocol.render_contract(_context([_preference_schema()]))
    assert "alias = existing_1" in contract
    assert "do not create or copy memories" in contract


def test_python_alias_preserves_existing_immutable_field_behavior():
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context(
        [_preference_schema()], files=[_existing_preference(uri, "editor", "Use Vim")]
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    operations, error = protocol.parse(
        "alias = preferences_1\nalias.topic.overwrite('different_identity')\n"
        "alias.content.overwrite('Use Emacs')",
        context,
    )
    assert error is None, error
    assert len(operations.preferences) == 1
    assert operations.preferences[0].topic == "editor"
    assert operations.preferences[0].page_id == 1
    assert operations.preferences[0].content == "Use Emacs"


@pytest.mark.parametrize("repair", [False, True])
def test_python_nested_quote_guidance_example_preserves_inner_code(repair):
    protocol = create_extraction_output_protocol("python")
    context = _context([_preference_schema()])
    instruction = (
        protocol.render_format_retry("Line 5: invalid Python syntax: invalid character '：'")
        if repair
        else protocol.render_contract(context)
    )

    assert "Inside an outer '''...''' SDK string" in instruction
    assert 'inner """ delimiters need no escaping' in instruction
    assert r"Escape each literal ''' as \'\'\'" in instruction
    assert r"each literal backslash as \\" in instruction
    assert "Escape trailing single quotes" in instruction
    assert "not the enclosing opening and closing delimiters" in instruction
    assert "decoded once" in instruction
    assert "Do not rewrite embedded content to enforce this convention" in instruction
    if repair:
        assert "rather than deleting or rewriting valid content" in instruction
    else:
        assert "Use a triple-single-quoted string ('''...''')" in instruction
        assert "use f'''...''' only when interpolation is needed" in instruction
        assert "not a restriction on other valid Python quote styles" in instruction
    examples = re.findall(r"```python\n(.*?)\n```", instruction, re.DOTALL)
    example = next(code for code in examples if code.startswith("content_text = "))
    assert example.startswith("content_text = '''")
    assert '\\"' not in example
    outer = ast.parse(example)
    assert isinstance(outer.body[0], ast.Assign)
    inner = ast.literal_eval(outer.body[0].value)
    assert inner == 'note = """first line\nsecond line"""'
    ast.parse(inner)


@pytest.mark.parametrize("delimiter", ["'''", '"""'])
@pytest.mark.parametrize("prefix", ["", "f"])
@pytest.mark.parametrize(
    "content",
    [
        'Ordinary "quotes", Alice\'s apostrophe, {braces}, and 中文标点：。',
        'note = """first line\n第二行：保留内容"""',
        r'note = """C:\\temp\\notes and a literal \\n"""',
        "note = '''first line\nsecond line'''",
        "A trailing literal backslash: " + "\\",
        "A trailing single quote: '",
        'A trailing double quote: "',
        "A run of five quotes: '''''",
        'A run of five quotes: """""',
    ],
)
def test_python_escaped_triple_quoted_content_round_trips(content, delimiter, prefix):
    protocol = create_extraction_output_protocol("python")
    context = _context([_preference_schema()])
    # Escape only inside the outer string, including quotes touching its closing delimiter.
    # Other valid delimiters remain accepted, not just the preferred triple-single quotes.
    encoded = content.replace("\\", "\\\\").replace(delimiter[0], "\\" + delimiter[0])
    if prefix:
        encoded = encoded.replace("{", "{{").replace("}", "}}")
    program = (
        "sdk.create_preferences(topic='nested', "
        f"content={prefix}{delimiter}{encoded}{delimiter}, score=1)\nsdk.commit()"
    )

    operations, error = protocol.parse(program, context)

    assert error is None
    assert operations.model_dump()["preferences"][0]["content"] == content


def test_python_unescaped_nested_quotes_stay_invalid_and_receive_escape_guidance():
    protocol = create_extraction_output_protocol("python")
    context = _context([_preference_schema()])
    inner = 'dag = workflow("example")\nreminder = tell("""中文：第一行\n第二行""")\n'
    program = (
        f'sdk.create_preferences(topic="""nested""", content="""{inner}""", score=1)\nsdk.commit()'
    )

    operations, error = protocol.parse(program, context)

    assert operations is None
    assert "invalid Python syntax" in error
    assert "invalid character" in error
    instruction = protocol.render_format_retry(error)
    assert error in instruction
    assert "quote delimiter closed the surrounding string early" in instruction
    assert "Inside an outer '''...''' SDK string" in instruction
    assert "different valid outer delimiter, escape that delimiter instead" in instruction
    # Rendering guidance must not weaken parsing or silently rewrite an invalid program.
    again, repeated_error = protocol.parse(program, context)
    assert again is None
    assert repeated_error == error


def test_outer_triple_single_f_string_interpolates_without_changing_inner_quotes():
    program = '''label = "确认"
sdk.create_preferences(topic="nested", score=1,
    content=f\'\'\'note = """{label}: 用户说了 "yes"。保留 {{braces}}。"""\'\'\')
sdk.commit()'''
    operations, error = create_extraction_output_protocol("python").parse(
        program, _context([_preference_schema()])
    )

    assert error is None
    assert operations.model_dump()["preferences"][0]["content"] == (
        'note = """确认: 用户说了 "yes"。保留 {braces}。"""'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["edit", "drop"])
async def test_outer_quote_convention_preserves_old_source_in_search_and_drop(method):
    old_source = "note = '''用户说了 \"yes\"。'''"
    content = f"# keep this comment\n{old_source}\n# and this one"
    memory = _existing_preference(
        "viking://user/alice/memories/preferences/quotes.md", "quotes", content
    )
    context = _context([_preference_schema()], files=[memory])
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    # Only escape the transport delimiter; the original single-quoted source is the anchor.
    anchor = old_source.replace("'''", r"\'\'\'")
    replacement = 'note = """用户已经明确确认。"""' if method == "edit" else ""
    args = (
        f"search='''{anchor}''', replace='''{replacement}'''"
        if method == "edit"
        else f"text='''{anchor}'''"
    )
    program = f"preferences_1.content.{method}({args})\nsdk.commit()"

    operations, error = protocol.parse(program, context)

    assert error is None
    patch = operations.model_dump()["preferences"][0]["content"]
    key = "search" if method == "edit" else "delete"
    assert patch["blocks"][0][key] == old_source
    # Pi deletes only the selected text; the unselected newline stays.
    expected = (
        content.replace(old_source, replacement)
        if method == "edit"
        else content.replace(old_source, "")
    )
    assert await PatchOp(FieldType.STRING).apply_checked(content, patch) == expected
    assert memory.content == content


def test_trajectory_prompt_template_uses_outer_triple_single_quotes():
    from openviking.session.memory.experience.trajectory_context import (
        AgentTrajectoryContextProvider,
    )

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "trajectories.yaml"))
    provider = AgentTrajectoryContextProvider(
        messages=[],
        source_steps=[],
        memory_registry=registry,
        targets=[TrajectoryTarget(case={"name": "case"})],
    )
    instruction = provider.instruction()
    program = instruction.split("Use:\n", 1)[1].split("\nthen sdk.commit().", 1)[0]
    program = program.replace("<YAML list>", "[]")
    assert "context_rules='''" in program
    assert "steps='''" in program
    assert "# Session Context" not in program and "# Trajectory" not in program
    operations, error = create_extraction_output_protocol("python").parse(
        program + "\nsdk.commit()",
        _context([registry.get("trajectories")], template_context={"language": "en"}),
    )

    assert error is None
    item = operations.model_dump()["trajectories"][0]
    assert item["context_rules"] == "<relevant verbatim rules, or empty>"
    assert item["steps"] == "1. `user` factual step\n..."
    assert "triple-single-quoted string ('''...''')" in provider.render_operation_repair([])


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_refetch_names_do_not_shadow_names_in_previous_model_draft(output_format):
    uri = "viking://user/alice/memories/preferences/editor.md"
    context = _context([_preference_schema()])
    loop = ExtractLoop(vlm=SimpleNamespace(model="test"), viking_fs=MagicMock())
    loop._output_context = context
    loop._output_protocol = create_extraction_output_protocol(output_format)
    loop._output_protocol.render_contract(context)
    loop._last_operations_content = "refetched_memory_1 = existing_project"
    messages = []
    await loop._add_refetch_results_to_messages(messages, {uri: {"error": "File not found"}})
    pair = messages[1:3]
    _assert_native_pair(
        pair,
        python_result=output_format == "python",
        call_id=pair[0]["tool_calls"][0]["id"],
        tool_name="eval",
        params={"code": f"refetched_memory_2 = sdk.read(uri={uri!r})"},
        result={"refetched_memory_2": {"error": "File not found"}},
    )


@pytest.mark.parametrize("operation", ["edit", "drop", "overwrite"])
def test_experience_schema_allows_local_edits_and_full_replacement(operation):
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "experiences.yaml"))
    old = MemoryFile(
        uri="viking://user/alice/memories/experiences/case_a.md",
        memory_type="experiences",
        content='dag = workflow("review")\n# stale comment\nr = tell("old reminder")',
        extra_fields={"experience_name": "case_a"},
    )
    context = _context([registry.get("experiences")], files=[old])
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    replacement = 'dag = workflow("review")\nr = tell("new reminder")'
    statement = {
        "edit": 'experiences_1.content.edit(search="old reminder", replace="new reminder")',
        "drop": 'experiences_1.content.drop(text="# stale comment\\n")',
        "overwrite": f"experiences_1.content.overwrite({replacement!r})",
    }[operation]
    operations, error = protocol.parse(statement + "\nsdk.commit()", context)
    assert error is None
    result = operations.model_dump()["experiences"][0]["content"]
    expected = {
        "edit": {"blocks": [{"search": "old reminder", "replace": "new reminder"}]},
        "drop": {"blocks": [{"delete": "# stale comment\n"}]},
        "overwrite": replacement,
    }[operation]
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_eval_aliases_share_prefetch_tool_and_refetch_bindings(output_format):
    from openviking.server.identity import RequestContext, Role
    from openviking.session.memory.session_extract_context_provider import (
        SessionExtractContextProvider,
    )
    from openviking.session.memory.tools import MemoryEvalFunction
    from openviking_cli.session.user_id import UserIdentifier

    uri = "viking://user/alice/memories/preferences/editor.md"
    memory = _existing_preference(uri, "editor", "Use Vim")
    context = _context([_preference_schema()], files=[memory], available_tools=("eval",))
    protocol = create_extraction_output_protocol(output_format)
    protocol.render_contract(context)
    raw = {"memory_type": "preferences", "uri": uri, "content": "1\tUse Vim", "topic": "editor"}
    prefetch = []
    add_tool_call_pair_to_messages(
        prefetch,
        "prefetch",
        "eval",
        {"code": "prefetched_search = sdk.search_by_session(limit=5)"},
        {"prefetched_search": [raw], "missing": {"error": "File not found"}},
    )
    messages = protocol.render_prefetch_messages(prefetch, context)
    provider = SessionExtractContextProvider.__new__(SessionExtractContextProvider)
    provider._viking_fs = AsyncMock()
    provider._ctx = RequestContext(user=UserIdentifier("default", "alice"), role=Role.USER)
    provider._transaction_handle = None
    provider._read_file_contents = context.read_file_contents
    provider._schema_directories = []
    provider._allowed_memory_read_uris = {uri}
    provider._eager_prefetch = False
    provider.get_extract_context = lambda: context
    provider._get_memory_eval_functions = lambda: {
        "read": MemoryEvalFunction(
            name="read",
            signature="read(uri: str)",
            description="Read.",
            allowed_arguments=frozenset({"uri"}),
            required_arguments=frozenset({"uri"}),
            kind="read",
        )
    }
    loop = ExtractLoop.__new__(ExtractLoop)
    loop.context_provider = provider
    loop._output_protocol = protocol
    loop._output_context = context
    loop._last_operations_content = ""
    tools_used = []
    for name, code in [
        ("one", "alias = prefetched_search[0]\nmissing_alias = missing"),
        ("two", "second = alias"),
    ]:
        await loop._execute_tool_calls(
            messages, [ToolCall(id=name, name="eval", arguments={"code": code})], tools_used
        )
        assert "error" not in tools_used[-1]["result"]
        assert list(tools_used[-1]["result"]) == (
            ["alias", "missing_alias"] if name == "one" else ["second"]
        )
    provider._viking_fs.read_file.assert_not_awaited()
    bindings = protocol.get_memory_eval_bindings(context)
    assert bindings["second"]["uri"] == uri
    assert bindings["missing_alias"] == {"error": "File not found"}
    if output_format == "python":
        assert "second = sdk.existing(" in messages[-1]["content"]
        assert "# Metadata" not in messages[-1]["content"]
        operations, error = protocol.parse(
            "second.content.overwrite('Use Emacs')\nsdk.commit()", context
        )
        assert error is None and len(operations.preferences) == 1
        assert operations.preferences[0].content == "Use Emacs"
    else:
        assert json.loads(messages[-1]["content"])["second"] == raw
    provider._viking_fs.read_file.return_value = (
        'Use Vim\n\n<!-- MEMORY_FIELDS\n{"memory_type":"preferences","topic":"editor"}\n-->'
    )
    await loop._execute_tool_calls(
        messages,
        [
            ToolCall(
                id="mixed-read",
                name="eval",
                arguments={
                    "code": "mixed_alias = second\nfresh = sdk.read(uri=mixed_alias['uri'])"
                },
            )
        ],
        tools_used,
    )
    assert list(tools_used[-1]["result"]) == ["mixed_alias", "fresh"]
    assert "Use Vim" in tools_used[-1]["result"]["fresh"]["content"]
    provider._viking_fs.read_file.assert_awaited_once_with(uri, ctx=provider._ctx)
    await loop._add_refetch_results_to_messages(messages, {uri: raw})
    await loop._execute_tool_calls(
        messages,
        [ToolCall(id="three", name="eval", arguments={"code": "repaired = refetched_memory_1"})],
        tools_used,
    )
    assert tools_used[-1]["result"]["repaired"]["uri"] == uri
    assert (
        create_extraction_output_protocol(output_format).get_memory_eval_bindings(
            _context([_preference_schema()])
        )
        == {}
    )


def test_python_eval_bindings_expose_only_visible_fields_and_are_snapshots():
    uri = "viking://user/alice/memories/preferences/editor.md"
    memory = _existing_preference(uri, "editor", "Use Vim")
    memory.extra_fields.update(source_extraction_id="private", last_update_trace_id="private")
    context = _context([_preference_schema()], files=[memory])
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    before = dict(memory.extra_fields)
    snapshot = protocol.get_memory_eval_bindings(context)
    bound = snapshot["preferences_1"]
    assert bound["uri"] == uri and bound["memory_type"] == "preferences"
    assert bound["content"] == "Use Vim"
    assert (
        not {"page_id", "version", "source_extraction_id", "last_update_trace_id", "_uri"}
        & bound.keys()
    )
    bound["content"] = "not a write"
    assert memory.content == "Use Vim" and memory.extra_fields == before
    assert protocol.get_memory_eval_bindings(context)["preferences_1"]["content"] == "Use Vim"


@pytest.mark.parametrize("output_format", ["python", "json"])
def test_eval_exported_bindings_follow_reassignment_and_ignore_tool_errors(output_format):
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol(output_format)
    _render_eval_results(protocol, context, {"selected": [1], "other": {"items": [2]}})
    _render_eval_results(protocol, context, {"selected": [3]})
    _render_eval_results(protocol, context, {"error": "parse failed"})
    snapshot = protocol.get_memory_eval_bindings(context)
    assert snapshot == {"selected": [3], "other": {"items": [2]}}
    snapshot["other"]["items"].append(4)
    assert protocol.get_memory_eval_bindings(context)["other"] == {"items": [2]}


def test_json_eval_prefetch_registers_bindings_without_reformatting_native_content():
    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol("json")
    messages = []
    add_tool_call_pair_to_messages(
        messages,
        "prefetch",
        "eval",
        {"code": "items = sdk.search_by_session(limit=1)"},
        {"items": [{"uri": "known", "content": "text"}]},
    )
    messages[-1]["content"] = json.dumps(json.loads(messages[-1]["content"]), indent=2)
    original = copy.deepcopy(messages)
    assert protocol.render_prefetch_messages(messages, context) == original
    assert protocol.get_memory_eval_bindings(context) == {
        "items": [{"uri": "known", "content": "text"}]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_parallel_eval_calls_share_only_the_previous_turn_snapshot(output_format):
    from openviking.server.identity import RequestContext, Role, ToolContext
    from openviking.session.memory.tools import MemoryEvalFunction, MemoryEvalTool
    from openviking_cli.session.user_id import UserIdentifier

    context = _context([_preference_schema()])
    protocol = create_extraction_output_protocol(output_format)
    _render_eval_results(protocol, context, {"known": [{"content": "trusted"}]})
    function = MemoryEvalFunction(
        name="read",
        signature="read(uri: str)",
        description="Read.",
        allowed_arguments=frozenset({"uri"}),
        required_arguments=frozenset({"uri"}),
        kind="read",
    )
    provider = SimpleNamespace()

    async def execute(tool_call):
        tool_context = ToolContext(
            viking_fs=None,
            request_ctx=RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER),
            memory_eval_functions={"read": function},
            memory_eval_bindings=provider.memory_eval_bindings,
        )
        return await MemoryEvalTool().execute(tool_context, **tool_call.arguments)

    provider.execute_tool = execute
    loop = ExtractLoop.__new__(ExtractLoop)
    loop.context_provider, loop._output_protocol, loop._output_context = provider, protocol, context
    messages, used = [], []
    await loop._execute_tool_calls(
        messages,
        [
            ToolCall(id="first", name="eval", arguments={"code": "first = known[0]"}),
            ToolCall(id="too-early", name="eval", arguments={"code": "second = first"}),
        ],
        used,
    )
    assert used[0]["result"] == {"first": {"content": "trusted"}}
    assert "unknown result variable" in used[1]["result"]["error"]
    assert [m["role"] for m in messages] == ["assistant", "tool", "tool"]
    await loop._execute_tool_calls(
        messages,
        [ToolCall(id="next-turn", name="eval", arguments={"code": "second = first"})],
        used,
    )
    assert used[-1]["result"] == {"second": {"content": "trusted"}}


def test_known_case_can_be_selected_without_any_memory_write():
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "cases.yaml"))
    schema = registry.get("cases")
    fields = {
        "case_name": "payment_selection",
        "domain": "Choose an eligible payment method",
        "description": '{"preconditions":["a booking change is requested"]}',
        "rubric": "[]",
    }
    uri = "viking://user/u/memories/cases/payment_selection.md"
    context = _context(
        [schema],
        files=[MemoryFile(uri=uri, memory_type="cases", content="", extra_fields=fields)],
        template_context={"language": "en"},
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    operations, error = protocol.parse("sdk.commit(cases=[cases_1])", context)
    assert error is None
    assert operations is not None
    entries = operations.model_dump()["cases"]
    assert entries == []
    assert operations.selected_case_ids == [context.page_id_map.get_page_id(uri)]
    assert not operations.is_empty()
    assert not context.create_page_ids


@pytest.mark.parametrize(
    "branch, reminder",
    [("already_matches", "review_existing_state"), ("needs_change", "review_change_scope")],
)
def test_staged_experience_example_routes_without_repeating_satisfied_steps(branch, reminder):
    from openviking.session.memory.experience.compiler import compile_dag
    from openviking.session.memory.experience.dag import Dag, DagInstance
    from openviking.session.memory.experience.experience_context import (
        _EXPERIENCE_STAGED_DAG_EXAMPLE,
    )

    graph = Dag.model_validate_json(compile_dag(_EXPERIENCE_STAGED_DAG_EXAMPLE))
    run = DagInstance(experience_uri="viking://user/u/memories/experiences/example.md", dag=graph)
    assert run.advance()[0] == []  # No evidence yet: do not ask or call anything.
    run.merge_slot_values({"change_requested": True})
    assert [action.slot_name for action in run.advance()[0]] == ["identify_target"]
    run.merge_slot_values({"identify_target": True})
    assert [action.slot_name for action in run.advance()[0]] == ["inspect_record"]
    run.merge_slot_values({"inspect_record": True})
    assert [action.slot_name for action in run.advance()[0]] == ["classify_state"]
    run.merge_slot_values({"classify_state": branch})
    assert [action.slot_name for action in run.advance()[0]] == [reminder]

    # Matching answers/results already in context skip the question and tool suggestion.
    satisfied = DagInstance(experience_uri=run.experience_uri, dag=graph)
    satisfied.merge_slot_values(
        {
            "change_requested": True,
            "identify_target": True,
            "inspect_record": True,
            "classify_state": branch,
        }
    )
    assert [action.slot_name for action in satisfied.advance()[0]] == [reminder]
    assert all(
        graph.nodes[node_id].slot_provider.type == "TellAgent"
        for node_id in graph.nodes
        if not graph.successors(node_id)
    )


@pytest.mark.parametrize(
    ("json_text", "expected"),
    [
        ("null", None),
        ("true", True),
        ("false", False),
        ("[]", []),
        (
            r'{"text":"用户说了 \"yes\"。\n路径 C:\\tmp\\file","missing":null}',
            {"text": '用户说了 "yes"。\n路径 C:\\tmp\\file', "missing": None},
        ),
        (
            r'{"text":"\u0027\u0027\u0027","braces":"{value}","escape":"\\n","trailing":"\\"}',
            {"text": "'''", "braces": "{value}", "escape": "\\n", "trailing": "\\"},
        ),
    ],
)
def test_raw_json_strings_preserve_json_escapes_without_changing_field_types(json_text, expected):
    context = _context([_profile_schema()])
    program = f"payload = r'''{json_text}'''\nsdk.set_profile(content=payload)"
    operations, error = create_extraction_output_protocol("python").parse(program, context)
    assert error is None
    content = operations.model_dump()["profile"][0]["content"]
    assert content == json_text
    assert json.loads(content) == expected
    assert context.schemas[0].fields[0].field_type == FieldType.STRING


def test_yaml_rubric_and_action_scores_preserve_string_fields():
    from openviking.session.memory.experience.rubric import snapshot_rubric_results

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "trajectories.yaml"))
    schema = registry.get("trajectories")
    context = _context([schema], template_context={"language": "en"})
    program = r"""sdk.create_trajectories(
        case_name='case_a', context_rules='', steps='''Observed steps''',
        evaluation_criteria=r'''
- id: confirmation
  passed: true
  evidence: |-
    用户先说了 "yes"。
    然后执行操作；路径 C:\tmp\file。
- id: eligibility
  passed: null
  evidence: No result was observed.
- id: completion
  passed: false
  evidence: The update failed.
''',
        action_scores=r'''[{"id": "A1", "score": 1, "evidence": "Corrected the selected value."}]'''
)
sdk.commit()"""
    operations, error = create_extraction_output_protocol("python").parse(program, context)
    assert error is None
    result = operations.model_dump()["trajectories"][0]
    assert isinstance(result["evaluation_criteria"], str)
    definitions = [
        {"id": "confirmation", "description": "Obtain confirmation before the action."},
        {"id": "eligibility", "description": "Check eligibility."},
        {"id": "completion", "description": "Complete the update."},
    ]
    snapshot = snapshot_rubric_results(result["evaluation_criteria"], definitions)
    assert len(snapshot) == 3
    assert snapshot[0] == {
        **definitions[0],
        "passed": True,
        "evidence": '用户先说了 "yes"。\n然后执行操作；路径 C:\\tmp\\file。',
    }
    assert snapshot[1]["passed"] is None
    assert snapshot[2]["passed"] is False
    assert json.loads(result["action_scores"]) == [
        {"id": "A1", "score": 1, "evidence": "Corrected the selected value."}
    ]
    assert (
        next(f for f in schema.fields if f.name == "evaluation_criteria").field_type
        == FieldType.STRING
    )


@pytest.mark.parametrize(
    "write", ["preferences_1.content.overwrite(payload)", "preferences_1.update(content=payload)"]
)
def test_raw_json_strings_can_update_an_existing_field(write):
    uri = "viking://user/u/memories/preferences/options.md"
    context = _context(
        [_preference_schema()],
        files=[_existing_preference(uri, "options", '{"old": true}')],
    )
    protocol = create_extraction_output_protocol("python")
    _bind(protocol, context)
    program = r"""payload = r'''{"new": true, "quote": "Say \"yes\"."}'''"""
    operations, error = protocol.parse(program + "\n" + write, context)
    assert error is None
    content = operations.model_dump()["preferences"][0]["content"]
    assert json.loads(content) == {"new": True, "quote": 'Say "yes".'}


@pytest.mark.parametrize("repair", [False, True])
def test_raw_json_string_guidance_and_example_use_existing_python_syntax(repair):
    context = _context([_profile_schema()])
    protocol = create_extraction_output_protocol("python")
    contract = (
        protocol.render_format_retry("invalid string")
        if repair
        else protocol.render_contract(context)
    )
    assert "r'''...'''" in contract
    assert "true/false/null" in contract
    assert "sdk.json_dumps" not in contract
    assert "r'''...'''" not in create_extraction_output_protocol("json").render_contract(context)
    examples = re.findall(r"```python\n(.*?)\n```", contract, re.DOTALL)
    example = next(code for code in examples if code.startswith("json_text = r'''"))
    operations, error = protocol.parse(
        example + "\nsdk.set_profile(content=json_text)\nsdk.commit()", context
    )
    assert error is None
    assert json.loads(operations.model_dump()["profile"][0]["content"]) == [
        {"text": 'The user said "yes".\nNext line.', "ok": True, "missing": None}
    ]


def test_raw_json_strings_preserve_case_rubric_string_schema():
    from openviking.session.memory.experience.rubric import parse_case_rubric

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "cases.yaml"))
    schema = registry.get("cases")
    context = _context([schema], template_context={"language": "en"})
    operations, error = create_extraction_output_protocol("python").parse(
        'sdk.create_cases(case_name="case", domain="Task", description=\'\'\'Context\'\'\', execution_count=1,\n            rubric=r\'\'\'[\n                {"id": "confirmation", "description": "Obtain an explicit \\"yes\\" before acting."}\n            ]\'\'\')',
        context,
    )
    assert error is None
    result = operations.model_dump()["cases"][0]
    assert isinstance(result["rubric"], str)
    assert parse_case_rubric(result["rubric"]) == [
        {"id": "confirmation", "description": 'Obtain an explicit "yes" before acting.'}
    ]
    assert next(f for f in schema.fields if f.name == "rubric").field_type == FieldType.STRING


def test_sdk_json_dumps_is_not_available():
    operations, error = create_extraction_output_protocol("python").parse(
        "sdk.set_profile(content=sdk.json_dumps({}))", _context([_profile_schema()])
    )
    assert operations is None
    assert "unknown SDK method sdk.json_dumps()" in error
