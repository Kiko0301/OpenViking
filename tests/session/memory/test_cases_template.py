from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from openviking.session.compressor_v3 import _memory_file_to_case
from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.extraction_output_protocol import (
    ExtractionOutputContext,
    create_extraction_output_protocol,
)
from openviking.session.memory.memory_type_registry import (
    MemoryTypeRegistry,
    resolve_memory_templates_dir,
)
from openviking.session.memory.merge_op import MergeOp
from openviking.session.memory.page_id_map import PageIdMap
from openviking.session.memory.schema_model_generator import SchemaModelGenerator
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.memory.utils.uri import generate_uri


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize("domain_name", ["airline_customer_service", "library_services"])
def test_case_schema_keeps_identity_and_allows_domain_content_updates(language, domain_name):
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "cases.yaml"))
    schema = registry.get("cases")
    generator = SchemaModelGenerator([schema], template_context={"language": language})
    model = generator.create_flat_data_model(schema)
    operations = generator.create_structured_operations_model()
    description = operations.model_fields["cases"].description
    assert "business domain" in description
    assert "Infer the business domain from the current service/task context" in description
    assert "one Case: airline_customer_service" not in description
    assert "Do not split a domain" in description
    assert "Never repurpose an existing Case into a different business domain" in description
    assert "Keep case_name unchanged" in description
    assert "matched_case.update(execution_count=1)" in description
    assert "selected_case_ids" in operations.model_fields
    assert "Do not group tasks merely because they share a broad domain" not in description
    assert "{{ language }}" not in description
    name_description = model.model_fields["case_name"].description
    assert "regardless of conversation language" in name_description
    assert "Never name a Case after a workflow" in name_description
    assert {field.name for field in schema.fields} == {
        "case_name",
        "domain",
        "description",
        "rubric",
        "execution_count",
    }
    assert {field.name: field.merge_op for field in schema.fields} == {
        "case_name": MergeOp.IMMUTABLE,
        "domain": MergeOp.PATCH,
        "description": MergeOp.PATCH,
        "rubric": MergeOp.PATCH,
        "execution_count": MergeOp.SUM,
    }
    counter = next(field for field in schema.fields if field.name == "execution_count")
    assert counter.field_type.value == "int64"
    fields = {
        "case_name": domain_name,
        "domain": f"Service scope for {domain_name}.",
        "description": f"Requests in {domain_name} use the current policy and supplied records.",
        "rubric": '[{"id":"confirm","description":"Confirm changes before execution."}]',
    }
    parsed = model.model_validate({"page_id": 100, **fields})
    assert parsed.case_name == fields["case_name"]
    assert generate_uri(schema, fields, user_space="default") == (
        f"viking://user/default/memories/cases/{domain_name}.md"
    )


@pytest.mark.parametrize("output_format", ["python", "json"])
@pytest.mark.parametrize(
    "criteria",
    [
        [],
        [
            {"id": "scope", "description": "不超出读者授权的范围。"},
            {"id": "renewal", "description": "仅在用户要求续借且剩余借期不超过3天时办理。"},
        ],
    ],
    ids=["no-explicit-criteria", "explicit-conditional-criteria"],
)
def test_case_rubric_round_trips_through_protocol_template_and_learning(output_format, criteria):
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "cases.yaml"))
    schema = registry.get("cases")
    context = ExtractionOutputContext(
        operations_model=SchemaModelGenerator([schema]).create_structured_operations_model(),
        schemas=(schema,),
        page_id_map=PageIdMap(),
        read_file_contents={},
        link_enabled=False,
    )
    fields = {
        "case_name": "library_services",
        "execution_count": 1,
        "domain": "办理图书借还、续借和预约请求。",
        "description": "基于当前借阅规则和读者提供的信息处理请求。\n保留工具字段的原始值。",
        "rubric": json.dumps(criteria, ensure_ascii=False),
    }
    protocol = create_extraction_output_protocol(output_format)
    contract = protocol.render_contract(context)
    assert "A short Case introduction" in contract
    assert "JSON list of domain-wide evaluation criteria" in contract
    for restriction in (
        "Extract only explicit external rubrics",
        "evaluation/acceptance criteria defined in context",
        "success conditions explicitly stated by the user",
        "Do not infer criteria from ordinary business rules",
        "Existing memories alone are not authority",
        "without changing meaning, conditions, thresholds or scope",
        "leave an existing rubric unchanged",
        "use [] for a new Case",
    ):
        assert restriction in contract
    assert "grounded in supplied rules, task requirements and" not in contract
    assert '"required":true' not in contract
    if output_format == "python":
        program = (
            "sdk.create_cases("
            + ", ".join(f"{name}={value!r}" for name, value in fields.items())
            + ")\nsdk.commit()"
        )
    else:
        program = json.dumps({"cases": [{"page_id": 100, **fields}]}, ensure_ascii=False)
    operations, error = protocol.parse(program, context)
    assert error is None
    item = operations.cases[0]
    assert item.execution_count == 1
    assert item.description == fields["description"] and item.rubric == fields["rubric"]
    uri = generate_uri(schema, fields, user_space="default")
    raw = MemoryFileUtils.write(
        MemoryFile(
            uri=uri,
            memory_type="cases",
            content="STALE_MODEL_WRITTEN_BODY",
            extra_fields=fields,
        ),
        content_template=schema.content_template,
    )
    memory = MemoryFileUtils.read(raw, uri=uri)
    assert "STALE_MODEL_WRITTEN_BODY" not in memory.content
    assert f"## Description\n{fields['description']}" in memory.content
    assert f"## Rubric\n```json\n{fields['rubric']}" in memory.content
    assert memory.extra_fields["description"] == fields["description"]
    assert memory.extra_fields["rubric"] == fields["rubric"]
    assert memory.extra_fields["execution_count"] == 1
    assert "execution_count" not in memory.content
    case = _memory_file_to_case(memory)
    assert case.task_signature == fields["domain"]
    assert case.input == {"summary": fields["description"]}
    assert case.rubric.description == ""
    assert [
        {"id": c.name, "description": c.description} for c in case.rubric.criteria
    ] == json.loads(fields["rubric"])
    serialized = asdict(case)
    assert len(serialized["rubric"]["criteria"]) == len(criteria)
    assert serialized["metadata"]["memory_fields"]["rubric"] == fields["rubric"]


def test_missing_rubric_does_not_invent_default_criteria():
    case = _memory_file_to_case(
        MemoryFile(
            uri="viking://user/default/memories/cases/schema_authoring.md",
            memory_type="cases",
            content="",
            extra_fields={"case_name": "schema_authoring"},
        )
    )
    assert case.rubric.description == ""
    assert case.rubric.criteria == []


@pytest.mark.parametrize("legacy_field", ["task_signature", "input", "domain_scope"])
def test_old_case_memory_fields_require_explicit_migration(legacy_field):
    memory = MemoryFile(
        uri="viking://user/default/memories/cases/airline.md",
        memory_type="cases",
        content="Original content must not be silently rewritten",
        extra_fields={"case_name": "airline", legacy_field: "Original domain information"},
    )
    before = memory.model_dump()
    with pytest.raises(ValueError, match="uses removed fields") as caught:
        _memory_file_to_case(memory)
    assert "task_signature or domain_scope to domain" in str(caught.value)
    assert "input to description" in str(caught.value)
    assert memory.model_dump() == before


def test_old_case_create_keywords_are_not_accepted_as_aliases():
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "cases.yaml"))
    schema = registry.get("cases")
    context = ExtractionOutputContext(
        operations_model=SchemaModelGenerator([schema]).create_structured_operations_model(),
        schemas=(schema,),
        page_id_map=PageIdMap(),
        read_file_contents={},
        link_enabled=False,
    )
    protocol = create_extraction_output_protocol("python")
    operations, error = protocol.parse(
        "sdk.create_cases(case_name='airline', task_signature='scope', input='context', "
        "execution_count=1, rubric='[]')",
        context,
    )
    assert operations is None
    assert "domain" in error and "description" in error
    assert "requires complete memory fields" in error


def test_benchmark_case_spec_keeps_execution_input_and_task_signature():
    from openviking.session.train import Case, Rollout, Rubric
    from openviking.session.train.components.session_commit import _case_spec_payload

    task_input = {"domain": "airline", "data_split": "train", "task_no": 5, "task_id": "7"}
    case = Case("task_a", "tau2:airline:train:7", task_input, Rubric("reward", "", []))
    payload = _case_spec_payload(Rollout(case, [], "snapshot"))["case"]
    assert payload["task_signature"] == case.task_signature
    assert payload["input"] == task_input
    assert "domain" not in payload and "description" not in payload


def test_experience_case_header_uses_domain_names_without_leaking_task_control_data():
    from openviking.session.memory.experience.experience_context import (
        _trajectory_comparison_messages,
    )
    from openviking.session.train import Case, Rubric

    case = Case(
        "airline",
        "Airline service scope",
        {"summary": "Stable background", "task_id": "PRIVATE_TASK"},
        Rubric("reward", "", []),
    )
    original = asdict(case)
    messages = _trajectory_comparison_messages(
        {
            "case": original,
            "successful_trajectories": [],
            "failed_trajectories": [],
            "other_trajectories": [],
        }
    )
    text = messages[0]["content"]
    assert "domain: Airline service scope" in text
    assert "description: Stable background" in text
    assert "task_signature:" not in text and "input:" not in text
    assert "PRIVATE_TASK" not in text
    assert asdict(case) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
@pytest.mark.parametrize("edit", ["patch", "overwrite"])
async def test_case_domain_and_description_support_updates(output_format, edit):
    from openviking.session.memory.merge_op import FieldType, PatchOp
    from tests.session.memory.test_extraction_output_protocol import _bind, _context

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "cases.yaml"))
    schema = registry.get("cases")
    fields = {
        "case_name": "library",
        "domain": "Library borrowing services.",
        "description": "Use library records.\nKeep authorization constraints.",
        "rubric": "[]",
        "execution_count": 3,
    }
    updated = {
        "domain": "Library borrowing and renewal services.",
        "description": "Use current library records.\nKeep authorization constraints.",
    }
    memory = MemoryFile(
        uri="viking://user/default/memories/cases/library.md",
        memory_type="cases",
        content="",
        extra_fields=dict(fields),
    )
    context = _context([schema], files=[memory])
    protocol = create_extraction_output_protocol(output_format)
    _bind(protocol, context)
    if output_format == "python":
        calls = [
            f"cases_1.{name}.edit(search={fields[name]!r}, replace={value!r})"
            if edit == "patch"
            else f"cases_1.{name}.overwrite({value!r})"
            for name, value in updated.items()
        ]
        program = "\n".join([*calls, "cases_1.update(execution_count=1)", "sdk.commit()"])
    else:
        program = json.dumps(
            {
                "cases": [
                    {
                        "page_id": 1,
                        "case_name": "library",
                        "execution_count": 1,
                        **{
                            name: {"blocks": [{"search": fields[name], "replace": value}]}
                            if edit == "patch"
                            else value
                            for name, value in updated.items()
                        },
                    }
                ]
            }
        )
    operations, error = protocol.parse(program, context)
    assert error is None
    (change,) = operations.model_dump()["cases"]
    assert change["case_name"] == "library" and change["execution_count"] == 1
    for name, value in updated.items():
        assert await PatchOp(FieldType.STRING).apply_checked(fields[name], change[name]) == value
    assert memory.extra_fields == fields  # Parsing never writes the original memory.
