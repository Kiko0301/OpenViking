from __future__ import annotations

import copy
import json
from dataclasses import asdict

import pytest
import yaml

from openviking.session.memory.experience.rubric import (
    normalize_case_rubric,
    parse_case_rubric,
    snapshot_rubric_results,
)
from openviking.session.memory.experience.trajectory_context import (
    AgentTrajectoryContextProvider,
    TrajectoryTarget,
)
from openviking.session.train import Case, Rubric, RubricCriterion

RUBRIC = [
    {"id": "confirm", "description": "Confirm changes before executing them."},
    {"id": "scope", "description": "Perform only the requested changes."},
]


@pytest.mark.parametrize("as_text", [False, True])
def test_rubric_roundtrip(as_text):
    value = json.dumps(RUBRIC) if as_text else RUBRIC
    assert parse_case_rubric(normalize_case_rubric(value)) == RUBRIC
    assert normalize_case_rubric(None) == "[]"


@pytest.mark.parametrize(
    "value",
    [
        "plain text",
        "{}",
        '[{"id":"a"}]',
        [{"id": "a", "description": ""}],
        [{"id": False, "description": "rule"}],
        [{"id": "a", "description": "rule", "score": 1}],
        [RUBRIC[0], RUBRIC[0]],
    ],
)
def test_invalid_rubric_is_rejected(value):
    with pytest.raises(ValueError):
        parse_case_rubric(value)


@pytest.mark.parametrize("passed", [True, False, None])
def test_results_preserve_model_order_and_freeze_missing_case_descriptions(passed):
    definitions = [dict(item) for item in RUBRIC]
    result = snapshot_rubric_results(
        [
            {"id": "scope", "passed": None, "evidence": "Not applicable to this request."},
            {"id": "confirm", "passed": passed, "evidence": "Observed confirmation evidence."},
        ],
        definitions,
    )
    assert result == [
        {**RUBRIC[1], "passed": None, "evidence": "Not applicable to this request."},
        {**RUBRIC[0], "passed": passed, "evidence": "Observed confirmation evidence."},
    ]
    definitions[0]["description"] = "changed"
    definitions.clear()
    assert result[1]["description"] == RUBRIC[0]["description"]


@pytest.mark.parametrize(
    "value",
    [
        [],
        None,
        False,
        0,
        {},
        "not JSON",
        '[{"id":"confirm","passed":true,"evidence":"{"nested":true}"}]',
        [None, False, "not an assessment"],
        [{"id": [], "passed": False, "evidence": "observed"}],
        [{"id": "", "passed": False, "evidence": "observed"}],
        [{"id": "   ", "passed": False, "evidence": "observed"}],
        [{"id": "confirm", "passed": "false", "evidence": "observed"}],
        [{"id": "confirm", "passed": 0, "evidence": "observed"}],
        [{"id": "confirm", "evidence": "observed"}],
        [{"id": "confirm", "passed": None, "evidence": ""}],
        [{"id": "confirm", "passed": False, "evidence": "   "}],
        [{"id": "confirm", "passed": False, "evidence": {"raw": "report"}}],
    ],
)
def test_unusable_results_are_omitted_without_inventing_judgments(value):
    assert snapshot_rubric_results(value, RUBRIC) == []


@pytest.mark.parametrize("as_text", [False, True])
def test_partial_results_preserve_model_definitions_and_unmatched_ids(as_text):
    value = [
        {
            "id": "scope",
            "description": "model-authored definition",
            "passed": True,
            "evidence": "Observed scope.",
        },
        {"id": "tau2_reward", "passed": False, "evidence": "External task judgment."},
        {"id": "confirm", "passed": "true", "evidence": "Not a boolean."},
    ]
    original = copy.deepcopy(value)
    assert snapshot_rubric_results(yaml.safe_dump(value) if as_text else value, RUBRIC) == [
        value[0],
        value[1],
    ]
    assert value == original


@pytest.mark.parametrize(
    "passed_values", [(True, False), (True, "false"), ("true", False), (True, False, None)]
)
def test_duplicate_ids_do_not_discard_independent_valid_assessments(passed_values):
    value = [
        {"id": "confirm", "passed": passed, "evidence": "Conflicting or invalid judgment."}
        for passed in passed_values
    ]
    value.append({"id": "scope", "passed": None, "evidence": "Not applicable."})
    expected = [
        {**RUBRIC[0], "passed": passed, "evidence": "Conflicting or invalid judgment."}
        for passed in passed_values
        if passed is None or type(passed) is bool
    ]
    expected.append({**RUBRIC[1], "passed": None, "evidence": "Not applicable."})
    assert snapshot_rubric_results(value, RUBRIC) == expected


def test_discarded_results_are_logged_without_logging_the_payload(monkeypatch):
    messages = []
    monkeypatch.setattr(
        "openviking.session.memory.experience.rubric.logger.warning",
        lambda message, *args: messages.append(message % args),
    )
    assert snapshot_rubric_results("- id: [private malformed assessment\n", RUBRIC) == []
    assert "Discarding unparseable evaluation_criteria YAML" in messages[0]
    assert "private malformed assessment" not in messages[0]
    assert snapshot_rubric_results([{"id": "unknown"}], RUBRIC) == []
    assert "Discarded 1 of 1 evaluation_criteria entries" in messages[1]


def test_excessively_nested_result_yaml_is_omitted():
    import sys

    depth = sys.getrecursionlimit() + 10
    assert snapshot_rubric_results("[" * depth + "0" + "]" * depth, RUBRIC) == []


async def test_bad_rubric_and_action_scores_do_not_block_factual_trajectory():
    from test_trajectory_context import operations

    provider = AgentTrajectoryContextProvider(
        messages=[], source_steps=[], targets=[TrajectoryTarget(case={"name": "case"})]
    )
    result = operations("Factual dialogue.")
    result.upsert_operations[0].memory_fields.update(
        evaluation_criteria="not a YAML list",
        action_scores=[{"id": "not_injected", "score": 1, "evidence": "unsupported"}],
    )
    errors = await provider.validate_resolved_operations(result)
    assert errors == []
    assert result.upsert_operations[0].memory_fields["action_scores"] == []
    assert result.upsert_operations[0].memory_fields["evaluation_criteria"] == []


@pytest.mark.asyncio
async def test_extraction_receives_case_rubric_and_arbitrary_external_evidence():
    from test_trajectory_context import operations

    case = Case(
        "library",
        "library",
        {},
        Rubric(
            "r",
            "",
            [RubricCriterion(item["id"], item["description"], True, 1.0) for item in RUBRIC],
        ),
    )
    # No passed/met/action_match/db_match keys in this evaluator-specific evidence.
    external = {"inspection": {"authorization": "absent", "affected_records": ["A"]}}
    provider = AgentTrajectoryContextProvider(
        messages=[],
        source_steps=[],
        targets=[TrajectoryTarget(case=asdict(case), external_evaluation=external)],
    )
    payload = json.loads((await provider.prefetch())[0]["content"])
    assert payload["cases"][0]["case"]["rubric"] == RUBRIC
    assert payload["cases"][0]["external_evaluation"] == external
    result = operations("Recorded dialogue.")
    result.upsert_operations[0].memory_fields["case_name"] = "library"
    result.upsert_operations[0].memory_fields["evaluation_criteria"] = yaml.safe_dump(
        [
            {
                "id": "confirm",
                "passed": False,
                "evidence": "External report: authorization absent.",
            },
            {"id": "scope", "passed": None, "evidence": "Requested scope not recorded."},
        ]
    )
    assert await provider.validate_resolved_operations(result) == []
    snapshots = result.upsert_operations[0].memory_fields["evaluation_criteria"]
    assert snapshots[0] == {
        **RUBRIC[0],
        "passed": False,
        "evidence": "External report: authorization absent.",
    }
    assert snapshots[1]["passed"] is None
    assert result.upsert_operations[0].memory_fields["steps"] == "Recorded dialogue."


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_cases", [True, False])
async def test_case_evidence_is_separate_from_conversation_and_respects_schema_filter(allow_cases):
    from unittest.mock import AsyncMock

    from test_memory_create_guard import CTX

    from openviking.message import Message, TextPart
    from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
    from openviking.session.memory.memory_type_registry import (
        MemoryTypeRegistry,
        resolve_memory_templates_dir,
    )
    from openviking.session.memory.session_extract_context_provider import (
        SessionExtractContextProvider,
    )

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "cases.yaml"))
    source = [Message(id="user", role="user", parts=[TextPart(text="Cancel the booking.")])]
    reports = ["# OpenViking OutcomeEvaluation\nExternal requirement and feedback"]
    provider = SessionExtractContextProvider(
        messages=source,
        ctx=CTX,
        memory_registry=registry,
        case_rubric_evidence=reports,
    )
    provider._isolation_handler = MemoryIsolationHandler(
        CTX,
        provider.get_extract_context(),
        allowed_memory_types={"cases"} if allow_cases else set(),
    )
    provider._eval_memory = AsyncMock(return_value={})
    messages = await provider.prefetch()
    assert "Cancel the booking." in messages[0]["content"]
    assert reports[0] not in messages[0]["content"]
    evidence_messages = [
        m for m in messages if (m.get("content") or "").startswith("# Case Rubric Evidence")
    ]
    assert len(evidence_messages) == int(allow_cases)
    if allow_cases:
        prompt = evidence_messages[0]["content"]
        assert "Only explicitly supplied rubric or acceptance criteria" in prompt
        assert "Task context and external evaluation reports are not additional criteria" in prompt
        assert "do not infer standards from behavior, scores or errors" in prompt
        assert "Keep the original meaning and scope" in prompt
        assert "Extract reusable evaluation standards" not in prompt
        assert prompt.endswith(reports[0])
    assert provider.messages == source


@pytest.mark.parametrize(("token", "passed"), [("true", True), ("false", False), ("null", None)])
def test_yaml_assessments_preserve_block_evidence_and_case_definitions(token, passed):
    text = r"""- id: scope
  passed: null
  evidence: |
    Not enough evidence for this criterion.
- id: confirm
  passed: TOKEN
  evidence: |-
    用户说了 "yes": # 这是证据，不是 YAML 注释。
    路径 C:\tmp\file；原始结果 {"updated": true}。
""".replace("TOKEN", token)
    result = snapshot_rubric_results(text, RUBRIC)
    assert result == [
        {
            **RUBRIC[1],
            "passed": None,
            "evidence": "Not enough evidence for this criterion.\n",
        },
        {
            **RUBRIC[0],
            "passed": passed,
            "evidence": '用户说了 "yes": # 这是证据，不是 YAML 注释。\n'
            '路径 C:\\tmp\\file；原始结果 {"updated": true}。',
        },
    ]


def test_yaml_assessments_do_not_execute_python_tags(monkeypatch):
    from unittest.mock import Mock

    run = Mock()
    monkeypatch.setattr("os.system", run)
    assert snapshot_rubric_results("!!python/object/apply:os.system ['must-not-run']", RUBRIC) == []
    run.assert_not_called()


def test_yaml_quoted_ids_and_block_evidence_are_not_coerced_to_booleans():
    definition = {"id": "on", "description": "Record the switch state."}
    text = """- id: "on"
  passed: false
  evidence: |-
    yes
"""
    assert snapshot_rubric_results(text, [definition]) == [
        {**definition, "passed": False, "evidence": "yes"}
    ]


@pytest.mark.parametrize("as_text", [False, True])
@pytest.mark.parametrize("passed", [True, False, None])
@pytest.mark.parametrize(
    "definitions", [[], [{"id": "tau2_reward_full", "description": "Full task completion."}]]
)
def test_unmatched_assessment_is_preserved_without_aliasing_or_warning(
    monkeypatch, as_text, passed, definitions
):
    def unexpected_warning(*args):
        pytest.fail("A Case rubric mismatch must not be treated as a failed extraction")

    monkeypatch.setattr(
        "openviking.session.memory.experience.rubric.logger.warning", unexpected_warning
    )
    assessment = {
        "id": "tau2_reward",
        "passed": passed,
        "evidence": "External report: required information not communicated.",
    }
    value = yaml.safe_dump([assessment]) if as_text else [assessment]
    assert snapshot_rubric_results(value, definitions) == [assessment]


def test_unknown_criterion_keeps_its_description_and_evidence():
    assessment = {
        "id": "additional_check",
        "description": "A context-specific requirement not covered by the Case rubric.",
        "passed": False,
        "evidence": "The observed action violated that requirement.",
    }
    assert snapshot_rubric_results([assessment], RUBRIC) == [assessment]


def test_rubric_prompt_uses_case_as_guidance_not_an_id_whitelist():
    from openviking.session.memory.memory_type_registry import (
        MemoryTypeRegistry,
        resolve_memory_templates_dir,
    )

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_yaml(str(resolve_memory_templates_dir() / "trajectories.yaml"))
    schema = registry.get("trajectories")
    field = next(field for field in schema.fields if field.name == "evaluation_criteria")
    description = " ".join(field.description.split())
    assert "guidance, not an exhaustive list or an ID whitelist" in description
    assert "Prefer a supplied Case criterion ID" in description
    assert "preserves model descriptions" in description
    assert "an unmatched ID does not discard the assessment" in description
    assert "copy criterion_name verbatim as id" in description
    assert "copy the reported passed/score unchanged" in description
    assert "Explain discrepancies in evidence without changing results" in description
    assert "An overall pass/fail does not establish each criterion's result" in description
    assert "only an exact case.rubric id" not in description
    assert "Do not substitute" not in description


@pytest.mark.parametrize("score", [0, 0.0, 0.75, -1.0, None])
def test_rubric_snapshot_preserves_explicit_score(score):
    assessment = {"id": "external", "passed": False, "score": score, "evidence": "Reported."}
    assert snapshot_rubric_results([assessment], []) == [assessment]


@pytest.mark.parametrize("score", [True, "0", float("nan"), float("inf")])
def test_invalid_optional_score_does_not_discard_the_judgment(score):
    assessment = {"id": "external", "passed": False, "score": score, "evidence": "Reported."}
    assert snapshot_rubric_results([assessment], []) == [
        {"id": "external", "passed": False, "evidence": "Reported."}
    ]


def external_result_provider(evaluation):
    return AgentTrajectoryContextProvider(
        messages=[],
        source_steps=[],
        targets=[
            TrajectoryTarget(
                case={
                    "name": "case",
                    "rubric": {
                        "criteria": [{"name": "domain_reward", "description": "Reported reward."}]
                    },
                },
                external_evaluation=evaluation,
            )
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "assessments",
    [
        [],
        [{"id": "external_reward", "passed": None, "score": 0.0, "evidence": "Disagree."}],
        [{"id": "external_reward", "passed": True, "score": 0.0, "evidence": "Tool succeeded."}],
        [{"id": "external_reward", "passed": False, "evidence": "Missing score."}],
        [{"id": "external_reward", "passed": False, "score": 1.0, "evidence": "Changed score."}],
        [{"id": "external_reward", "passed": False, "score": False, "evidence": "Bad score."}],
        [{"id": "external_reward", "passed": 0, "score": 0.0, "evidence": "Bad judgment."}],
        [{"id": "domain_reward", "passed": False, "score": 0.0, "evidence": "Renamed ID."}],
        [
            {"id": "external_reward", "passed": False, "score": 0.0, "evidence": "Reported."},
            {"id": "external_reward", "passed": None, "score": 0.0, "evidence": "Disagree."},
        ],
    ],
    ids=[
        "omitted",
        "null",
        "true",
        "score-omitted",
        "score-changed",
        "bool-score",
        "int-passed",
        "renamed",
        "duplicate",
    ],
)
async def test_external_result_mismatch_requests_repair_without_overwriting(assessments):
    from test_trajectory_context import operations

    external = {
        "passed": False,
        "score": 0.0,
        "criterion_results": [{"criterion_name": "external_reward", "passed": False, "score": 0.0}],
    }
    original = copy.deepcopy(external)
    provider = external_result_provider(external)
    result = operations("Tool completed, but external acceptance failed.")
    result.upsert_operations[0].memory_fields["evaluation_criteria"] = copy.deepcopy(assessments)
    errors = await provider.validate_resolved_operations(result)
    assert errors
    assert errors[0]["case_name"] == "case"
    assert errors[0]["expected"] == {"id": "external_reward", "passed": False, "score": 0.0}
    assert result.upsert_operations[0].memory_fields[
        "evaluation_criteria"
    ] == snapshot_rubric_results(assessments, provider.targets["case"].rubric)
    assert external == original
    assert "external_reward" in provider.render_operation_repair(errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("as_text", [False, True])
@pytest.mark.parametrize(("passed", "score"), [(False, 0.0), (True, 0.75), (None, None)])
async def test_external_results_and_context_assessments_share_one_output(as_text, passed, score):
    from test_trajectory_context import operations

    provider = external_result_provider(
        {
            "passed": False,  # Overall failure must not determine every criterion.
            "score": 0.0,
            "criterion_results": [
                {"criterion_name": "external_reward", "passed": passed, "score": score}
            ],
        }
    )
    assessments = [
        {"id": "external_reward", "passed": passed, "score": score, "evidence": "External report."},
        {"id": "user_confirmation", "passed": True, "evidence": "User explicitly confirmed."},
    ]
    result = operations("Recorded facts.")
    result.upsert_operations[0].memory_fields["evaluation_criteria"] = (
        yaml.safe_dump(assessments) if as_text else assessments
    )
    assert await provider.validate_resolved_operations(result) == []
    assert result.upsert_operations[0].memory_fields["evaluation_criteria"] == assessments


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "external",
    [
        None,
        {},
        {"inspection": {"authorization": "absent"}},
        {"passed": False, "score": 0.0, "criterion_results": []},
        {"criterion_results": [{"criterion_name": "context", "feedback": ["Inspect evidence."]}]},
    ],
)
async def test_no_external_criterion_judgment_keeps_context_only_extraction(external):
    from test_trajectory_context import operations

    provider = external_result_provider(external)
    assessments = [{"id": "context", "passed": True, "evidence": "The user confirmed."}]
    result = operations("Observed user confirmation.")
    result.upsert_operations[0].memory_fields["evaluation_criteria"] = assessments
    assert await provider.validate_resolved_operations(result) == []
    assert result.upsert_operations[0].memory_fields["evaluation_criteria"] == assessments


@pytest.mark.asyncio
@pytest.mark.parametrize("judgments", [{"passed": False}, {"score": 0.25}])
async def test_partial_external_result_requires_only_fields_actually_reported(judgments):
    from test_trajectory_context import operations

    provider = external_result_provider(
        {"criterion_results": [{"criterion_name": "external_reward", **judgments}]}
    )
    assessment = {
        "id": "external_reward",
        "passed": None,
        **judgments,
        "evidence": "External fields preserved; unreported judgment may remain unknown.",
    }
    result = operations("Recorded facts.")
    result.upsert_operations[0].memory_fields["evaluation_criteria"] = [assessment]
    assert await provider.validate_resolved_operations(result) == []
    assert result.upsert_operations[0].memory_fields["evaluation_criteria"] == [assessment]
