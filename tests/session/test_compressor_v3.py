# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openviking.message import Message, TextPart
from openviking.server.identity import RequestContext, Role
from openviking.session import create_session_compressor
from openviking.session.compressor_v3 import (
    SessionCompressorV3,
    _canonical_cases_from_update_result,
    _commit_experience_snapshot,
    _contexts_from_update_result,
    _experience_root_uri,
    _experience_session_map,
    _experience_snapshot_provenance,
    _report_extraction_telemetry,
    _training_evaluation_from_messages,
    _training_outcome_from_messages,
    _visible_experience_snapshot_uris,
)
from openviking.session.memory.dataclass import (
    MemoryFile,
    MemoryOperationSkipCode,
    ResolvedOperation,
    ResolvedOperations,
    SkippedMemoryOperation,
    StoredLink,
)
from openviking.session.memory.memory_updater import MemoryUpdateResult
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.train import (
    Case,
    CriterionResult,
    ExperienceSet,
    PolicyApplyResult,
    PolicyPlanItem,
    PolicyUpdatePlan,
    Rollout,
    RolloutAnalysis,
    RolloutTrainingResult,
    Rubric,
    RubricCriterion,
    RubricEvaluation,
    StreamingPolicyTrainerConfig,
    Trajectory,
)
from openviking.session.train.components.session_commit import (
    _case_spec_message_to_request,
    _evaluation_message_to_request,
)
from openviking.telemetry import OperationTelemetry, bind_telemetry
from openviking_cli.exceptions import ConflictError
from openviking_cli.session.user_id import UserIdentifier


@pytest.fixture(autouse=True)
def isolated_compressor_storage(monkeypatch):
    monkeypatch.setattr("openviking.session.compressor_v3.get_viking_fs", lambda: SimpleNamespace())

    async def registry_snapshot(fs, account_id, registry):
        return registry

    monkeypatch.setattr(
        "openviking.session.memory.account_templates.resolve_account_memory_registry",
        registry_snapshot,
    )


def _ctx() -> RequestContext:
    return RequestContext(user=UserIdentifier.the_default_user("u"), role=Role.ROOT)


def _messages() -> list[Message]:
    return [
        Message(
            id="m1",
            role="user",
            parts=[TextPart("请处理重复预订，只取消确认是重复的那一单。")],
        ),
        Message(
            id="m2",
            role="assistant",
            parts=[TextPart("已读取两个预订，确认第二个是重复记录并取消。")],
        ),
    ]


def _case_operation() -> ResolvedOperation:
    return ResolvedOperation(
        old_memory_file_content=None,
        memory_type="cases",
        uris=["viking://user/u/memories/cases/重复预订处理.md"],
        memory_fields={
            "case_name": "重复预订处理",
            "domain": "处理重复预订并只取消确认重复的订单",
            "description": '{"summary":"用户要求处理重复预订","preconditions":["存在两个相似预订"]}',
            "rubric": '[{"id":"verify_duplicate","description":"取消前必须确认哪一单是重复订单"},{"id":"scope","description":"不得影响有效订单"}]',
            "evidence": "助手根据读取结果确认重复项并完成取消。",
        },
    )


def test_factory_defaults_to_v3():
    compressor = create_session_compressor(vikingdb=None)
    assert isinstance(compressor, SessionCompressorV3)


@pytest.mark.asyncio
async def test_commit_extraction_propagates_template_conflict():
    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=SimpleNamespace())
    conflict = ConflictError("Conflicting memory template snapshots; Re-extract before retrying")
    compressor._extract_user_memories = AsyncMock(side_effect=conflict)
    compressor._write_final_memory_diff = AsyncMock()

    with pytest.raises(ConflictError) as raised:
        await compressor.extract_long_term_memories(
            messages=_messages(),
            ctx=_ctx(),
            allowed_memory_types={"profile"},
            # The Session commit worker uses this mode: a conflict must reach it,
            # not turn into an empty successful extraction or memory diff.
            strict_extract_errors=True,
        )
    assert raised.value is conflict
    compressor._write_final_memory_diff.assert_not_awaited()


def test_factory_ignores_deprecated_memory_version():
    assert isinstance(
        create_session_compressor(vikingdb=None, memory_version="v2"), SessionCompressorV3
    )
    assert isinstance(
        create_session_compressor(vikingdb=None, memory_version="unsupported"),
        SessionCompressorV3,
    )


def test_extract_long_term_memories_appends_extraction_context_after_legacy_parameters():
    parameter_names = list(
        inspect.signature(SessionCompressorV3.extract_long_term_memories).parameters
    )

    assert parameter_names[-7:] == [
        "allow_self_memory",
        "allowed_peer_ids",
        "event_search_tags",
        "peer_memory_enabled",
        "extraction_context",
        "extract_max_iterations",
        "extract_thinking",
    ]
    assert (
        inspect.signature(SessionCompressorV3.extract_long_term_memories)
        .parameters["extract_max_iterations"]
        .kind
        == inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(SessionCompressorV3.extract_long_term_memories)
        .parameters["extract_thinking"]
        .kind
        == inspect.Parameter.KEYWORD_ONLY
    )


@pytest.mark.parametrize("iterations", [None, 9])
def test_react_iteration_budget_defaults_and_override(iterations):
    compressor = SessionCompressorV3(vikingdb=None)
    kwargs = {} if iterations is None else {"max_iterations": iterations}

    loop = compressor._get_or_create_react(
        ctx=_ctx(),
        context_provider=SimpleNamespace(),
        vlm_config=SimpleNamespace(model="test"),
        **kwargs,
    )

    assert loop.max_iterations == (3 if iterations is None else iterations)


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [None, False, True])
async def test_react_thinking_defaults_and_reaches_llm(thinking):
    vlm = SimpleNamespace(
        model="test", thinking=True, get_completion_async=AsyncMock(return_value="sdk.commit()")
    )
    compressor = SessionCompressorV3(vikingdb=None)
    kwargs = {} if thinking is None else {"thinking": thinking}
    loop = compressor._get_or_create_react(
        ctx=_ctx(), context_provider=SimpleNamespace(), vlm_config=vlm, **kwargs
    )
    loop._output_protocol = SimpleNamespace(parse=Mock(return_value=(object(), None)))

    await loop._call_llm([{"role": "user", "content": "extract memory"}])

    expected = False if thinking is None else thinking
    assert loop.thinking is expected
    assert vlm.get_completion_async.await_args.kwargs["thinking"] is expected


def test_report_extraction_telemetry_handles_placeholder_error_targets():
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                old_memory_file_content=None,
                memory_fields={},
                memory_type="events",
                uris=["unknown"],
            )
        ],
        delete_file_contents=[],
        errors=[],
    )
    result = MemoryUpdateResult()
    result.add_written("unknown")
    result.add_edited("viking://user/u/memories/preferences/pref.md")
    result.add_deleted("viking://user/u/memories/events/old.md")
    result.add_error("events(page_id=xyz)", ValueError("Missing resolved URI"))

    telemetry = OperationTelemetry(operation="session.commit", enabled=True)
    with bind_telemetry(telemetry):
        _report_extraction_telemetry(result, operations)

    summary = telemetry.finish().summary
    extract = summary["memory"]["extract"]
    assert extract["actions"] == {
        "created": 1,
        "merged": 1,
        "deleted": 1,
        "failed": 1,
    }
    assert extract["actions_by_type"] == {
        "events": {"created": 1, "deleted": 1},
        "preferences": {"merged": 1},
        "unknown": {"failed": 1},
    }


@pytest.mark.asyncio
async def test_memory_diff_includes_intentionally_skipped_operations(monkeypatch):
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(),
    )
    compressor = SessionCompressorV3(vikingdb=None)
    result = MemoryUpdateResult()
    result.add_skipped(
        SkippedMemoryOperation(
            memory_type="events",
            page_id=101,
            reason_code=MemoryOperationSkipCode.INVALID_RANGES,
            reason="No valid event range could be resolved",
        )
    )

    diff = await compressor._build_memory_diff(
        result=result,
        operations=ResolvedOperations(
            upsert_operations=[],
            delete_file_contents=[],
            errors=[],
        ),
        viking_fs=SimpleNamespace(),
        ctx=_ctx(),
        archive_uri="viking://user/u/sessions/s1/history/archive_001",
    )

    assert diff["skipped_operations"] == [
        {
            "memory_type": "events",
            "page_id": 101,
            "reason_code": "invalid_ranges",
            "reason": "No valid event range could be resolved",
        }
    ]
    assert diff["summary"] == {
        "total_adds": 0,
        "total_updates": 0,
        "total_deletes": 0,
        "total_skipped": 1,
    }


@pytest.mark.asyncio
async def test_v3_skips_agent_training_when_agent_evolution_is_disabled(monkeypatch):
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(),
    )
    compressor = SessionCompressorV3(vikingdb=None)
    compressor._extract_user_memories = AsyncMock(
        return_value=SimpleNamespace(
            contexts=[],
            cases=[_training_case()],
            memory_diff={"operations": {}},
            case_uri_by_name={},
            skipped_operations=[
                {
                    "memory_type": "profile",
                    "reason_code": "peer_memory_disabled",
                    "reason": "Peer memory writes are disabled",
                }
            ],
        )
    )
    compressor.train_from_extracted_cases = AsyncMock()
    compressor._write_final_memory_diff = AsyncMock()

    result = await compressor.extract_long_term_memories(
        messages=_messages(),
        ctx=_ctx(),
        allowed_memory_types={"cases", "profile"},
        agent_evolution_enabled=False,
    )

    compressor.train_from_extracted_cases.assert_not_awaited()
    assert result == []


@pytest.mark.asyncio
async def test_v3_extracts_session_skills_when_agent_evolution_is_disabled(monkeypatch):
    config = SimpleNamespace(
        memory=SimpleNamespace(session_skill_extraction_enabled=True),
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_openviking_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(),
    )
    compressor = SessionCompressorV3(
        vikingdb=None,
        skill_processor=SimpleNamespace(),
    )
    compressor._extract_user_memories = AsyncMock(
        return_value=SimpleNamespace(
            contexts=[],
            cases=[],
            memory_diff={"operations": {}},
            case_uri_by_name={},
        )
    )
    compressor.train_from_extracted_cases = AsyncMock()
    compressor.extract_session_skills = AsyncMock(
        return_value={
            "case_count": 0,
            "submitted": 0,
            "skill_submitted": 1,
            "skill_uris": ["viking://user/u/skills/code-review/SKILL.md"],
        }
    )
    compressor._write_final_memory_diff = AsyncMock()

    result = await compressor.extract_long_term_memories(
        messages=_messages(),
        ctx=_ctx(),
        allowed_memory_types={"profile", "preferences"},
        agent_evolution_enabled=False,
    )

    compressor.train_from_extracted_cases.assert_not_awaited()
    compressor.extract_session_skills.assert_awaited_once()
    assert result["session_skills"] == [
        {
            "uri": "viking://user/u/skills/code-review/SKILL.md",
            "archive_uri": "",
        }
    ]


@pytest.mark.asyncio
async def test_v3_skill_only_extraction_submits_gradients_without_agent_memories(monkeypatch):
    from openviking.session.train import PatchSemanticGradient

    skill_uri = "viking://user/u/skills/code-review/SKILL.md"
    gradient = PatchSemanticGradient(
        before_file=None,
        after_file=MemoryFile(
            uri=skill_uri,
            content="## Workflow\n- Read changed files first.",
            memory_type="skills",
            extra_fields={"skill_name": "code-review"},
        ),
        base_version=None,
        rationale="test",
        links=[],
        confidence=0.9,
        metadata={},
    )
    analyzer = SimpleNamespace(
        extract_session_skills=AsyncMock(
            return_value={"contexts": [], "skill_gradients": [gradient]}
        )
    )
    trainer = SimpleNamespace(
        submit_gradients=AsyncMock(
            return_value=SimpleNamespace(apply_result=SimpleNamespace(written_uris=[skill_uri]))
        )
    )
    compressor = SessionCompressorV3(
        vikingdb=None,
        rollout_analyzer=analyzer,
        skill_processor=SimpleNamespace(),
    )
    compressor._session_skill_extraction_enabled = lambda: True
    compressor._get_session_skill_trainer = AsyncMock(return_value=trainer)
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(),
    )

    result = await compressor.extract_session_skills(
        messages=_messages(),
        ctx=_ctx(),
        archive_uri="viking://user/u/sessions/s1/history/archive_001",
    )

    analyzer.extract_session_skills.assert_awaited_once()
    assert "include_trajectories" not in analyzer.extract_session_skills.await_args.kwargs
    trainer.submit_gradients.assert_awaited_once_with([gradient])
    assert result == {
        "case_count": 0,
        "submitted": 0,
        "skill_submitted": 1,
        "skill_uris": [skill_uri],
    }


@pytest.mark.asyncio
async def test_v3_skips_agent_training_when_execution_memory_types_are_filtered(monkeypatch):
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(),
    )
    compressor = SessionCompressorV3(vikingdb=None)
    compressor._extract_user_memories = AsyncMock(
        return_value=SimpleNamespace(
            contexts=[],
            cases=[_training_case()],
            memory_diff={"operations": {}},
            case_uri_by_name={},
        )
    )
    compressor.train_from_extracted_cases = AsyncMock()
    compressor._write_final_memory_diff = AsyncMock()

    await compressor.extract_long_term_memories(
        messages=_messages(),
        ctx=_ctx(),
        allowed_memory_types={"cases", "profile"},
        agent_evolution_enabled=True,
    )

    compressor.train_from_extracted_cases.assert_not_awaited()


@pytest.mark.asyncio
async def test_v3_passes_allowed_execution_types_to_agent_training(monkeypatch):
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(),
    )
    compressor = SessionCompressorV3(vikingdb=None)
    compressor._extract_user_memories = AsyncMock(
        return_value=SimpleNamespace(
            contexts=[],
            cases=[_training_case()],
            memory_diff={"operations": {}},
            case_uri_by_name={},
        )
    )
    compressor.train_from_extracted_cases = AsyncMock(
        return_value={"case_count": 1, "submitted": 1}
    )
    compressor._write_final_memory_diff = AsyncMock()

    await compressor.extract_long_term_memories(
        messages=_messages(),
        ctx=_ctx(),
        allowed_memory_types={"cases", "trajectories"},
        agent_evolution_enabled=True,
    )

    assert compressor.train_from_extracted_cases.await_args.kwargs["allowed_memory_types"] == {
        "trajectories"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("iterations", [3, 6])
@pytest.mark.parametrize("thinking", [False, True])
async def test_v3_initializes_only_allowed_memory_files(monkeypatch, iterations, thinking):
    initialized_with = []
    loop_kwargs = []

    class DummyRegistry:
        async def initialize_memory_files(self, ctx, allowed_memory_types=None):
            del ctx
            initialized_with.append(allowed_memory_types)

    class DummyOrchestrator:
        async def run(self):
            return None, []

    compressor = SessionCompressorV3(
        vikingdb=None,
        vlm_resolver=SimpleNamespace(get_vlm=AsyncMock(return_value=SimpleNamespace(model="test"))),
    )

    def create_loop(**kwargs):
        loop_kwargs.append(kwargs)
        return DummyOrchestrator()

    compressor._get_or_create_react = create_loop
    compressor._write_final_memory_diff = AsyncMock()
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_default_registry",
        lambda: DummyRegistry(),
    )

    await compressor.extract_long_term_memories(
        messages=_messages(),
        ctx=_ctx(),
        allowed_memory_types={"profile"},
        agent_evolution_enabled=False,
        extract_max_iterations=iterations,
        extract_thinking=thinking,
    )

    assert initialized_with == [{"profile"}]
    assert loop_kwargs[0]["max_iterations"] == iterations
    assert loop_kwargs[0]["thinking"] is thinking


def test_experience_root_uri_requires_request_user():
    with pytest.raises(ValueError, match="RequestContext.user.user_id is required"):
        _experience_root_uri(SimpleNamespace(user=None))


@pytest.mark.asyncio
async def test_train_from_extracted_cases_persists_only_trajectory_phase(monkeypatch):
    submitted_gradients = []
    submitted_analyses = []
    analyzed_rollouts = []

    class FakeTrainer:
        policy_set = ExperienceSet(
            root_uri="viking://user/u/memories/experiences",
            policies=[],
        )

        async def submit_gradients(
            self, gradients, *, analysis=None, rollout=None, batch_finalizer=None
        ):
            submitted_gradients.append(gradients)
            submitted_analyses.append(analysis)
            return RolloutTrainingResult(
                analyses=[analysis] if analysis else [],
                gradients=list(gradients),
                plan=PolicyUpdatePlan(items=[], metadata={}),
                apply_result=PolicyApplyResult(
                    updated_policy_set=self.policy_set,
                    written_uris=[],
                    errors=[],
                ),
                metadata={},
            )

    memory = MemoryFile(
        uri="viking://user/u/memories/trajectories/duplicate_booking/run.md",
        content="Factual observation",
        memory_type="trajectories",
    )

    class FakeAnalyzer:
        async def analyze_many(self, rollouts, context):
            return [await self.analyze(rollout, context) for rollout in rollouts]

        async def analyze(self, rollout, context):
            assert context.memory_registry is not None
            analyzed_rollouts.append(rollout)
            return RolloutAnalysis(
                rollout=rollout,
                evaluation=rollout.evaluation,
                trajectories=[
                    Trajectory(
                        name="run", uri=memory.uri, content=memory.content, outcome="failure"
                    )
                ],
                metadata={
                    "source_session_uri": "viking://user/u/sessions/t1/history/archive_001",
                    "trajectory_memory": memory,
                },
                gradients=[],
            )

    extra_extraction = AsyncMock(side_effect=AssertionError("Compressor must not extract twice"))

    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(ls=AsyncMock(return_value=[])),
    )
    trainer_factory = AsyncMock(return_value=FakeTrainer())
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_streaming_policy_trainer",
        trainer_factory,
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.SessionRolloutAnalyzer._extract_trajectories",
        extra_extraction,
    )
    compressor = SessionCompressorV3(
        vikingdb=None,
        rollout_analyzer=FakeAnalyzer(),
        streaming_trainer_config=StreamingPolicyTrainerConfig(
            max_wait_seconds=60,
            max_gradients_per_update=8,
        ),
    )
    cases = [_training_case()]
    source_rollout = Rollout(
        case=cases[0],
        messages=[],
        policy_snapshot_id="snapshot-1",
        evaluation=RubricEvaluation(
            passed=False,
            score=0.0,
            criterion_results=[],
            feedback=["failed"],
        ),
        metadata={
            "dag_runtime": {
                "events": [
                    {
                        "experience_uri": "viking://user/u/memories/experiences/e1.md",
                        "state": "completed",
                        "revision": 2,
                        "slot_values": {"remind": True},
                        "node_slots": {1: "remind"},
                        "executed_nodes": [1],
                        "current_nodes": [],
                    }
                ]
            }
        },
    )
    outcome_request = _evaluation_message_to_request(source_rollout)
    outcome_message = Message(
        id="evaluation",
        role="user",
        parts=[TextPart(text=outcome_request["parts"][0]["text"])],
    )
    result = await compressor.train_from_extracted_cases(
        cases=cases,
        messages=[*_messages(), outcome_message],
        ctx=_ctx(),
        session_id="s1",
    )

    assert result["case_count"] == 1
    assert result["submitted"] == 1
    assert result["trajectory_uris"] == [memory.uri]
    extra_extraction.assert_not_awaited()
    assert result["experience_learning_deferred"] is True
    assert result["persisted_experience_count"] == 0
    trainer_factory.assert_not_awaited()
    assert submitted_gradients == []
    assert submitted_analyses == []
    # Verify analysis was used
    assert cases[0].name == "duplicate_booking"
    assert [message.id for message in analyzed_rollouts[0].messages] == ["m1", "m2"]
    assert analyzed_rollouts[0].evaluation == source_rollout.evaluation
    assert analyzed_rollouts[0].metadata["experience_execution"] == {
        "viking://user/u/memories/experiences/e1.md": {
            "experience_name": "e1",
            "state": "completed",
            "revision": 2,
            "slot_values": {"remind": True},
            "executed_nodes": ["remind"],
            "current_nodes": [],
        }
    }


@pytest.mark.asyncio
async def test_train_from_extracted_cases_skips_trajectory_extraction_when_not_allowed(
    monkeypatch,
):
    analyzed = []

    class FakeTrainer:
        policy_set = ExperienceSet(
            root_uri="viking://user/u/memories/experiences",
            policies=[],
        )

    class FakeAnalyzer:
        async def analyze_many(self, rollouts, context):
            return [await self.analyze(rollout, context) for rollout in rollouts]

        async def analyze(self, rollout, context):
            del context
            analyzed.append(rollout)
            return RolloutAnalysis(
                evaluation=RubricEvaluation(
                    passed=True,
                    score=1.0,
                    criterion_results=[],
                    feedback=[],
                ),
                trajectories=[],
                metadata={"source_session_uri": "viking://user/u/sessions/t1/history/archive_001"},
                gradients=[],
            )

    estimate = AsyncMock(side_effect=AssertionError("experience estimation must be skipped"))
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(ls=AsyncMock(return_value=[])),
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_streaming_policy_trainer",
        AsyncMock(return_value=FakeTrainer()),
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.SessionRolloutAnalyzer._extract_trajectories",
        estimate,
    )

    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=FakeAnalyzer())
    result = await compressor.train_from_extracted_cases(
        cases=[_training_case()],
        messages=_messages(),
        ctx=_ctx(),
        session_id="s1",
        allowed_memory_types={"experiences"},
    )

    assert len(analyzed) == 0
    assert result["submitted"] == 0
    estimate.assert_not_awaited()


@pytest.mark.asyncio
async def test_train_from_extracted_multiple_case_memories_analyzes_bound_rollouts(monkeypatch):
    seen_rollouts = []
    rollout_messages = _messages()

    class FakeTrainer:
        policy_set = ExperienceSet(
            root_uri="viking://user/u/memories/experiences",
            policies=[],
        )

    class FakeAnalyzer:
        async def analyze_many(self, rollouts, context):
            return [await self.analyze(rollout, context) for rollout in rollouts]

        async def analyze(self, rollout, context):
            del context
            seen_rollouts.append(rollout)
            return RolloutAnalysis(
                evaluation=RubricEvaluation(
                    passed=True,
                    score=1.0,
                    criterion_results=[],
                    feedback=[],
                ),
                trajectories=[],
                gradients=[],
            )

    async def fake_estimate_exp_gradients(self, *args, **kwargs):
        return []

    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(ls=AsyncMock(return_value=[])),
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_streaming_policy_trainer",
        AsyncMock(return_value=FakeTrainer()),
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.SessionRolloutAnalyzer._extract_trajectories",
        fake_estimate_exp_gradients,
    )

    case_a = _training_case()
    case_b = Case(
        name="case_b",
        task_signature="Handle a second extracted case.",
        input={"summary": "second case"},
        rubric=case_a.rubric,
    )

    compressor = SessionCompressorV3(
        vikingdb=None,
        rollout_analyzer=FakeAnalyzer(),
        streaming_trainer_config=StreamingPolicyTrainerConfig(
            max_wait_seconds=60,
            max_gradients_per_update=8,
        ),
    )

    result = await compressor.train_from_extracted_cases(
        cases=[case_a, case_b],
        messages=rollout_messages,
        ctx=_ctx(),
        session_id="s1",
    )

    assert result["case_count"] == 2
    assert result["submitted"] == 0
    assert [rollout.case.name for rollout in seen_rollouts] == [
        "duplicate_booking",
        "case_b",
    ]
    assert [rollout.messages for rollout in seen_rollouts] == [
        rollout_messages,
        rollout_messages,
    ]


@pytest.mark.asyncio
async def test_v3_extract_uses_patch_merge_without_directory_lock(monkeypatch):
    applied_operations = []
    trained_cases = []

    class DummyRegistry:
        async def initialize_memory_files(self, ctx, allowed_memory_types=None):
            del ctx, allowed_memory_types
            return None

    class DummyOrchestrator:
        async def run(self):
            return (
                ResolvedOperations(
                    upsert_operations=[_case_operation()],
                    delete_file_contents=[],
                    errors=[],
                ),
                [],
            )

    class FakeStreamingUpdater:
        async def submit(self, request):
            applied_operations.append(request.operations)
            result = MemoryUpdateResult()
            result.add_written(_case_operation().uris[0])
            return SimpleNamespace(operations=request.operations, apply_result=result)

    compressor = SessionCompressorV3(
        vikingdb=None,
        vlm_resolver=SimpleNamespace(get_vlm=AsyncMock(return_value=SimpleNamespace(model="test"))),
    )
    compressor._get_or_create_react = lambda **kwargs: DummyOrchestrator()

    async def fake_train_from_extracted_cases(**kwargs):
        trained_cases.extend(kwargs["cases"])
        return {"case_count": len(kwargs["cases"]), "submitted": len(kwargs["cases"])}

    compressor.train_from_extracted_cases = fake_train_from_extracted_cases

    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(
            write_file=AsyncMock(),
            read_file=AsyncMock(
                return_value=MemoryFileUtils.write(
                    MemoryFile(
                        uri=_case_operation().uris[0],
                        memory_type="cases",
                        content="",
                        extra_fields=_case_operation().memory_fields,
                    )
                )
            ),
        ),
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_default_registry",
        lambda: DummyRegistry(),
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_streaming_memory_updater",
        AsyncMock(return_value=FakeStreamingUpdater()),
    )

    contexts = await compressor.extract_long_term_memories(
        messages=_messages(),
        ctx=_ctx(),
        allowed_memory_types={"cases", "profile", "trajectories"},
    )

    assert len(applied_operations) == 1
    assert applied_operations[0].upsert_operations[0].memory_type == "cases"
    assert [case.name for case in trained_cases] == ["重复预订处理"]
    assert contexts[0].uri.endswith("重复预订处理.md")


@pytest.mark.asyncio
async def test_v3_extract_trains_only_canonical_case_after_patch_merge(monkeypatch):
    trained_kwargs = []
    canonical_uri = "viking://user/u/memories/cases/duplicate_booking.md"
    loser_uri = "viking://user/u/memories/cases/duplicate_booking_duplicate.md"
    canonical_fields = {
        "case_name": "duplicate_booking",
        "domain": "Handle duplicate bookings safely.",
        "description": '{"summary":"cancel only the confirmed duplicate booking"}',
        "rubric": (
            '[{"id":"verify_duplicate","description":"The assistant verifies '
            'which booking is duplicate before cancellation."}]'
        ),
        "evidence": "Canonical merged case evidence.",
    }

    def case_op(uri: str, name: str) -> ResolvedOperation:
        fields = dict(canonical_fields)
        fields["case_name"] = name
        return ResolvedOperation(
            old_memory_file_content=None,
            memory_type="cases",
            uris=[uri],
            memory_fields=fields,
        )

    class DummyRegistry:
        async def initialize_memory_files(self, ctx, allowed_memory_types=None):
            del ctx, allowed_memory_types
            return None

    class DummyOrchestrator:
        async def run(self):
            return (
                ResolvedOperations(
                    upsert_operations=[
                        case_op(canonical_uri, "duplicate_booking"),
                        case_op(loser_uri, "duplicate_booking_duplicate"),
                    ],
                    delete_file_contents=[],
                    errors=[],
                ),
                [],
            )

    class FakeFS:
        async def read_file(self, uri, ctx=None):
            del ctx
            if uri != canonical_uri:
                raise FileNotFoundError(uri)
            return MemoryFileUtils.write(
                MemoryFile(
                    uri=canonical_uri,
                    content="",
                    memory_type="cases",
                    extra_fields=dict(canonical_fields),
                )
            )

        async def write_file(self, uri, content, ctx=None):
            del uri, content, ctx

    class FakeStreamingUpdater:
        async def submit(self, request):
            del request
            result = MemoryUpdateResult()
            result.add_written(canonical_uri)
            return SimpleNamespace(
                operations=ResolvedOperations(
                    upsert_operations=[case_op(canonical_uri, "duplicate_booking")],
                    delete_file_contents=[],
                    errors=[],
                ),
                apply_result=result,
            )

    compressor = SessionCompressorV3(
        vikingdb=None,
        vlm_resolver=SimpleNamespace(get_vlm=AsyncMock(return_value=SimpleNamespace(model="test"))),
    )
    compressor._get_or_create_react = lambda **kwargs: DummyOrchestrator()

    async def fake_train_from_extracted_cases(**kwargs):
        trained_kwargs.append(kwargs)
        return {"case_count": len(kwargs["cases"]), "submitted": len(kwargs["cases"])}

    compressor.train_from_extracted_cases = fake_train_from_extracted_cases

    monkeypatch.setattr("openviking.session.compressor_v3.get_viking_fs", lambda: FakeFS())
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_default_registry",
        lambda: DummyRegistry(),
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_streaming_memory_updater",
        AsyncMock(return_value=FakeStreamingUpdater()),
    )

    contexts = await compressor.extract_long_term_memories(
        messages=_messages(),
        ctx=_ctx(),
        allowed_memory_types={"cases", "profile", "trajectories"},
    )

    assert [context.uri for context in contexts] == [canonical_uri]
    assert len(trained_kwargs) == 1
    assert [case.name for case in trained_kwargs[0]["cases"]] == ["duplicate_booking"]
    assert trained_kwargs[0]["cases"][0].metadata["case_uris"] == [canonical_uri]
    assert trained_kwargs[0]["case_uri_by_name"] == {"duplicate_booking": canonical_uri}


def _training_case() -> Case:
    return Case(
        name="duplicate_booking",
        task_signature="Handle duplicate bookings safely.",
        input={"summary": "cancel only the duplicate booking", "task_id": "task-1"},
        rubric=Rubric(
            name="duplicate_booking_rubric",
            description="Verify duplicates before cancellation.",
            criteria=[
                RubricCriterion(
                    name="verify_duplicate",
                    description="The assistant verifies which booking is the duplicate before acting.",
                    required=True,
                    weight=1.0,
                )
            ],
        ),
        metadata={"evidence": "The rollout contains duplicate-booking handling evidence."},
    )


def _case_spec_message(case: Case | None = None) -> Message:
    rollout = Rollout(
        case=case or _training_case(),
        messages=[],
        policy_snapshot_id="snapshot-1",
    )
    request = _case_spec_message_to_request(rollout)
    return Message(
        id="case-spec",
        role="system",
        parts=[TextPart(text=request["parts"][0]["text"])],
    )


@pytest.mark.asyncio
async def test_training_task_uses_shared_case_extraction_and_keeps_concrete_spec():
    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=SimpleNamespace())
    case = _training_case()
    case.name = "shared_booking_scenario"
    case_uri = f"viking://user/u/memories/cases/{case.name}.md"
    result = MemoryUpdateResult()
    result.add_written(case_uri)
    compressor._extract_user_memories = AsyncMock(
        return_value=SimpleNamespace(
            contexts=_contexts_from_update_result(result),
            cases=[case],
            case_uri_by_name={case.name: case_uri},
            memory_diff=None,
        )
    )
    compressor.train_from_extracted_cases = AsyncMock(return_value={"submitted": 1})
    compressor._write_final_memory_diff = AsyncMock()
    evaluation = Message(
        id="eval", role="system", parts=[TextPart(text="# OpenViking OutcomeEvaluation\n{}")]
    )
    messages = [_case_spec_message(), *_messages(), evaluation]
    contexts = await compressor.extract_long_term_memories(
        messages=messages,
        ctx=_ctx(),
        allowed_memory_types={"cases", "trajectories"},
    )
    assert contexts[0].uri == case_uri
    assert compressor._extract_user_memories.await_args.kwargs["messages"] == messages[1:-1]
    evidence = compressor._extract_user_memories.await_args.kwargs["case_rubric_evidence"]
    assert len(evidence) == 1  # The empty OutcomeEvaluation has no report to expose.
    payload = json.loads(evidence[0])
    assert payload["rubric"]["description"] == _training_case().rubric.description
    assert "case" not in payload and "name" not in payload["rubric"]
    assert "duplicate_booking" not in evidence[0]
    assert "task-1" not in evidence[0]
    trained = compressor.train_from_extracted_cases.await_args.kwargs
    assert trained["cases"] == [case]
    assert trained["task"].name == "duplicate_booking"
    assert trained["task"].input["task_id"] == "task-1"
    assert trained["task"].name != trained["cases"][0].name
    assert trained["messages"] == messages


@pytest.mark.asyncio
async def test_v3_training_case_spec_does_not_write_case_when_evolution_disabled():
    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=SimpleNamespace())
    compressor._extract_user_memories = AsyncMock(
        return_value=SimpleNamespace(
            contexts=[],
            cases=[],
            memory_diff=None,
            case_uri_by_name={},
        )
    )
    compressor.train_from_extracted_cases = AsyncMock()
    compressor._write_final_memory_diff = AsyncMock()

    contexts = await compressor.extract_long_term_memories(
        messages=[_case_spec_message(), *_messages()],
        ctx=_ctx(),
        session_id="s1",
        archive_uri="viking://user/u/sessions/s1/history/archive_001",
        allowed_memory_types={"cases", "trajectories"},
        agent_evolution_enabled=False,
    )

    assert contexts == []
    compressor.train_from_extracted_cases.assert_not_awaited()
    assert compressor._extract_user_memories.await_args.kwargs["allowed_memory_types"] == set()
    assert compressor._extract_user_memories.await_args.kwargs["case_rubric_evidence"] == []


@pytest.mark.asyncio
async def test_v3_training_case_spec_respects_user_memory_policy():
    extracted = False
    trained = []

    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=SimpleNamespace())

    async def fake_extract_user_memories(**kwargs):
        nonlocal extracted
        extracted = True
        return SimpleNamespace(contexts=[], cases=[])

    async def fake_train_from_extracted_cases(**kwargs):
        trained.append(kwargs)
        return {"case_count": 0, "submitted": 0}

    compressor._extract_user_memories = fake_extract_user_memories
    compressor.train_from_extracted_cases = fake_train_from_extracted_cases

    contexts = await compressor.extract_long_term_memories(
        messages=[_case_spec_message(), *_messages()],
        ctx=_ctx(),
        allowed_memory_types={"cases", "profile"},
    )

    assert contexts == []
    assert extracted is True
    assert trained == []


@pytest.mark.asyncio
async def test_v3_training_case_spec_rejects_invalid_protocol():
    message = _case_spec_message()
    assert isinstance(message.parts[0], TextPart)
    message.parts[0].text = message.parts[0].text.replace(
        "openviking.batch_train.case_spec.v1",
        "openviking.batch_train.case_spec.v0",
    )
    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=SimpleNamespace())

    with pytest.raises(ValueError, match="protocol mismatch"):
        await compressor.extract_long_term_memories(
            messages=[message, *_messages()],
            ctx=_ctx(),
            allowed_memory_types={"cases", "trajectories"},
        )


def test_training_case_spec_message_preserves_transport_protocol():
    message = _case_spec_message()
    part = message.parts[0]
    assert isinstance(part, TextPart)
    text = part.text

    assert text.startswith("# OpenViking Batch Training CaseSpec v1")
    assert "openviking.batch_train.case_spec.v1" in text
    assert "duplicate_booking_rubric" in text


def test_training_case_spec_message_uses_original_case_name_for_trials():
    case = _training_case()
    case.name = "tau2_airline_train_1_t0"
    case.task_signature = "tau2:airline:train:1:trial:0"
    case.input.update(
        {
            "domain": "airline",
            "split": "train",
            "data_split": "airline_train",
            "task_id": "1",
            "task_no": 1,
            "train_trial": 0,
            "original_case_name": "tau2_airline_train_1",
        }
    )
    message = _case_spec_message(case)
    payload = __import__(
        "openviking.session.compressor_v3", fromlist=["_training_case_spec_payload_from_message"]
    )._training_case_spec_payload_from_message(message)

    assert payload["case"]["name"] == "tau2_airline_train_1"
    assert payload["case"]["task_signature"] == "tau2:airline:train:1"
    assert payload["case"]["metadata"]["rollout_case_name"] == "tau2_airline_train_1_t0"


def test_training_outcome_carries_compact_experience_execution():
    experience_uri = "viking://user/u/memories/experiences/cancel.md"
    rollout = Rollout(
        case=_training_case(),
        messages=[],
        policy_snapshot_id="snapshot-1",
        metadata={
            "dag_runtime": {
                "session_id": "runtime-1",
                "events": [
                    {
                        "experience_uri": experience_uri,
                        "state": "running",
                        "revision": 2,
                        "executed_nodes": [1],
                        "current_nodes": [2],
                        "node_slots": {1: "known", 2: "report"},
                        "actions": [{"node_id": 2, "slot_name": "report"}],
                        "evidence": [{"id": "private", "summary": "must not be transported"}],
                    }
                ],
            }
        },
    )
    case_request = _case_spec_message_to_request(rollout)
    case_message = Message(
        id="case-spec",
        role="system",
        parts=[TextPart(text=case_request["parts"][0]["text"])],
    )
    case_payload = __import__(
        "openviking.session.compressor_v3", fromlist=["_training_case_spec_payload_from_message"]
    )._training_case_spec_payload_from_message(case_message)
    assert "experience_execution" not in case_payload["case"]["metadata"]

    outcome_request = _evaluation_message_to_request(rollout)
    outcome_message = Message(
        id="evaluation",
        role="user",
        parts=[TextPart(text=outcome_request["parts"][0]["text"])],
    )
    _evaluation, execution = _training_outcome_from_messages([outcome_message])

    assert execution[experience_uri]["executed_nodes"] == ["known"]
    assert execution[experience_uri]["current_nodes"] == ["report"]
    assert "evidence" not in execution[experience_uri]


def test_training_outcome_evaluation_round_trips_to_server_rollout():
    evaluation = RubricEvaluation(
        passed=False,
        score=0.25,
        criterion_results=[
            CriterionResult(
                criterion_name="communicate_total",
                passed=False,
                score=0.0,
                feedback=["Information 1628 was not communicated."],
                evidence=["assistant said 708"],
                metadata={"reward_basis": "COMMUNICATE"},
            )
        ],
        feedback=["tau2 reward is below 1.0"],
        metadata={"reward": 0.25},
    )
    rollout = Rollout(
        case=_training_case(),
        messages=[],
        policy_snapshot_id="snapshot-1",
        evaluation=evaluation,
    )
    request = _evaluation_message_to_request(rollout)
    message = Message(
        id="evaluation",
        role="user",
        parts=[TextPart(text=request["parts"][0]["text"])],
    )

    restored = _training_evaluation_from_messages([message])

    assert restored == evaluation


@pytest.mark.asyncio
async def test_v3_shared_case_path_writes_final_memory_diff_with_case_and_exp(monkeypatch):
    archive_uri = "viking://user/u/sessions/s1/history/archive_001"
    writes: dict[str, str] = {}

    class FakeFS:
        async def write_file(self, uri, content, ctx=None):
            del ctx
            writes[uri] = content

        async def read_file(self, uri, ctx=None):
            del ctx
            if uri.endswith("/cases/duplicate_booking.md"):
                return "# duplicate_booking\n\n<!-- MEMORY_FIELDS\n{}\n-->"
            if uri.endswith("/experiences/booking_duplicate_handling.md"):
                return "new exp content\n\n<!-- MEMORY_FIELDS\n{}\n-->"
            raise FileNotFoundError(uri)

    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=SimpleNamespace())

    async def fake_extract_user_memories(**kwargs):
        result = MemoryUpdateResult()
        result.add_written("viking://user/u/memories/cases/duplicate_booking.md")
        return SimpleNamespace(
            contexts=_contexts_from_update_result(result),
            cases=[_training_case()],
            case_uri_by_name={},
            memory_diff={
                "archive_uri": archive_uri,
                "trace_id": None,
                "extracted_at": "now",
                "operations": {
                    "adds": [
                        {
                            "uri": "viking://user/u/memories/cases/duplicate_booking.md",
                            "memory_type": "cases",
                            "after": "case",
                        }
                    ],
                    "updates": [],
                    "deletes": [],
                },
                "skipped_operations": [
                    {
                        "memory_type": "preferences",
                        "page_id": 102,
                        "reason_code": "peer_not_allowed",
                        "reason": "Target peer is outside the allowed memory scope",
                    }
                ],
                "summary": {
                    "total_adds": 1,
                    "total_updates": 0,
                    "total_deletes": 0,
                    "total_skipped": 1,
                },
            },
        )

    async def fake_train_from_extracted_cases(**kwargs):
        return {
            "case_count": 1,
            "submitted": 1,
            "memory_diff": {
                "archive_uri": archive_uri,
                "trace_id": None,
                "extracted_at": "now",
                "operations": {
                    "adds": [],
                    "updates": [
                        {
                            "uri": "viking://user/u/memories/experiences/booking_duplicate_handling.md",
                            "memory_type": "experiences",
                            "before": "old exp content",
                            "after": "new exp content",
                        }
                    ],
                    "deletes": [],
                },
                "summary": {"total_adds": 0, "total_updates": 1, "total_deletes": 0},
            },
        }

    compressor._extract_user_memories = fake_extract_user_memories
    compressor.train_from_extracted_cases = fake_train_from_extracted_cases
    monkeypatch.setattr("openviking.session.compressor_v3.get_viking_fs", lambda: FakeFS())

    contexts = await compressor.extract_long_term_memories(
        messages=[_case_spec_message(), *_messages()],
        ctx=_ctx(),
        session_id="s1",
        archive_uri=archive_uri,
        allowed_memory_types={"cases", "trajectories"},
    )

    assert contexts[0].uri.endswith("/cases/duplicate_booking.md")
    diff = __import__("json").loads(writes[f"{archive_uri}/memory_diff.json"])
    assert [item["memory_type"] for item in diff["operations"]["adds"]] == [
        "cases",
    ]
    assert [item["memory_type"] for item in diff["operations"]["updates"]] == ["experiences"]
    assert diff["skipped_operations"] == [
        {
            "memory_type": "preferences",
            "page_id": 102,
            "reason_code": "peer_not_allowed",
            "reason": "Target peer is outside the allowed memory scope",
        }
    ]
    assert diff["summary"] == {
        "total_adds": 1,
        "total_updates": 1,
        "total_deletes": 0,
        "total_skipped": 1,
    }


@pytest.mark.asyncio
async def test_v3_builds_training_memory_diff_from_streaming_result(monkeypatch):
    archive_uri = "viking://user/u/sessions/s1/history/archive_001"

    class FakeFS:
        async def read_file(self, uri, ctx=None):
            del ctx
            assert uri.endswith("/experiences/booking_duplicate_handling.md")
            return "new exp content\n\n<!-- MEMORY_FIELDS\n{}\n-->"

    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=SimpleNamespace())
    plan = PolicyUpdatePlan(
        items=[
            PolicyPlanItem(
                kind="upsert",
                memory_type="experiences",
                target_name="booking_duplicate_handling",
                target_uri="viking://user/u/memories/experiences/booking_duplicate_handling.md",
                before_content="old exp content",
                after_content="new exp content fallback",
                links=[
                    StoredLink(
                        from_uri="viking://user/u/memories/experiences/booking_duplicate_handling.md",
                        to_uri="viking://user/u/sessions/duplicate_booking/history/archive_001",
                        link_type="derived_from",
                        weight=1.0,
                    )
                ],
            )
        ]
    )
    training_result = RolloutTrainingResult(
        analyses=[
            RolloutAnalysis(
                evaluation=RubricEvaluation(
                    passed=True,
                    score=1.0,
                    criterion_results=[],
                    feedback=[],
                ),
                trajectories=[],
                metadata={
                    "source_session_uri": "viking://user/u/sessions/duplicate_booking/history/archive_001"
                },
            )
        ],
        gradients=[],
        plan=plan,
        apply_result=PolicyApplyResult(
            updated_policy_set=ExperienceSet(
                root_uri="viking://user/u/memories/experiences",
                policies=[],
            ),
            written_uris=["viking://user/u/memories/experiences/booking_duplicate_handling.md"],
        ),
    )

    diff = await compressor._build_training_memory_diff(
        training_result=training_result,
        viking_fs=FakeFS(),
        ctx=_ctx(),
        archive_uri=archive_uri,
    )

    assert diff["summary"] == {
        "total_adds": 0,
        "total_updates": 1,
        "total_deletes": 0,
        "total_skipped": 0,
    }
    assert diff["operations"]["adds"] == []
    update = diff["operations"]["updates"][0]
    assert update["memory_type"] == "experiences"
    assert update["before"] == "old exp content"
    assert update["after"] == "new exp content"


@pytest.mark.asyncio
async def test_v3_training_memory_diff_filters_batch_items_by_source_session(
    monkeypatch,
):
    archive_uri = "viking://user/u/sessions/s1/history/archive_001"
    traj_a = "viking://user/u/sessions/traj_a/history/archive_001"
    traj_b = "viking://user/u/sessions/traj_b/history/archive_001"
    exp_a = "viking://user/u/memories/experiences/exp_a.md"
    exp_b = "viking://user/u/memories/experiences/exp_b.md"

    class FakeFS:
        async def read_file(self, uri, ctx=None):
            del ctx
            return {
                exp_a: "exp a\n\n<!-- MEMORY_FIELDS\n{}\n-->",
                exp_b: "exp b\n\n<!-- MEMORY_FIELDS\n{}\n-->",
            }[uri]

    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=SimpleNamespace())
    training_result = RolloutTrainingResult(
        analyses=[
            RolloutAnalysis(
                evaluation=RubricEvaluation(
                    passed=True, score=1.0, criterion_results=[], feedback=[]
                ),
                trajectories=[],
                metadata={"source_session_uri": traj_a},
            )
        ],
        gradients=[],
        plan=PolicyUpdatePlan(
            items=[
                PolicyPlanItem(
                    kind="upsert",
                    memory_type="experiences",
                    target_name="exp_a",
                    target_uri=exp_a,
                    before_content=None,
                    after_content="exp a fallback",
                    links=[
                        StoredLink(
                            from_uri=exp_a,
                            to_uri=traj_a,
                            link_type="derived_from",
                            weight=1.0,
                        )
                    ],
                ),
                PolicyPlanItem(
                    kind="upsert",
                    memory_type="experiences",
                    target_name="exp_b",
                    target_uri=exp_b,
                    before_content=None,
                    after_content="exp b fallback",
                    links=[
                        StoredLink(
                            from_uri=exp_b,
                            to_uri=traj_b,
                            link_type="derived_from",
                            weight=1.0,
                        )
                    ],
                ),
            ]
        ),
        apply_result=PolicyApplyResult(
            updated_policy_set=ExperienceSet(
                root_uri="viking://user/u/memories/experiences",
                policies=[],
            ),
            written_uris=[exp_a, exp_b],
        ),
    )

    diff = await compressor._build_training_memory_diff(
        training_result=training_result,
        viking_fs=FakeFS(),
        ctx=_ctx(),
        archive_uri=archive_uri,
    )

    assert diff["summary"] == {
        "total_adds": 1,
        "total_updates": 0,
        "total_deletes": 0,
        "total_skipped": 0,
    }
    assert [op["uri"] for op in diff["operations"]["adds"]] == [exp_a]


@pytest.mark.parametrize("trajectory_links", [False, True])
def test_v3_maps_each_written_experience_to_its_source_sessions(trajectory_links):
    traj_a = "viking://user/u/sessions/traj_a/history/archive_001"
    traj_b = "viking://user/u/sessions/traj_b/history/archive_001"
    exp_a = "viking://user/u/memories/experiences/exp_a.md"
    exp_b = "viking://user/u/memories/experiences/exp_b.md"
    plan = PolicyUpdatePlan(
        items=[
            PolicyPlanItem(
                kind="upsert",
                memory_type="experiences",
                target_name="exp_a",
                target_uri=exp_a,
                before_content=None,
                after_content="exp a",
                links=[
                    StoredLink(
                        from_uri=exp_a,
                        to_uri=traj_a,
                        link_type="derived_from",
                        weight=1.0,
                    )
                ],
            ),
            PolicyPlanItem(
                kind="upsert",
                memory_type="experiences",
                target_name="exp_b",
                target_uri=exp_b,
                before_content=None,
                after_content="exp b",
                links=[
                    StoredLink(
                        from_uri=exp_b,
                        to_uri=traj_b,
                        link_type="derived_from",
                        weight=1.0,
                    )
                ],
            ),
        ]
    )
    if trajectory_links:
        for item, source_uri in zip(plan.items, (traj_a, traj_b), strict=True):
            item.metadata["source_sessions"] = [{"source_session_uri": source_uri}]
            item.links[0] = item.links[0].model_copy(
                update={"to_uri": f"viking://user/u/memories/trajectories/{item.target_name}/t.md"}
            )
    apply_result = PolicyApplyResult(
        updated_policy_set=ExperienceSet(
            root_uri="viking://user/u/memories/experiences",
            policies=[],
        ),
        written_uris=[exp_a, exp_b],
    )

    assert _experience_session_map(
        plan=plan,
        apply_result=apply_result,
        session_uris={traj_a, traj_b},
    ) == {
        exp_a: [traj_a],
        exp_b: [traj_b],
    }


def test_v3_snapshot_provenance_uses_complete_shared_batch_result():
    traj_a = "viking://user/u/sessions/traj_a/history/archive_001"
    traj_b = "viking://user/u/sessions/traj_b/history/archive_001"
    exp_a = "viking://user/u/memories/experiences/exp_a.md"
    exp_b = "viking://user/u/memories/experiences/exp_b.md"

    def plan_item(experience_uri: str, trajectory_uri: str) -> PolicyPlanItem:
        return PolicyPlanItem(
            kind="upsert",
            memory_type="experiences",
            target_name=experience_uri.rsplit("/", 1)[-1].removesuffix(".md"),
            target_uri=experience_uri,
            before_content=None,
            after_content="updated",
            links=[
                StoredLink(
                    from_uri=experience_uri,
                    to_uri=trajectory_uri,
                    link_type="derived_from",
                    weight=1.0,
                )
            ],
        )

    root = "viking://user/u/memories/experiences"
    batch_result = SimpleNamespace(
        analyses=[
            SimpleNamespace(metadata={"source_session_uri": traj_a}),
            SimpleNamespace(metadata={"source_session_uri": traj_b}),
        ],
        plan=PolicyUpdatePlan(items=[plan_item(exp_a, traj_a), plan_item(exp_b, traj_b)]),
        apply_result=PolicyApplyResult(
            updated_policy_set=ExperienceSet(root_uri=root, policies=[]),
            written_uris=[exp_a, exp_b],
        ),
    )
    scoped_result = SimpleNamespace(
        analyses=[batch_result.analyses[0]],
        plan=PolicyUpdatePlan(items=[batch_result.plan.items[0]]),
        apply_result=PolicyApplyResult(
            updated_policy_set=ExperienceSet(root_uri=root, policies=[]),
            written_uris=[exp_a],
        ),
        batch_result=batch_result,
    )

    apply_result, trajectory_map = _experience_snapshot_provenance(scoped_result)

    assert apply_result is batch_result.apply_result
    assert trajectory_map == {
        exp_a: [traj_a],
        exp_b: [traj_b],
    }


def test_visible_experience_snapshot_uris_only_returns_applied_body_changes():
    root = "viking://user/u/memories/experiences"
    changed_uri = f"{root}/changed.md"
    metadata_only_uri = f"{root}/metadata_only.md"
    failed_uri = f"{root}/failed.md"
    deleted_uri = f"{root}/deleted.md"
    content_write_uri = f"{root}/content_write.md"
    plan = PolicyUpdatePlan(
        items=[
            PolicyPlanItem(
                kind="upsert",
                memory_type="experiences",
                target_name="changed",
                target_uri=changed_uri,
                before_content="before",
                after_content="after",
            ),
            PolicyPlanItem(
                kind="upsert",
                memory_type="experiences",
                target_name="metadata_only",
                target_uri=metadata_only_uri,
                before_content="unchanged",
                after_content="unchanged",
            ),
            PolicyPlanItem(
                kind="upsert",
                memory_type="experiences",
                target_name="failed",
                target_uri=failed_uri,
                before_content="before",
                after_content="after",
            ),
            PolicyPlanItem(
                kind="delete",
                memory_type="experiences",
                target_name="deleted",
                target_uri=deleted_uri,
                before_content="old content",
                after_content=None,
            ),
        ]
    )
    apply_result = PolicyApplyResult(
        updated_policy_set=ExperienceSet(root_uri=root, policies=[]),
        written_uris=[changed_uri, metadata_only_uri, content_write_uri],
        deleted_uris=[deleted_uri],
    )

    assert _visible_experience_snapshot_uris(
        plan=plan,
        apply_result=apply_result,
    ) == [changed_uri, deleted_uri]


@pytest.mark.asyncio
async def test_commit_experience_snapshot_skips_when_no_visible_content_changed():
    viking_fs = SimpleNamespace(commit=AsyncMock())

    await _commit_experience_snapshot(
        viking_fs,
        ctx=_ctx(),
        experience_uris=[],
        archive_uri="viking://user/u/sessions/session-1/history/archive_001",
    )

    viking_fs.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_case_learning_records_trajectory_diff_before_synthesis(monkeypatch):
    from openviking.session.train.components.trajectory_observation import ObservationGradient

    archive = "viking://user/u/sessions/s1/history/archive_001"
    root = "viking://user/u/memories/experiences"
    memory = MemoryFile(
        uri="viking://user/u/memories/trajectories/duplicate_booking/run.md",
        memory_type="trajectories",
        content='{"uncertainties": ["unknown cause"]}',
    )

    async def observe(self, analyses, context):
        (analysis,) = analyses
        assert context.memory_registry is not None
        analysis.metadata["trajectory_memory"] = memory
        analysis.trajectories = [
            Trajectory(name="run", uri=memory.uri, content=memory.content, outcome="unknown")
        ]

    async def submit(gradients, *, analysis, rollout, batch_finalizer):
        assert isinstance(gradients[0], ObservationGradient)
        return RolloutTrainingResult(
            analyses=[analysis],
            gradients=gradients,
            plan=PolicyUpdatePlan(items=[]),
            apply_result=PolicyApplyResult(updated_policy_set=ExperienceSet(root, [])),
        )

    monkeypatch.setattr(
        "openviking.session.compressor_v3.SessionRolloutAnalyzer._extract_trajectories", observe
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_viking_fs",
        lambda: SimpleNamespace(ls=AsyncMock(return_value=[])),
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_streaming_policy_trainer",
        AsyncMock(
            return_value=SimpleNamespace(
                policy_set=ExperienceSet(root, []), submit_gradients=submit
            )
        ),
    )
    compressor = SessionCompressorV3(vikingdb=None)
    compressor._session_skill_extraction_enabled = lambda: False
    result = await compressor.train_from_extracted_cases(
        cases=[_training_case()],
        messages=_messages(),
        ctx=_ctx(),
        archive_uri=archive,
        collect_memory_diff=True,
        allowed_memory_types={"trajectories", "experiences"},
    )
    assert result["submitted"] == 1
    assert result["memory_diff"]["operations"]["adds"] == [
        {"uri": memory.uri, "memory_type": "trajectories", "after": memory.content}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("path", ["ordinary", "training"])
async def test_extraction_failures_follow_strict_mode(monkeypatch, strict, path):
    from openviking.session.compressor_v3 import ExperienceTrainingError
    from openviking.telemetry import tracer

    compressor = SessionCompressorV3(vikingdb=None)
    error = (
        RuntimeError("parse failed")
        if path == "ordinary"
        else ExperienceTrainingError("parse failed")
    )
    compressor._extract_user_memories = AsyncMock(
        side_effect=error if path == "ordinary" else None,
        return_value=SimpleNamespace(
            contexts=[], cases=[_training_case()], case_uri_by_name={}, memory_diff=None
        ),
    )
    compressor.train_from_extracted_cases = AsyncMock(side_effect=error)
    trace_error = Mock()
    monkeypatch.setattr(tracer, "error", trace_error)
    call = compressor.extract_long_term_memories(
        messages=_messages(),
        ctx=_ctx(),
        allowed_memory_types={"cases", "trajectories"},
        strict_extract_errors=strict,
    )
    if strict:
        with pytest.raises(type(error)) as raised:
            await call
        assert raised.value is error
    else:
        assert await call == {"contexts": [], "session_skills": []}
        trace_error.assert_called_once_with(
            "V3 memory extraction failed; returning empty result",
            e=error,
            console=False,
        )


@pytest.mark.asyncio
async def test_enabled_skills_run_once_independently_of_case_count(monkeypatch):
    from openviking.session.train.components.session_analyzer import SessionRolloutAnalyzer

    fs = SimpleNamespace(ls=AsyncMock(return_value=[]))
    trainer = SimpleNamespace(policy_set=ExperienceSet("viking://user/u/memories/experiences", []))
    monkeypatch.setattr("openviking.session.compressor_v3.get_viking_fs", lambda: fs)
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_streaming_policy_trainer",
        AsyncMock(return_value=trainer),
    )
    monkeypatch.setattr(
        SessionRolloutAnalyzer, "_extract_trajectories", AsyncMock(return_value=None)
    )
    compressor = SessionCompressorV3(vikingdb=None)
    compressor._session_skill_extraction_enabled = lambda: True
    compressor.extract_session_skills = AsyncMock(
        return_value={"skill_submitted": 1, "skill_uris": ["skill-uri"]}
    )
    other_case = _training_case()
    other_case.name = "other_case"
    result = await compressor.train_from_extracted_cases(
        cases=[_training_case(), other_case],
        messages=_messages(),
        ctx=_ctx(),
        archive_uri="viking://user/u/sessions/s/history/archive_001",
    )
    compressor.extract_session_skills.assert_awaited_once()
    assert result["submitted"] == 0
    assert result["skill_submitted"] == 1
    assert result["skill_uris"] == ["skill-uri"]


@pytest.mark.asyncio
async def test_lenient_extraction_does_not_swallow_cancellation():
    compressor = SessionCompressorV3(vikingdb=None)
    compressor._extract_user_memories = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await compressor.extract_long_term_memories(messages=_messages(), ctx=_ctx())


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", [None, "deleted", "filtered", "failed"])
async def test_known_case_noop_contributes_without_accepting_failed_or_filtered_targets(blocked):
    op = _case_operation()
    uri = op.uris[0]
    raw = MemoryFileUtils.write(
        MemoryFile(uri=uri, memory_type="cases", content="", extra_fields=op.memory_fields)
    )
    fs = SimpleNamespace(read_file=AsyncMock(return_value=raw))
    result = MemoryUpdateResult()
    if blocked == "deleted":
        result.deleted_uris.append(uri)
    elif blocked == "filtered":
        result.skipped_operations.append(
            SkippedMemoryOperation(
                memory_type="cases",
                uri=uri,
                reason_code=MemoryOperationSkipCode.MEMORY_TYPE_FILTERED,
                reason="filtered",
            )
        )
    elif blocked == "failed":
        result.errors.append((uri, "failed"))
    cases = await _canonical_cases_from_update_result(
        operations=ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[]),
        requested_operations=ResolvedOperations(
            upsert_operations=[op], delete_file_contents=[], errors=[]
        ),
        result=result,
        viking_fs=fs,
        ctx=_ctx(),
    )
    assert [case.name for case in cases] == ([] if blocked else [op.memory_fields["case_name"]])
    if blocked:
        fs.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_case_association_does_not_treat_storage_failure_as_absence():
    op = _case_operation()
    with pytest.raises(PermissionError, match="denied"):
        await _canonical_cases_from_update_result(
            operations=ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[]),
            requested_operations=ResolvedOperations(
                upsert_operations=[op], delete_file_contents=[], errors=[]
            ),
            result=MemoryUpdateResult(),
            viking_fs=SimpleNamespace(read_file=AsyncMock(side_effect=PermissionError("denied"))),
            ctx=_ctx(),
        )
