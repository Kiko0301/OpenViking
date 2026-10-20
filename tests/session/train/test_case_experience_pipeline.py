from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from test_gradient_estimator_component import ROOT, analysis, operations, validated_operations
from test_train_components import FakeVikingFS

from openviking.message import Message, TextPart, ToolPart
from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v3 import SessionCompressorV3
from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.experience.experience_context import (
    AgentExperienceContextProvider,
    _trajectory_model_content,
)
from openviking.session.memory.experience.trajectory_context import trajectory_source_steps
from openviking.session.memory.memory_type_registry import get_default_registry
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.memory.utils.uri import generate_uri
from openviking.session.train import (
    ExperienceGradientContext,
    ExperienceGradientEstimator,
    ExperienceSet,
    MemoryFilePolicyUpdater,
    PatchMergePolicyOptimizerContext,
    PipelineContext,
    Rubric,
    RubricCriterion,
    SessionAnalyzerContext,
    SessionRolloutAnalyzer,
    StreamingPolicyTrainer,
    StreamingPolicyTrainerConfig,
)
from openviking.session.train.components.trajectory_lifecycle import (
    archive_trajectory,
    trajectory_archive_uri,
)
from openviking.session.train.components.trajectory_observation import (
    TRAJECTORY_FORMAT,
    ObservationGradient,
    observation_evidence,
    trajectory_source_session_uri,
)
from openviking.session.train.components.trajectory_optimizer import (
    TrajectoryExperienceOptimizer,
)
from openviking_cli.session.user_id import UserIdentifier
from tests.session.train.test_fakes import render_trajectory_template

OBSERVATION = "1. `user` requested cancellation of the duplicate booking"
STORED_OBSERVATION = "# Session Context\n\n\n\n# Trajectory\n\n1. `user` requested cancellation of the duplicate booking"


def trajectory_program(
    steps: str = OBSERVATION,
    *,
    context_rules: str = "",
    case_name: str = "case_a",
    evaluation_criteria: list[dict] | None = None,
) -> str:
    criteria = (
        f", evaluation_criteria={json.dumps(evaluation_criteria)!r}"
        if evaluation_criteria is not None
        else ""
    )
    return f"sdk.create_trajectories(case_name={case_name!r}, context_rules={context_rules!r}, steps={steps!r}{criteria})\nsdk.commit()"


async def presented_trajectories(provider):
    messages = provider.format_prefetch_messages(await provider.prefetch())
    records = []
    for message in messages:
        if message["content"].startswith("# Trajectory Record\n"):
            fields, content = message["content"].split("```yaml\n", 1)[1].split("\n```\n\n", 1)
            records.append({**yaml.safe_load(fields), "content": message["content"]})
    return records


class ObservationFS(FakeVikingFS):
    def __init__(self):
        super().__init__({})
        self.locks = {}
        self.find = AsyncMock(return_value=SimpleNamespace(memories=[]))
        self.policy_lock = asyncio.Lock()
        self._async_agfs.pathlock_acquire_exact = self.acquire
        self._async_agfs.pathlock_acquire_exact_batch = self.acquire_tree
        self._async_agfs.pathlock_acquire_tree = self.acquire_tree
        self._async_agfs.pathlock_release = self.release

    async def acquire(self, path, **kwargs):
        lock = self.locks.setdefault(path, asyncio.Lock())
        await lock.acquire()
        return lock

    async def acquire_tree(self, *args, **kwargs):
        await self.policy_lock.acquire()
        return self.policy_lock

    async def release(self, lock):
        lock.release()

    async def read_file(self, uri, ctx=None):
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return self.files[uri]

    async def exists(self, uri, ctx=None):
        return uri in self.files

    async def mkdir(self, uri, **kwargs):
        pass

    async def cp(self, old_uri, new_uri, **kwargs):
        self.files[new_uri] = self.files[old_uri]


async def observe(analyzer, run, policies, context):
    """Persist via the public analyzer, then build the independent Phase 2 signal."""
    run.rollout.evaluation = run.evaluation
    run.rollout.metadata.update(run.metadata)
    result = await analyzer.analyze(
        run.rollout,
        SessionAnalyzerContext(
            request_context=context.request_context,
            strict_extract_errors=context.strict_extract_errors,
            source_archive_uri=run.metadata["source_session_uri"],
            memory_registry=context.memory_registry,
        ),
    )
    run.evaluation = result.evaluation
    run.trajectories = result.trajectories
    run.metadata.update(result.metadata)
    memory = result.metadata["trajectory_memory"]
    target = generate_uri(
        get_default_registry().get("experiences"),
        {"experience_name": run.rollout.case.name},
        user_space=context.request_context.user.user_id,
    )
    assert target.startswith(policies.root_uri.rstrip("/") + "/")
    return [ObservationGradient(run, run.rollout.case.name, target, memory)]


async def store_existing_experience(fs, policies, run):
    target = f"{ROOT}/{run.rollout.case.name}.md"
    fs.files[target] = MemoryFileUtils.write(
        MemoryFile(
            uri=target,
            content='dag = workflow("cancel")\nstep = tell("verify duplicate identity")',
            memory_type="experiences",
            extra_fields={
                "experience_name": run.rollout.case.name,
                "case_name": run.rollout.case.name,
                "version": 1,
            },
        )
    )
    return await policies.reload()


@pytest.fixture
def setup_observation():
    fs = ObservationFS()
    ctx = RequestContext(user=UserIdentifier("default", "u"), role=Role.ROOT)
    model = SimpleNamespace(
        model="test-vlm",
        get_completion_async=AsyncMock(return_value=trajectory_program()),
    )
    analyzer = SessionRolloutAnalyzer(viking_fs=fs, vlm=model)
    context = ExperienceGradientContext(ctx, [])
    policies = ExperienceSet(ROOT, [], viking_fs=fs, request_context=ctx)
    return fs, ctx, model, analyzer, context, policies


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rubric_results", "expected_ids"),
    [
        (None, []),
        ('[{"id":"confirm","passed":true,"evidence":"{"nested":true}"}]', []),
        ([{"id": "tau2_reward", "passed": True, "evidence": "External result."}], ["tau2_reward"]),
        (
            [{"id": "tau2_reward", "passed": False, "evidence": "Required information missing."}],
            ["tau2_reward"],
        ),
        (
            [{"id": "tau2_reward", "passed": None, "evidence": "No sufficient evidence."}],
            ["tau2_reward"],
        ),
        (
            [
                {
                    "id": "confirm",
                    "description": "Model-assessed confirmation requirement.",
                    "passed": True,
                    "evidence": "Confirmation observed.",
                },
                {"id": "scope", "passed": "false", "evidence": "Wrong type."},
                {"id": "tau2_reward", "passed": True, "evidence": "Additional assessment."},
            ],
            ["confirm", "tau2_reward"],
        ),
    ],
    ids=[
        "omitted",
        "malformed-json",
        "unmatched-success",
        "unmatched-failure",
        "unmatched-unknown",
        "partial",
    ],
)
async def test_bad_or_missing_rubric_results_do_not_block_trajectory_persistence(
    setup_observation, rubric_results, expected_ids
):
    fs, _ctx, model, analyzer, context, policies = setup_observation
    run = analysis(passed=False)
    run.rollout.case.rubric = Rubric(
        "case_rubric",
        "",
        [
            RubricCriterion("confirm", "Confirm before changing the record.", True, 1.0),
            RubricCriterion("scope", "Only change the requested fields.", True, 1.0),
        ],
    )
    program = trajectory_program().replace("outcome='unknown'", "outcome='success'")
    if rubric_results is not None:
        value = rubric_results if isinstance(rubric_results, str) else json.dumps(rubric_results)
        program = program.replace("steps=", f"evaluation_criteria={value!r}, steps=", 1)
    model.get_completion_async.return_value = program

    signals = await observe(analyzer, run, policies, context)

    model.get_completion_async.assert_awaited_once()
    assert len(signals) == len(run.trajectories) == 1
    memory = MemoryFileUtils.read(fs.files[run.trajectories[0].uri])
    header = yaml.safe_load(memory.content.split("```yaml\n", 1)[1].split("\n```", 1)[0])
    assert memory.content.endswith(STORED_OBSERVATION)
    assert header["outcome"] == "failure"
    assert "evaluation" not in header
    assert memory.extra_fields["evaluation"] == {"passed": False, "score": 0.0}
    snapshots = memory.extra_fields["evaluation_criteria"]
    assert [r["id"] for r in snapshots] == expected_ids
    assert header["evaluation_criteria"] == [r for r in snapshots if r["passed"] is False]
    assert memory.extra_fields["experience_update"]["state"] == "pending"
    if isinstance(rubric_results, list):
        assert snapshots == [
            item
            for item in rubric_results
            if item.get("passed") is None or type(item.get("passed")) is bool
        ]
    # Experience synthesis keeps unmatched failed assessments in its compact model view.
    projected = _trajectory_model_content(memory.content)
    model_header = yaml.safe_load(projected.split("```yaml\n", 1)[1].split("\n```", 1)[0])
    assert model_header["evaluation_criteria"] == header["evaluation_criteria"]


@pytest.mark.asyncio
async def test_observation_is_factual_new_on_each_call_and_server_owned(
    setup_observation, monkeypatch
):
    fs, ctx, model, analyzer, context, policies = setup_observation
    original_apply = MemoryUpdater.apply_operations
    apply_calls = []

    async def apply_operations(updater, operations, request_context, **kwargs):
        apply_calls.append(operations)
        return await original_apply(updater, operations, request_context, **kwargs)

    monkeypatch.setattr(MemoryUpdater, "apply_operations", apply_operations)
    run = analysis(passed=False)
    run.trajectories = []
    signals = await observe(analyzer, run, policies, context)
    record = run.trajectories[0]
    assert record.outcome == "failure"
    assert record.metadata["evaluation"]["passed"] is False
    assert record.metadata["source_session_uri"] == run.metadata["source_session_uri"]
    assert record.content.endswith(STORED_OBSERVATION)
    assert record.content.count("# Trajectory Record\n") == 1
    assert (
        yaml.safe_load(record.content.split("```yaml\n", 1)[1].split("\n```", 1)[0])["outcome"]
        == "failure"
    )
    assert "/trajectories/case_a/" in record.uri
    assert signals[0].target_uri == f"{ROOT}/case_a.md"
    stored = MemoryFileUtils.read(fs.files[record.uri], uri=record.uri)
    assert stored.links[0]["to_uri"] == run.metadata["source_session_uri"]
    assert trajectory_source_session_uri(stored) == run.metadata["source_session_uri"]
    assert stored.extra_fields.get("source_session_uri") is None
    assert stored.extra_fields.get("task_query") is None
    assert stored.extra_fields.get("evidence") is None
    assert stored.extra_fields["experience_update"]["state"] == "pending"
    assert stored.extra_fields["experience_update"]["attempts"] == 0
    assert stored.extra_fields["trajectory_format"] == TRAJECTORY_FORMAT
    assert "cited_evidence" not in stored.extra_fields
    assert len(stored.links) == 1 and stored.backlinks == []
    assert "experience_uri" not in stored.extra_fields["experience_update"]
    assert len(apply_calls) == 1
    assert apply_calls[0].upsert_operations[0].memory_type == "trajectories"
    assert apply_calls[0].delete_file_contents == []
    before = dict(fs.files)
    await observe(analyzer, run, policies, context)
    assert all(fs.files[uri] == raw for uri, raw in before.items())
    assert len(fs.files) == len(before) + 1
    assert model.get_completion_async.await_count == 2
    extraction_messages = model.get_completion_async.await_args.kwargs["messages"]
    assert "sdk.create_trajectories" in extraction_messages[0]["content"]
    assert "at most 800 characters" in extraction_messages[0]["content"]
    assert "source_steps" in extraction_messages[1]["content"]


def test_source_steps_preserve_user_assistant_and_tool_order():
    steps = trajectory_source_steps(
        [
            Message(id="u", role="user", parts=[TextPart(text="Cancel booking A")]),
            Message(id="a", role="assistant", parts=[TextPart(text="I will check it")]),
            Message(
                id="t",
                role="user",
                parts=[
                    ToolPart(
                        tool_id="tool",
                        tool_name="get_booking",
                        tool_input={"id": "A"},
                        tool_output='{"status":"active"}',
                        tool_status="completed",
                    )
                ],
            ),
        ]
    )

    assert [step["type"] for step in steps] == ["user", "assistant", "tool:get_booking"]
    assert "Cancel booking A" in steps[0]["content"]
    assert "I will check it" in steps[1]["content"]
    assert json.loads(steps[2]["content"])["tool_output"] == '{"status":"active"}'


@pytest.mark.asyncio
async def test_selected_session_rules_persist_and_reach_experience_input(setup_observation):
    fs, _ctx, model, analyzer, context, policies = setup_observation
    rules = "Verify eligibility before cancelling.\nException: duplicate bookings qualify."
    unrelated = "For new bookings, collect the destination."
    run = analysis(passed=False)
    run.rollout.messages.insert(
        0,
        Message(
            id="system",
            role="user",
            parts=[TextPart(text=f"system:\n{rules}\n\n{unrelated}")],
        ),
    )
    model.get_completion_async.return_value = trajectory_program(context_rules=rules)

    await observe(analyzer, run, policies, context)

    trajectory = run.trajectories[0]
    stored = MemoryFileUtils.read(fs.files[trajectory.uri], uri=trajectory.uri)
    assert stored.content.endswith(
        f"# Session Context\n\n{rules}\n\n"
        "# Trajectory\n\n1. `user` requested cancellation of the duplicate booking"
    )
    assert unrelated not in stored.content
    assert stored.extra_fields["context_rules"] == rules
    assert stored.extra_fields["steps"] == OBSERVATION
    assert "session_context" not in stored.extra_fields
    model.get_completion_async.assert_awaited_once()
    extraction_messages = model.get_completion_async.await_args.kwargs["messages"]
    payload = json.loads(extraction_messages[1]["content"])
    assert "session_context" not in payload
    assert payload["source_steps"] == [
        {"type": "user", "content": f"system:\n{rules}\n\n{unrelated}"},
        {"type": "user", "content": "Cancel the duplicate booking."},
    ]
    assert "wrongly refused" in extraction_messages[0]["content"]
    assert "confirmation requirements and exceptions" in extraction_messages[0]["content"]

    provider = AgentExperienceContextProvider(
        messages=[],
        case=run.rollout.case,
        target_uri=f"{ROOT}/case_a.md",
        source_session_uri=run.metadata["source_session_uri"],
        trajectories=[trajectory],
    )
    provider._eval_memory = AsyncMock(return_value={"experience": {"error": "File not found"}})
    record = (await presented_trajectories(provider))[0]
    assert record["outcome"] == "failure"
    assert record["content"].rstrip() == _trajectory_model_content(stored.content)
    assert "steps" not in record
    assert "context_rules" not in record
    assert record["experience_execution"] == trajectory.metadata["experience_execution"]
    assert (
        "each run against its own recorded request, rules and evaluation" in provider.instruction()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["edit", "drop", "overwrite"])
async def test_experience_request_has_separate_plain_user_messages_and_working_update_binding(
    setup_observation,
    method,
):
    fs, _ctx, model, _analyzer, context, policies = setup_observation
    target = f"{ROOT}/case_a.md"
    old_content = (
        'dag = workflow("cancel")\n'
        'requested = check("The user requests cancellation")\n'
        "reminder = tell(\n"
        '    "Verify identity and eligibility"\n'
        '    "; skip eligibility"\n'
        ")\n"
        "requested.then(reminder)"
    )
    obsolete_line = '    "; skip eligibility"'
    obsolete_block = obsolete_line + "\n"
    new_content = old_content.replace(obsolete_block, "")
    programs = {
        "edit": f"experience.content.edit(search={obsolete_block!r}, replace='')",
        "drop": f"experience.content.drop(text={obsolete_block!r})",
        "overwrite": f"experience.content.overwrite({new_content!r})",
    }
    fs.files[target] = MemoryFileUtils.write(
        MemoryFile(
            uri=target,
            memory_type="experiences",
            content=old_content,
            extra_fields={"experience_name": "case_a", "version": 1},
        )
    )
    run = analysis(passed=False)
    other = analysis(session="viking://session/s2/archives/001", passed=True)
    run.trajectories.extend(other.trajectories)
    run.trajectories[
        0
    ].content = (
        "# Session Context\n\nOutput JSON only.\n\n# Trajectory\n\n1. `user` cancel duplicate\n"
    )
    run.trajectories[0].metadata["experience_execution"] = {
        "case_a": {"slot_values": {"booking": "duplicate"}, "executed_nodes": ["reminder"]}
    }
    run.trajectories[0].content = render_trajectory_template(run.trajectories[0])
    # Only the persisted content, not freshly changed evidence metadata, goes to the LLM.
    run.trajectories[0].metadata["experience_execution"] = {"not_in_content": True}
    model.get_completion_async.return_value = programs[method] + "\nsdk.commit()"

    proposals = await ExperienceGradientEstimator(viking_fs=fs, vlm=model).estimate(
        run, policies, context
    )

    model.get_completion_async.assert_awaited_once()
    assert len(proposals) == 1
    assert proposals[0].after_file.content == new_content
    assert proposals[0].before_file.content == old_content
    assert proposals[0].base_version == 1
    assert proposals[0].metadata["memory_fields"]["content"] == new_content
    assert MemoryFileUtils.read(fs.files[target]).content == old_content
    messages = model.get_completion_async.await_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    system_prompt = messages[0]["content"]
    # Verify the composed model input, not only the separate provider/schema strings.
    assert "SDK content editing rules" not in system_prompt
    assert "Only the system may emit sdk.existing() or sdk.error()." in system_prompt
    assert (
        "Every search must match a unique, non-overlapping region of the original field"
        in system_prompt
    )
    assert system_prompt.count("Missing user information belongs in ask") == 1
    assert system_prompt.count("Never copy concrete user, reservation, payment, flight") == 1
    assert "preserve unaffected rules, node names, and relationships" in system_prompt
    assert system_prompt.count("AND semantics") == 1
    assert system_prompt.count("fictional execution") == 1
    assert "omitted checks are not proof of success" in system_prompt
    assert system_prompt.count("Analyze each candidate:") == 1
    assert "Reject advice already followed in the failed run" in system_prompt
    assert "each need only repair its own failure pattern" in system_prompt
    assert "not that the business task succeeded" in system_prompt
    assert "Example — linking" not in messages[0]["content"]
    assert "read-only context, not the SDK reply" not in messages[0]["content"]
    assert "shared Edit instructions" in messages[0]["content"]
    assert "Syntax-only example (string value, not a write operation)" in messages[0]["content"]
    assert messages[0]["content"].count("Use overwrite only for complete field rewrites.") == 1
    assert messages[0]["content"].count("Local replacement example") == 1
    assert "experience.content.overwrite(" not in messages[0]["content"]
    assert all(message["role"] == "user" for message in messages[1:])
    assert [message["content"].splitlines()[0] for message in messages[1:-2]] == [
        "# Case and Trajectory Batch",
        "# Existing Experience",
        "# Trajectory Record",
        "# Trajectory Record",
    ]
    assert messages[-2]["content"].startswith("Use the Create/Update rules")
    assert messages[-1]["content"] == (
        "Existing-memory access ONLY via: experience. "
        "Assign aliases before use; read unlisted memories first."
    )
    assert all(set(message) == {"role", "content"} for message in messages)
    records = [m for m in messages if m["content"].startswith("# Trajectory Record\n")]
    assert len(records) == 2
    for record, trajectory in zip(
        records, [other.trajectories[0], run.trajectories[0]], strict=True
    ):
        assert record["content"] == _trajectory_model_content(trajectory.content)
    assert "Output JSON only." in records[1]["content"]
    assert "booking: duplicate" in records[1]["content"]
    assert "not_in_content" not in records[1]["content"]
    existing = [
        message for message in messages if message["content"].startswith("# Existing Experience\n")
    ]
    assert len(existing) == 1
    assert existing[0]["content"].endswith(old_content)
    assert "1\tdag" not in existing[0]["content"]
    assert not any(m["content"].lstrip().startswith("{") for m in messages[1:])


@pytest.mark.asyncio
async def test_noncontiguous_related_rules_do_not_trigger_repair(setup_observation):
    fs, _ctx, model, analyzer, context, policies = setup_observation
    source = (
        "Checked bag allowance:\n"
        "- Regular members: 1 free checked bag for each economy passenger.\n"
        "- Silver members: 2 free checked bags for each economy passenger.\n"
        "- Gold members: 3 free checked bag for each economy passenger.\n"
        "- Each extra baggage is 50 dollars."
    )
    selected_rules = (
        "Checked bag allowance:\n"
        "- Gold members: 3 free checked bags for each economy passenger.\n"
        "- Each extra baggage is 50 dollars."
    )
    assert " ".join(selected_rules.split()) not in " ".join(source.split())
    run = analysis()
    run.rollout.messages = [
        Message(id="system", role="system", parts=[TextPart(text=source)]),
        Message(id="user", role="user", parts=[TextPart(text="Add one checked bag.")]),
    ]
    model.get_completion_async.return_value = trajectory_program(
        "1. Requested one additional checked bag.", context_rules=selected_rules
    )

    await observe(analyzer, run, policies, context)

    model.get_completion_async.assert_awaited_once()
    stored = MemoryFileUtils.read(fs.files[run.trajectories[0].uri])
    assert stored.content.endswith(
        f"# Session Context\n\n{selected_rules}\n\n# Trajectory\n\n"
        "1. Requested one additional checked bag."
    )


@pytest.mark.asyncio
async def test_historical_success_accepts_unstructured_content(setup_observation):
    fs, ctx, model, analyzer, context, policies = setup_observation
    model.get_completion_async.return_value = (
        "sdk.create_trajectories(case_name='case_a', context_rules='', "
        "steps='The requested change succeeded.')\nsdk.commit()"
    )
    historical = analysis(session="viking://session/historical/archives/001", passed=True)
    historical.metadata["task_spec"] = asdict(historical.rollout.case)
    await observe(analyzer, historical, policies, context)
    memory = historical.metadata["trajectory_memory"]
    memory.extra_fields["experience_update"]["state"] = "applied"
    fs.files[memory.uri] = MemoryFileUtils.write(memory)
    memory = await archive_trajectory(
        memory, viking_fs=fs, ctx=ctx, registry=get_default_registry(), vikingdb=None
    )
    policies = await store_existing_experience(fs, policies, historical)
    current = analysis(session="viking://session/current/archives/001", passed=False)
    current.metadata["task_spec"] = asdict(current.rollout.case)
    signals = await observe(analyzer, current, policies, context)
    fs.find.return_value = SimpleNamespace(memories=[SimpleNamespace(uri=memory.uri, score=0.9)])
    estimator = SimpleNamespace(estimate=AsyncMock(return_value=[]))

    await TrajectoryExperienceOptimizer(fs, estimator).plan(
        signals, policies, PatchMergePolicyOptimizerContext(ctx)
    )

    selected = estimator.estimate.await_args.args[0].trajectories
    assert [trajectory.uri for trajectory in selected] == [current.trajectories[0].uri, memory.uri]
    assert all(
        trajectory.content.endswith("The requested change succeeded.") for trajectory in selected
    )


@pytest.mark.asyncio
async def test_pending_scan_does_not_reinterpret_old_trajectory_format(
    setup_observation, monkeypatch
):
    fs, ctx, _model, analyzer, context, policies = setup_observation
    run = analysis()
    await observe(analyzer, run, policies, context)
    uri = run.trajectories[0].uri
    memory = MemoryFileUtils.read(fs.files[uri], uri=uri)
    memory.extra_fields["trajectory_format"] = "ordered_markdown_v2"
    memory.content = "# Trajectory\n\n1. `user` legacy observation"
    fs.files[uri] = MemoryFileUtils.write(memory)

    monkeypatch.setattr("openviking.session.compressor_v3.get_viking_fs", lambda: fs)
    memories = await SessionCompressorV3(vikingdb=None)._load_pending_trajectory_memories(
        viking_fs=fs,
        ctx=ctx,
        registry=get_default_registry(),
        trajectory_uris=[uri],
        case_names=None,
        retry_failed=False,
    )

    assert memories == []
    assert MemoryFileUtils.read(fs.files[uri]).content == memory.content


def test_source_steps_keep_every_record_within_shared_budget():
    messages = [
        Message(id=str(index), role="user", parts=[TextPart(text="x" * 2_000)])
        for index in range(100)
    ]

    steps = trajectory_source_steps(messages)

    assert len(steps) == 100
    assert all(step["type"] == "user" for step in steps)
    assert sum(len(step["content"]) for step in steps) <= 64_000


def test_experience_input_keeps_unstructured_content_and_inline_tool_json():
    trajectory = analysis().trajectories[0]
    trajectory.content = 'flights=[{"flight_number":"HAT072","date":"2024-05-19"}], payment_id=gift'
    assert render_trajectory_template(trajectory).endswith(trajectory.content)


@pytest.mark.asyncio
async def test_unknown_evaluation_and_same_archive_different_cases(setup_observation):
    fs, ctx, model, analyzer, context, policies = setup_observation
    first, second = analysis(passed=None), analysis(name="other", passed=None)
    model.get_completion_async.side_effect = [
        trajectory_program(),
        trajectory_program(case_name="other"),
    ]
    await observe(analyzer, first, policies, context)
    await observe(analyzer, second, policies, context)
    assert first.trajectories[0].outcome == "unknown"
    assert first.trajectories[0].metadata.get("evaluation") is None
    assert first.trajectories[0].uri != second.trajectories[0].uri


def test_tool_facts_beyond_500_chars_are_retained_and_cited():
    output = json.dumps(
        {"flights": "x" * 700, "payment_history": [{"amount": 402}], "insurance": "yes"}
    )
    evidence = observation_evidence(
        [
            Message(
                id="m",
                role="user",
                parts=[
                    ToolPart(
                        tool_id="t",
                        tool_name="get_reservation_details",
                        tool_output=output,
                        tool_status="completed",
                    )
                ],
            )
        ],
        None,
    )
    assert "402" in evidence[0]["summary"]
    assert "insurance" in evidence[0]["summary"]
    assert json.loads(evidence[0]["summary"])["tool_output"] == output


@pytest.mark.asyncio
async def test_step_count_and_numbering_mismatch_are_kept_without_repair(setup_observation):
    fs, _ctx, model, analyzer, context, policies = setup_observation
    model.get_completion_async.return_value = trajectory_program("2. first\n5. second")

    run = analysis()
    await observe(analyzer, run, policies, context)
    await observe(analyzer, run, policies, context)

    assert model.get_completion_async.await_count == 2
    stored = MemoryFileUtils.read(next(iter(fs.files.values())))
    assert stored.content.endswith("# Trajectory\n\n2. first\n5. second")


def test_external_evaluation_is_preserved_without_hardcoded_criterion_field_selection():
    from openviking.session.train import RubricEvaluation

    evaluation = RubricEvaluation(
        passed=True,
        score=1.0,
        criterion_results=[],
        feedback=[],
        metadata={
            "evaluation_result": {
                "communicate_checks": [
                    {
                        "info": "1628",
                        "justification": "very large evaluator explanation",
                        "met": True,
                    }
                ]
            }
        },
    )
    evidence = observation_evidence([], evaluation)
    assert len(evidence) == 1
    assert evidence[0]["kind"] == "external_evaluation"
    report = json.loads(evidence[0]["summary"])
    assert report["metadata"] == evaluation.metadata
    assert report["passed"] is True


@pytest.mark.asyncio
async def test_concurrent_sessions_batch_once_and_load_one_historical_success(
    setup_observation, monkeypatch
):
    fs, ctx, model, analyzer, context, policies = setup_observation
    older = analysis(session="viking://user/u/sessions/older/history/archive_001", passed=True)
    historical = analysis(session="viking://user/u/sessions/old/history/archive_001", passed=True)
    failed = analysis(session="viking://user/u/sessions/failed/history/archive_001", passed=False)
    unknown = analysis(session="viking://user/u/sessions/unknown/history/archive_001", passed=None)
    foreign = analysis(name="other", session="viking://user/u/sessions/other/history/archive_001")
    pending = analysis(session="viking://user/u/sessions/pending/history/archive_001", passed=True)
    history_runs = [older, historical, failed, unknown, foreign, pending]
    for day, run in enumerate(history_runs, start=1):
        run.metadata["task_spec"] = asdict(run.rollout.case)
        model.get_completion_async.return_value = trajectory_program(
            case_name=run.rollout.case.name
        )
        await observe(analyzer, run, policies, context)
        memory = run.metadata["trajectory_memory"]
        memory.extra_fields["created_at"] = f"2024-01-{day:02d}T00:00:00+00:00"
        memory.extra_fields["experience_update"]["state"] = (
            "pending" if run is pending else "no_op" if run is historical else "applied"
        )
        fs.files[memory.uri] = MemoryFileUtils.write(memory)
        if run is not pending:
            archived = await archive_trajectory(
                memory, viking_fs=fs, ctx=ctx, registry=get_default_registry(), vikingdb=None
            )
            run.trajectories[0].uri = archived.uri
    policies = await store_existing_experience(fs, policies, historical)
    runs = [
        analysis(session=f"viking://user/u/sessions/s{i}/history/archive_001", passed=False)
        for i in range(2)
    ]
    for run in runs:
        run.metadata["task_spec"] = asdict(run.rollout.case)
    signals = await asyncio.gather(*(observe(analyzer, run, policies, context) for run in runs))
    captured = []
    fs.find.return_value = SimpleNamespace(
        memories=[
            SimpleNamespace(uri=historical.trajectories[0].uri, score=0.9),
            SimpleNamespace(uri=older.trajectories[0].uri, score=0.8),
        ]
    )

    async def synthesize(self, provider, context):
        assert not fs.policy_lock.locked()
        provider.read_file = AsyncMock(return_value=None)
        payload = await presented_trajectories(provider)
        captured.append(payload)
        return await validated_operations(
            provider,
            operations(
                content='dag = workflow("cancel")\nc = check("duplicate requested")\nr = tell("verify duplicate identity")\nc.then(r)'
            ),
        )

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    trainer = StreamingPolicyTrainer(
        policy_set=policies,
        rollout_analyzer=SessionRolloutAnalyzer(),
        gradient_estimator=ExperienceGradientEstimator(viking_fs=fs),
        policy_optimizer=TrajectoryExperienceOptimizer(
            fs, ExperienceGradientEstimator(viking_fs=fs)
        ),
        policy_updater=MemoryFilePolicyUpdater(viking_fs=fs),
        context=PipelineContext(
            analysis_context=SessionAnalyzerContext(ctx),
            gradient_context=context,
            optimization_context=PatchMergePolicyOptimizerContext(ctx),
            apply_context=ctx,
        ),
        config=StreamingPolicyTrainerConfig(max_gradients_per_update=2, max_wait_seconds=1),
    )
    try:
        results = await asyncio.gather(
            *(
                trainer.submit_gradients(signal, analysis=run, rollout=run.rollout)
                for signal, run in zip(signals, runs, strict=True)
            )
        )
    finally:
        await trainer.close()
    assert len(captured) == 1
    payload = captured[0]
    assert [record["outcome"] for record in payload] == ["success", "failure", "failure"]
    assert payload[0]["content"] == _trajectory_model_content(historical.trajectories[0].content)
    assert all("trajectory_uri" not in record and "case_name" not in record for record in payload)
    assert results[0].batch_result is results[1].batch_result
    assert all(result.apply_result.written_uris == [f"{ROOT}/case_a.md"] for result in results)
    stored = MemoryFileUtils.read(fs.files[f"{ROOT}/case_a.md"])
    assert stored.extra_fields["case_name"] == "case_a"
    assert "source_sessions" not in stored.extra_fields
    assert "success_trajectory_refs" not in stored.extra_fields
    assert stored.links == []
    assert len([uri for uri in fs.files if "/trajectories/" in uri]) == len(history_runs) + len(
        runs
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcomes", [(False, True), (True, False), (None, True)])
async def test_current_success_and_failure_keep_current_records_without_scanning(
    setup_observation, monkeypatch, outcomes
):
    fs, ctx, _model, analyzer, context, policies = setup_observation
    signals, runs = [], []
    for index, passed in enumerate(outcomes):
        run = analysis(session=f"viking://session/current{index}/archives/001", passed=passed)
        runs.append(run)
        signals.extend(await observe(analyzer, run, policies, context))
    lookup = AsyncMock(side_effect=AssertionError("A successful current batch needs no history"))
    monkeypatch.setattr(fs, "ls", lookup)
    estimator = SimpleNamespace(estimate=AsyncMock(return_value=[]))

    await TrajectoryExperienceOptimizer(fs, estimator).plan(
        signals, policies, PatchMergePolicyOptimizerContext(ctx)
    )

    lookup.assert_not_awaited()
    estimator.estimate.assert_awaited_once()
    selected = estimator.estimate.await_args.args[0]
    expected = [trajectory.uri for run in runs for trajectory in run.trajectories]
    assert [trajectory.uri for trajectory in selected.trajectories] == expected
    assert selected.metadata["current_trajectory_uris"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [False, None])
@pytest.mark.parametrize("outcome", ["failure", "partial", "unfinished", "unknown"])
async def test_no_historical_success_does_not_use_other_outcomes(
    setup_observation, monkeypatch, passed, outcome
):
    fs, ctx, _model, analyzer, context, policies = setup_observation
    historical = analysis(session=f"viking://session/{outcome}/archives/001", passed=False)
    historical.metadata["task_spec"] = asdict(historical.rollout.case)
    await observe(analyzer, historical, policies, context)
    memory = historical.metadata["trajectory_memory"]
    memory.extra_fields["outcome"] = outcome
    memory.extra_fields["experience_update"]["state"] = "applied"
    # A stale ANN success tag must not promote a non-successful trajectory.
    memory.content = "irrelevant history"
    fs.files[memory.uri] = MemoryFileUtils.write(memory)
    policies = await store_existing_experience(fs, policies, historical)
    run = analysis(session="viking://session/current/archives/001", passed=passed)
    run.metadata["task_spec"] = asdict(run.rollout.case)
    signals = await observe(analyzer, run, policies, context)
    lookup = AsyncMock(wraps=fs.ls)
    monkeypatch.setattr(fs, "ls", lookup)
    fs.find.return_value = SimpleNamespace(memories=[SimpleNamespace(uri=memory.uri, score=0.9)])
    estimator = SimpleNamespace(estimate=AsyncMock(return_value=[]))

    await TrajectoryExperienceOptimizer(fs, estimator).plan(
        signals, policies, PatchMergePolicyOptimizerContext(ctx)
    )

    lookup.assert_not_awaited()
    fs.find.assert_awaited_once()
    selected = estimator.estimate.await_args.args[0]
    assert [trajectory.uri for trajectory in selected.trajectories] == [run.trajectories[0].uri]


@pytest.mark.asyncio
async def test_failure_only_batch_can_produce_noop_without_losing_trajectory(
    setup_observation, monkeypatch
):
    fs, ctx, model, analyzer, context, policies = setup_observation
    run = analysis(passed=False)
    signals = await observe(analyzer, run, policies, context)

    async def no_change(self, provider, context):
        provider.read_file = AsyncMock(return_value=None)
        payload = await presented_trajectories(provider)
        assert len(payload) == 1
        assert payload[0]["outcome"] == "failure"
        return None

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", no_change)
    plan = await TrajectoryExperienceOptimizer(fs, ExperienceGradientEstimator(viking_fs=fs)).plan(
        signals, policies, PatchMergePolicyOptimizerContext(ctx)
    )
    assert plan.items == []
    assert plan.read_policy_uris == [f"{ROOT}/case_a.md"]
    assert run.trajectories[0].uri in fs.files


@pytest.mark.asyncio
async def test_candidate_gate_failure_propagates_but_keeps_observation(
    setup_observation, monkeypatch
):
    fs, ctx, model, analyzer, context, policies = setup_observation
    signals = await observe(analyzer, analysis(passed=False), policies, context)

    async def candidate(self, provider, context):
        return await validated_operations(provider, operations())

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", candidate)
    from openviking.session.train.gates.errors import ExperienceGateRejected

    gate = SimpleNamespace(
        ENABLED=True,
        validate=AsyncMock(
            side_effect=ExperienceGateRejected(
                [{"gate": "test", "error": "unsupported correction"}]
            )
        ),
    )
    with pytest.raises(ValueError, match="unsupported correction"):
        await TrajectoryExperienceOptimizer(
            fs, ExperienceGradientEstimator(viking_fs=fs, gates=[gate])
        ).plan(signals, policies, PatchMergePolicyOptimizerContext(ctx))
    gate.validate.assert_awaited_once()
    assert signals[0].after_file.uri in fs.files
    proposal = gate.validate.await_args.args[1][0]
    assert len(proposal.metadata["_experience_source_contexts"]) == len(signals)


@pytest.mark.asyncio
async def test_deferred_experience_learning_claims_and_applies_pending_trajectory(
    setup_observation,
    monkeypatch,
):
    fs, ctx, _model, analyzer, context, policies = setup_observation
    run = analysis(passed=False)
    (signal,) = await observe(analyzer, run, policies, context)
    original_content = signal.after_file.content
    assert "experience_update:" not in original_content
    case = run.rollout.case
    case_uri = run.metadata["case_uri"]
    fs.files[case_uri] = MemoryFileUtils.write(
        MemoryFile(
            uri=case_uri,
            memory_type="cases",
            content="case",
            extra_fields={
                "case_name": case.name,
                "domain": case.task_signature,
                "description": json.dumps(case.input),
                "rubric": "[]",
            },
        )
    )
    legacy_uri = signal.after_file.uri.rsplit("/", 1)[0] + "/legacy.md"
    fs.files[legacy_uri] = MemoryFileUtils.write(
        MemoryFile(
            uri=legacy_uri,
            memory_type="trajectories",
            content="legacy procedure-style Trajectory",
            extra_fields={
                "trajectory_name": "legacy",
                "case_name": case.name,
                "outcome": "failure",
            },
        )
    )

    async def synthesize(self, provider, synth_context):
        provider.read_file = AsyncMock(return_value=None)
        messages = provider.format_prefetch_messages(await provider.prefetch())
        assert messages[2] == {
            "role": "user",
            "content": _trajectory_model_content(original_content),
        }
        stored = MemoryFileUtils.read(fs.files[signal.after_file.uri])
        assert stored.extra_fields["experience_update"]["state"] == "processing"
        assert stored.content == original_content
        return await validated_operations(
            provider,
            operations(
                content=(
                    'dag = workflow("cancel")\n'
                    'c = check("duplicate requested")\n'
                    'r = tell("verify duplicate identity")\n'
                    "c.then(r)"
                )
            ),
        )

    async def registry_snapshot(viking_fs, account_id, registry):
        return registry

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    monkeypatch.setattr("openviking.session.compressor_v3.get_viking_fs", lambda: fs)
    monkeypatch.setattr(
        "openviking.session.memory.account_templates.resolve_account_memory_registry",
        registry_snapshot,
    )
    monkeypatch.setattr(
        "openviking.session.compressor_v3.get_openviking_config",
        lambda: SimpleNamespace(
            agent_evolution=SimpleNamespace(
                dag_decider=None,
            ),
            jev=None,
        ),
    )
    compressor = SessionCompressorV3(vikingdb=None)

    result = await compressor.learn_pending_experiences(
        ctx=ctx,
        trajectory_uris=[signal.after_file.uri],
    )

    assert result["applied_count"] == 1
    assert result["written_experience_uris"] == [f"{ROOT}/case_a.md"]
    final_uri = trajectory_archive_uri(signal.after_file.uri)
    assert signal.after_file.uri not in fs.files
    stored = MemoryFileUtils.read(fs.files[final_uri], uri=final_uri)
    assert stored.extra_fields["experience_update"]["state"] == "applied"
    assert stored.content == original_content
    assert stored.content.count("# Trajectory Record\n") == 1
    assert stored.extra_fields["experience_update"]["attempts"] == 1
    assert all(
        not uri.endswith("/legacy.md")
        for uri in result["results"][0]["plan_metadata"]["chunks"][0]["comparisons"][0][
            "trajectory_uris"
        ]
    )

    repeated = await compressor.learn_pending_experiences(
        ctx=ctx,
        trajectory_uris=[signal.after_file.uri],
    )
    assert repeated["trajectory_count"] == 0


@pytest.mark.asyncio
async def test_deferred_experience_filters_form_a_union(setup_observation):
    fs, ctx, _model, analyzer, context, policies = setup_observation
    explicit_run = analysis(passed=False)
    scanned_run = analysis(name="other", passed=True)
    (explicit_signal,) = await observe(analyzer, explicit_run, policies, context)
    _model.get_completion_async.return_value = trajectory_program(case_name="other")
    (scanned_signal,) = await observe(analyzer, scanned_run, policies, context)
    legacy_uri = scanned_signal.after_file.uri.rsplit("/", 1)[0] + "/legacy.md"
    fs.files[legacy_uri] = MemoryFileUtils.write(
        MemoryFile(
            uri=legacy_uri,
            memory_type="trajectories",
            content="legacy procedure-style Trajectory",
            extra_fields={
                "trajectory_name": "legacy",
                "case_name": scanned_run.rollout.case.name,
                "outcome": "success",
            },
        )
    )
    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=SimpleNamespace())

    memories = await compressor._load_pending_trajectory_memories(
        viking_fs=fs,
        ctx=ctx,
        registry=get_default_registry(),
        trajectory_uris=[explicit_signal.after_file.uri],
        case_names=[scanned_run.rollout.case.name],
        retry_failed=False,
    )

    assert {memory.uri for memory in memories} == {
        explicit_signal.after_file.uri,
        scanned_signal.after_file.uri,
    }


@pytest.mark.asyncio
async def test_deferred_experience_learning_records_failed_state(setup_observation, monkeypatch):
    fs, ctx, _model, analyzer, context, policies = setup_observation
    run = analysis(passed=False)
    (signal,) = await observe(analyzer, run, policies, context)

    async def registry_snapshot(viking_fs, account_id, registry):
        return registry

    monkeypatch.setattr("openviking.session.compressor_v3.get_viking_fs", lambda: fs)
    monkeypatch.setattr(
        "openviking.session.memory.account_templates.resolve_account_memory_registry",
        registry_snapshot,
    )
    compressor = SessionCompressorV3(vikingdb=None)
    compressor._load_pending_trajectory_memories = AsyncMock(return_value=[signal.after_file])
    compressor._learn_one_trajectory_case = AsyncMock(side_effect=RuntimeError("synthesis failed"))

    result = await compressor.learn_pending_experiences(
        ctx=ctx,
        trajectory_uris=[signal.after_file.uri],
    )

    assert result["failed_count"] == 1
    stored = MemoryFileUtils.read(fs.files[signal.after_file.uri], uri=signal.after_file.uri)
    state = stored.extra_fields["experience_update"]
    assert state["state"] == "failed"
    assert state["attempts"] == 1
    assert state["error"] == "synthesis failed"


@pytest.mark.asyncio
async def test_inserted_tell_reaches_commit_persisted_trajectory_and_experience_input(
    setup_observation, tmp_path, monkeypatch
):
    from vikingbot.config.schema import SessionKey
    from vikingbot.session.manager import SessionManager

    from benchmark.tau2.train.dag_experience_runtime import Tau2DagExperienceRuntime
    from openviking.session.compressor_v3 import _training_outcome_from_messages
    from openviking.session.train.components.session_commit import _evaluation_message_to_request

    fs, ctx, model, analyzer, context, policies = setup_observation
    # Supply the existence check required by the create-only memory guard.
    monkeypatch.setattr(
        fs, "exists", AsyncMock(side_effect=lambda uri, **kwargs: uri in fs.files), raising=False
    )
    uri = f"{ROOT}/case_a.md"
    instruction = {
        "experience_uri": uri,
        "node_id": 2,
        "slot_name": "review_scope",
        "provider": {"type": "TellAgent", "instruction": "Review the cancellation scope"},
        "conditions": [],
    }
    runtime = Tau2DagExperienceRuntime()
    runtime.bind_session(
        SessionManager(tmp_path), SessionKey(type="cli", channel_id="test", chat_id="execution")
    )
    runtime.client = SimpleNamespace(
        ensure_session=AsyncMock(),
        search_advice=AsyncMock(
            return_value={
                "instructions": [instruction],
                "experiences": [
                    {
                        "experience_uri": uri,
                        "state": "running",
                        "node_slots": {1: "cancel_requested", 2: "review_scope"},
                        "slot_values": {"cancel_requested": True},
                        "executed_nodes": [1],
                        "current_nodes": [2],
                        "actions": [instruction],
                    }
                ],
                "errors": [],
            }
        ),
    )
    runtime.record_user_message("cancel")
    async for note in runtime.recall([]):
        await runtime.record_injected(note)
    run = analysis(passed=True)
    run.rollout.evaluation = run.evaluation
    run.rollout.metadata["dag_runtime"] = {"events": runtime.events}
    envelope = _evaluation_message_to_request(run.rollout)
    text = envelope["parts"][0]["text"]
    assert "delivered_advice" in text and "action_outcomes" not in text
    run.evaluation, execution = _training_outcome_from_messages(
        [Message(id="outcome", role="user", parts=[TextPart(text=text)])]
    )
    run.metadata["experience_execution"] = execution
    run.trajectories = []
    await observe(analyzer, run, policies, context)
    trajectory = run.trajectories[0]
    stored = MemoryFileUtils.read(fs.files[trajectory.uri], uri=trajectory.uri)
    expected = {
        "case_a": {
            "slot_values": {"cancel_requested": True},
            "executed_nodes": ["cancel_requested"],
            "current_nodes": ["review_scope"],
        }
    }
    assert stored.extra_fields["experience_execution"] == expected
    provider = AgentExperienceContextProvider(
        messages=[],
        case=run.rollout.case,
        target_uri=uri,
        source_session_uri=run.metadata["source_session_uri"],
        trajectories=[trajectory],
    )
    provider._eval_memory = AsyncMock(return_value={"experience": {"error": "File not found"}})
    record = (await presented_trajectories(provider))[0]
    assert record["experience_execution"] == expected
    assert record["outcome"] == "success"
    assert "evaluation" not in record
    assert stored.extra_fields["evaluation"] == {"passed": True, "score": 1.0}


@pytest.mark.asyncio
async def test_distinct_tasks_share_case_and_experience_but_keep_their_own_evidence(
    setup_observation, monkeypatch
):
    from dataclasses import asdict

    from openviking.session.train import Case, Rubric, RubricCriterion
    from openviking.session.train.components.session_commit import _evaluation_message_to_request

    fs, ctx, _model, analyzer, context, policies = setup_observation
    monkeypatch.setattr("openviking.session.compressor_v3.get_viking_fs", lambda: fs)
    monkeypatch.setattr(
        "openviking.session.memory.account_templates.resolve_account_memory_registry",
        AsyncMock(return_value=get_default_registry()),
    )
    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=analyzer)
    compressor._session_skill_extraction_enabled = lambda: False
    shared = Case("payment_selection", "Choose an eligible payment method", {}, Rubric("r", "", []))
    case_uri = f"viking://user/u/memories/cases/{shared.name}.md"
    shared.metadata["case_uris"] = [case_uri]
    fs.files[case_uri] = MemoryFileUtils.write(
        MemoryFile(
            uri=case_uri,
            memory_type="cases",
            content="Payment selection scenario",
            extra_fields={
                "case_name": shared.name,
                "domain": shared.task_signature,
                "description": "Use the supplied booking and payment information.",
                "rubric": '[{"id":"payment","description":"Respect the user\'s payment preference and confirm changes."}]',
            },
        )
    )
    from openviking.session.compressor_v3 import _canonical_cases_from_update_result
    from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
    from openviking.session.memory.memory_updater import MemoryUpdateResult

    unchanged = fs.files[case_uri]
    selected = ResolvedOperations(
        case_references=[
            ResolvedOperation(
                memory_type="cases",
                uris=[case_uri],
                memory_fields={},
                old_memory_file_content=MemoryFileUtils.read(unchanged, uri=case_uri),
            )
        ],
        upsert_operations=[],
        delete_file_contents=[],
        errors=[],
    )
    (shared,) = await _canonical_cases_from_update_result(
        operations=selected,
        requested_operations=selected,
        result=MemoryUpdateResult(),
        viking_fs=fs,
        ctx=ctx,
    )
    assert fs.files[case_uri] == unchanged
    assert shared.input == {"summary": "Use the supplied booking and payment information."}
    assert shared.rubric.description == ""
    assert (
        shared.rubric.criteria[0].description
        == "Respect the user's payment preference and confirm changes."
    )
    _model.get_completion_async.return_value = trajectory_program(case_name=shared.name).replace(
        "steps=",
        "evaluation_criteria="
        + repr(
            json.dumps(
                [
                    {
                        "id": "payment",
                        "passed": None,
                        "evidence": "Only a cancellation request is observed.",
                    }
                ]
            )
        )
        + ", steps=",
        1,
    )
    records = []
    signals = []
    for index, passed in enumerate([True, False]):
        task = Case(
            f"task_{index}",
            f"dataset:task:{index}",
            {"reservation_id": f"R{index}", "payment_id": f"card_{index}"},
            Rubric(
                f"r{index}",
                "Concrete task",
                [RubricCriterion("payment", f"Use card_{index}", True, 1.0)],
            ),
        )
        run = analysis(
            name=shared.name, session=f"viking://session/task{index}/archives/001", passed=passed
        )
        run.rollout.evaluation = run.evaluation
        outcome = _evaluation_message_to_request(run.rollout)
        messages = [
            *run.rollout.messages,
            Message(
                id="eval", role=outcome["role"], parts=[TextPart(text=outcome["parts"][0]["text"])]
            ),
        ]
        result = await compressor.train_from_extracted_cases(
            cases=[shared],
            task=task,
            messages=messages,
            ctx=ctx,
            archive_uri=run.metadata["source_session_uri"],
            strict_extract_errors=True,
        )
        assert len(result["trajectory_uris"]) == 1
        memory = MemoryFileUtils.read(
            fs.files[result["trajectory_uris"][0]], uri=result["trajectory_uris"][0]
        )
        assert memory.extra_fields["task_spec"] == asdict(task)
        assert memory.extra_fields["evaluation"] == {"passed": passed, "score": float(passed)}
        assert memory.extra_fields["case_name"] == shared.name
        # Exercise retry with the same concrete task, then pass the persisted records to Phase 2.
        run.metadata["task_spec"] = asdict(task)
        (signal,) = await observe(analyzer, run, policies, context)
        signals.append(signal)
        records.append(run.trajectories[0])
    assert len({t.uri for t in records}) == 2
    assert len({s.target_uri for s in signals}) == 1
    estimator = SimpleNamespace(estimate=AsyncMock(return_value=[]))
    await TrajectoryExperienceOptimizer(fs, estimator).plan(
        signals, policies, PatchMergePolicyOptimizerContext(ctx)
    )
    estimator.estimate.assert_awaited_once()
    batch = estimator.estimate.await_args.args[0].trajectories
    provider = AgentExperienceContextProvider(
        messages=[],
        case=shared,
        target_uri=signals[0].target_uri,
        source_session_uri=records[0].metadata["source_session_uri"],
        trajectories=batch,
    )
    provider._ctx, provider._viking_fs = ctx, fs
    presented = await presented_trajectories(provider)
    assert [item["outcome"] for item in presented] == ["success", "failure"]
    assert all("evaluation" not in item for item in presented)
    assert [record.metadata["evaluation"]["passed"] for record in records] == [True, False]
    assert all("task_spec" not in item and "source_session_uri" not in item for item in presented)
    assert len({record.metadata["source_session_uri"] for record in records}) == 2


@pytest.mark.asyncio
async def test_repeated_source_preserves_old_task_and_creates_new_task_record(setup_observation):
    _fs, _ctx, model, analyzer, context, policies = setup_observation
    run = analysis()
    run.metadata["task_spec"] = {"name": "task_a", "input": {"booking": "A"}}
    await observe(analyzer, run, policies, context)
    previous_uri = run.trajectories[0].uri
    previous_raw = _fs.files[previous_uri]
    run.metadata["task_spec"] = {"name": "task_b", "input": {"booking": "B"}}
    await observe(analyzer, run, policies, context)
    assert run.trajectories[0].uri != previous_uri
    assert run.trajectories[0].metadata["task_spec"]["name"] == "task_b"
    assert _fs.files[previous_uri] == previous_raw
    assert model.get_completion_async.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vikingbot", "native"])
async def test_outcome_roundtrip_persists_diagnostics_without_synthetic_steps(
    setup_observation, backend
):
    from benchmark.tau2.train._rollout_helpers import _tau2_evaluation
    from benchmark.tau2.train.rollout_executor_native import _build_rollout_messages_from_simulation
    from benchmark.tau2.train.rollout_executor_vikingbot import _build_rollout_messages
    from openviking.session.compressor_v3 import (
        _training_conversation_messages,
        _training_outcome_from_messages,
    )
    from openviking.session.train.components.session_commit import _evaluation_message_to_request

    fs, _ctx, model, analyzer, context, policies = setup_observation
    note = "Simulation terminated prematurely. Termination reason: max_steps"
    raw_result = {
        "info": {"note": note},
        "action_checks": [
            {
                "action": {"name": "cancel_reservation", "arguments": {"id": "A"}},
                "action_match": True,
            }
        ],
        "communicate_checks": [{"info": "708", "met": False}],
    }
    run = analysis(passed=False)
    if backend == "vikingbot":
        messages = _build_rollout_messages(
            system_prompt="Obtain explicit confirmation.",
            user_prompt="Cancel A.",
            tools_used=[
                {"tool_name": "done", "args": {}, "result": "Error: USER_CONVERSATION_ACTIVE"}
            ],
            final_content=None,
        )
    else:
        messages = _build_rollout_messages_from_simulation(
            simulation=SimpleNamespace(
                messages=[
                    SimpleNamespace(role="system", content="Obtain explicit confirmation."),
                    SimpleNamespace(role="user", content="Cancel A."),
                    SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(id="done1", name="done", arguments={}),
                        ],
                    ),
                    SimpleNamespace(
                        role="tool", id="done1", content="Error: USER_CONVERSATION_ACTIVE"
                    ),
                ]
            )
        )
    run.rollout.evaluation = _tau2_evaluation(reward=0.0, evaluation_result=raw_result)
    envelope = _evaluation_message_to_request(run.rollout)
    committed_messages = [
        *messages,
        Message(
            id="outcome",
            role=envelope["role"],
            parts=[
                TextPart(text=envelope["parts"][0]["text"]),
            ],
        ),
    ]
    run.evaluation, _ = _training_outcome_from_messages(committed_messages)
    run.rollout.messages = _training_conversation_messages(committed_messages)
    assert run.rollout.messages == messages
    expected_criteria = [{"id": "tau2_reward", "passed": False, "score": 0.0, "evidence": note}]
    model.get_completion_async.return_value = trajectory_program(
        evaluation_criteria=expected_criteria
    )
    # Both adapters use the same source-step builder and the same persisted result path.
    await observe(analyzer, run, policies, context)
    payload = json.loads(model.get_completion_async.await_args.kwargs["messages"][1]["content"])
    assert [s["type"] for s in payload["source_steps"]] == ["user", "user", "tool:done"]
    assert all(s["content"] for s in payload["source_steps"])
    assert payload["source_steps"][0]["content"] == "system:\nObtain explicit confirmation."
    assert "USER_CONVERSATION_ACTIVE" in payload["source_steps"][-1]["content"]
    assert not any("task_success:" in s["content"] for s in payload["source_steps"])
    trajectory = run.trajectories[0]
    stored = MemoryFileUtils.read(fs.files[trajectory.uri], uri=trajectory.uri)
    result = yaml.safe_load(stored.content.split("```yaml\n", 1)[1].split("\n```\n\n", 1)[0])
    assert result["outcome"] == "failure"
    assert result["evaluation"] == {"details": {"info": {"note": note}}}
    assert stored.extra_fields["evaluation"] == {
        "passed": False,
        "score": 0.0,
        "details": {"info": {"note": note}},
    }
    assert payload["cases"][0]["external_evaluation"]["metadata"]["evaluation_result"] == raw_result
    # Explicit external results are extracted even when the Case rubric is empty.
    assert result["evaluation_criteria"] == expected_criteria
    provider = AgentExperienceContextProvider(
        messages=[],
        case=run.rollout.case,
        target_uri=f"{ROOT}/case_a.md",
        source_session_uri=run.metadata["source_session_uri"],
        trajectories=[trajectory],
    )
    provider._eval_memory = AsyncMock(return_value={"experience": {"error": "File not found"}})
    presented = provider.format_prefetch_messages(await provider.prefetch())
    assert presented[2] == {"role": "user", "content": _trajectory_model_content(stored.content)}
    assert note in presented[2]["content"]
    await observe(analyzer, run, policies, context)
    assert MemoryFileUtils.read(fs.files[trajectory.uri]).content == stored.content
    assert model.get_completion_async.await_count == 2


@pytest.mark.asyncio
async def test_trajectory_template_renders_once_and_lifecycle_keeps_frozen_body(setup_observation):
    from openviking.session.memory.memory_type_registry import MemoryTypeRegistry

    fs, ctx, model, analyzer, context, policies = setup_observation
    registry = MemoryTypeRegistry()
    schema = registry.get("trajectories")
    assert schema.content_template
    schema.content_template = schema.content_template.replace(
        "# Trajectory Record", "# Template-owned Trajectory"
    )
    context.memory_registry = registry
    run = analysis(passed=False)
    await observe(analyzer, run, policies, context)
    memory = run.metadata["trajectory_memory"]
    frozen = memory.content
    assert frozen.startswith("# Template-owned Trajectory\n\n```yaml\n")
    assert "# Trajectory Record" not in frozen
    assert frozen.count("# Template-owned Trajectory") == 1
    assert frozen.endswith(STORED_OBSERVATION)

    # Lifecycle updates must never render the immutable body with a new template.
    # A repeated extraction is a NEW record and uses the current template.
    schema.content_template = "{{ content | deliberately_missing_filter }}"
    compressor = object.__new__(SessionCompressorV3)
    memory.extra_fields["experience_update"]["feedback_recorded"] = True
    fs.files[memory.uri] = MemoryFileUtils.write(memory)
    for state in ("processing", "applied"):
        memory = await compressor._write_trajectory_update_state(
            viking_fs=fs, memory=memory, ctx=ctx, state=state, batch_id="test-batch"
        )
        assert memory.content == frozen
        assert memory.extra_fields["experience_update"]["state"] == state
        assert memory.extra_fields["experience_update"]["feedback_recorded"] is True
    with pytest.raises(ValueError, match="deliberately_missing_filter"):
        await observe(analyzer, run, policies, context)
    assert MemoryFileUtils.read(fs.files[memory.uri]).content == frozen
    assert model.get_completion_async.await_count == 2


@pytest.mark.asyncio
async def test_invalid_trajectory_template_does_not_persist_an_unrendered_body(setup_observation):
    from openviking.session.memory.memory_type_registry import MemoryTypeRegistry

    fs, _ctx, _model, analyzer, context, policies = setup_observation
    registry = MemoryTypeRegistry()
    registry.get("trajectories").content_template = "{{ content | deliberately_missing_filter }}"
    context.memory_registry = registry
    with pytest.raises(ValueError, match="deliberately_missing_filter"):
        await observe(analyzer, analysis(passed=False), policies, context)
    assert not any("/memories/trajectories/" in uri for uri in fs.files)


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_forwarded", [False, True])
async def test_communication_provenance_reaches_trajectory_model_input(
    setup_observation, auto_forwarded
):
    from benchmark.tau2.train.rollout_executor_vikingbot import _build_rollout_messages
    from openviking.session.compressor_v3 import (
        _training_conversation_messages,
        _training_outcome_from_messages,
    )
    from openviking.session.train.components.session_commit import build_commit_messages

    _fs, _ctx, model, analyzer, context, policies = setup_observation
    run = analysis()
    run.rollout.messages = _build_rollout_messages(
        system_prompt="Use communicate_with_user to speak to the user.",
        user_prompt="Cancel A.",
        tools_used=[
            {
                "tool_name": "communicate_with_user",
                "args": {"content": "Please confirm cancellation of A."},
                "result": "Yes.",
                "auto": auto_forwarded,
            }
        ],
        final_content="A plain final response.",
    )
    run.rollout.evaluation = run.evaluation
    run.metadata["experience_execution"] = {
        f"{ROOT}/case_a.md": {
            "delivered_advice": [
                {"slot_name": "tell_comm_tool", "instruction": "Use communicate_with_user."}
            ]
        }
    }
    # Exercise the actual commit serializer, message reconstruction and metadata removal.
    wire_messages = json.loads(json.dumps(build_commit_messages(run.rollout)))
    committed = [Message.from_dict({"id": f"m{i}", **m}) for i, m in enumerate(wire_messages)]
    run.evaluation, _ = _training_outcome_from_messages(committed)
    run.rollout.messages = _training_conversation_messages(committed)
    await observe(analyzer, run, policies, context)
    model.get_completion_async.assert_awaited_once()
    request = model.get_completion_async.await_args.kwargs["messages"]
    payload = next(
        json.loads(m["content"]) for m in request if m["content"].startswith('{"cases":')
    )
    steps = payload["source_steps"]
    assert [step["type"] for step in steps] == ["user", "user", "assistant", "user", "assistant"]
    prefix = "auto-forwarded via" if auto_forwarded else "via"
    assert steps[2]["content"] == (
        f"[{prefix} communicate_with_user]\nPlease confirm cancellation of A."
    )
    assert steps[3]["content"] == "Yes."
    assert steps[4]["content"] == "A plain final response."
    assert payload["cases"][0]["injected_actions"][0]["slot_name"] == "tell_comm_tool"
