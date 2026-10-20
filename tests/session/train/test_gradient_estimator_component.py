from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from test_fakes import fake_request_context

from openviking.message import Message, TextPart
from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation, ResolvedOperations
from openviking.session.train import (
    Case,
    ExperienceGradientContext,
    ExperienceGradientEstimator,
    ExperienceSet,
    Rollout,
    RolloutAnalysis,
    Rubric,
    RubricEvaluation,
    Trajectory,
)
from openviking.session.train.components.trajectory_observation import (
    TRAJECTORY_FORMAT,
)
from openviking.session.train.gates.identifiers import ExperienceIdentifierGate
from openviking.session.train.utils import (
    EXPERIENCE_SOURCE_CONTEXTS_KEY,
)
from tests.session.train.test_fakes import render_trajectory_template

ROOT = "viking://user/u/memories/experiences"
SESSION = "viking://session/s1/archives/001"


def analysis(name="case_a", session=SESSION, passed=True):
    case = Case(
        name=name,
        task_signature="cancel duplicate",
        input={},
        rubric=Rubric(name="done", description="", criteria=[]),
    )
    rollout = Rollout(
        case=case,
        messages=[
            Message(id="m1", role="user", parts=[TextPart(text="Cancel the duplicate booking.")])
        ],
        policy_snapshot_id="p0",
    )
    result = RolloutAnalysis(
        rollout=rollout,
        trajectories=[
            Trajectory(
                name="sample",
                uri=(
                    "viking://user/u/memories/trajectories/"
                    + name
                    + "/"
                    + session.rsplit("/sessions/", 1)[-1].replace("/", "_")
                    + ".md"
                ),
                content="# Session Context\n\n# Trajectory\n\n1. `user` cancel duplicate\n",
                outcome="unknown" if passed is None else "success" if passed else "failure",
                metadata={
                    "case_name": name,
                    "source_session_uri": session,
                    "evidence": [],
                    "experience_execution": {},
                    "evaluation": None,
                },
            )
        ],
        evaluation=(
            None
            if passed is None
            else RubricEvaluation(
                passed=passed,
                score=float(passed),
                criterion_results=[],
                feedback=[] if passed else ["duplicate remains"],
            )
        ),
        metadata={
            "source_session_uri": session,
            "case_uri": f"viking://user/u/memories/cases/{name}.md",
        },
    )

    for trajectory in result.trajectories:
        trajectory.metadata["trajectory_format"] = TRAJECTORY_FORMAT
        trajectory.content = render_trajectory_template(trajectory)
    return result


def operations(name="case_a", **fields):
    return ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                uris=[f"{ROOT}/{name}.md"],
                memory_fields={
                    "experience_name": name,
                    "content": 'dag = workflow("cancel")\ncancel = call("cancel", "duplicate")',
                    **fields,
                },
            )
        ],
        delete_file_contents=[],
        errors=[],
    )


async def validated_operations(provider, result):
    """Simulate the validation hook in component-level extraction stubs."""
    errors = await provider.validate_resolved_operations(result)
    if errors:
        raise ValueError(json.dumps(errors, ensure_ascii=False))
    return result


def mock_extract_loop(result):
    async def run(provider, context):
        return await validated_operations(provider, result)

    return AsyncMock(side_effect=run)


async def estimate(estimator, item):
    return await estimator.estimate(
        item,
        ExperienceSet(root_uri=ROOT, policies=[]),
        ExperienceGradientContext(request_context=fake_request_context(user_id="u"), messages=[]),
    )


@pytest.mark.asyncio
async def test_direct_session_proposal_has_fixed_case_and_archive_provenance(monkeypatch):
    run = mock_extract_loop(operations())
    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", run)
    item = analysis(passed=False)
    gradients = await estimate(ExperienceGradientEstimator(), item)
    assert len(item.trajectories) == 1
    assert len(gradients) == 1
    proposal = gradients[0]
    assert proposal.target_name == item.rollout.case.name
    assert proposal.links[0].to_uri == item.trajectories[0].uri
    gate_context = proposal.metadata[EXPERIENCE_SOURCE_CONTEXTS_KEY][0]
    assert gate_context["passed"] is False
    assert gate_context["feedback"] == ["duplicate remains"]
    assert gate_context["evidence"][0]["summary"] == "Cancel the duplicate booking."
    provider = run.await_args.args[0]
    assert provider.target_uri == proposal.target_uri
    assert provider.messages == []
    assert provider.trajectories == item.trajectories


@pytest.mark.asyncio
async def test_unknown_outcome_is_not_promoted_to_success(monkeypatch):
    monkeypatch.setattr(
        ExperienceGradientEstimator, "_run_extract_loop", mock_extract_loop(operations())
    )
    (proposal,) = await estimate(ExperienceGradientEstimator(), analysis(passed=None))
    assert proposal.metadata[EXPERIENCE_SOURCE_CONTEXTS_KEY][0]["passed"] is None


@pytest.mark.asyncio
async def test_candidate_can_generalize_evaluator_backed_success_value(monkeypatch):
    item = analysis(passed=True)
    item.trajectories[0].metadata["evaluation_criteria"] = [
        {
            "id": "evaluation:metadata.result.communicate_checks[0].met",
            "kind": "criterion",
            "passed": True,
            "summary": '{"info":"1628","met":true}',
        }
    ]

    async def generalized_candidate(self, provider, context):
        provider.read_file = AsyncMock(return_value=None)
        await provider.prefetch()
        return await validated_operations(
            provider,
            operations(
                content='dag = workflow("total")\nr = tell("calculate the total from current records")'
            ),
        )

    monkeypatch.setattr(
        ExperienceGradientEstimator,
        "_run_extract_loop",
        generalized_candidate,
    )

    proposals = await estimate(ExperienceGradientEstimator(), item)
    assert len(proposals) == 1


@pytest.mark.asyncio
async def test_candidate_copying_session_entity_ids_into_reminder_is_rejected(monkeypatch):
    monkeypatch.setattr(ExperienceIdentifierGate, "ENABLED", True)
    item = analysis(passed=True)
    item.rollout.case.input = {"user_query": "Cancel the reservations XEHM4B and 59XX6W."}

    monkeypatch.setattr(
        ExperienceGradientEstimator,
        "_run_extract_loop",
        mock_extract_loop(
            operations(
                content=(
                    'dag = workflow("scope")\n'
                    'a = check("A")\n'
                    'b = check("B")\n'
                    "either = any_of(a, b)\n"
                    'r = tell("Only modify XEHM4B and 59XX6W unless the user adds another")\n'
                    "either.then(r)"
                )
            )
        ),
    )

    with pytest.raises(ValueError, match="ExperienceIdentifierGate") as error:
        await estimate(ExperienceGradientEstimator(), item)
    assert "59XX6W" in str(error.value)
    assert "XEHM4B" in str(error.value)
    assert "experience_synthesis_noop" not in item.metadata


@pytest.mark.asyncio
async def test_generalized_reminder_and_numeric_rule_value_are_accepted(monkeypatch):
    monkeypatch.setattr(ExperienceIdentifierGate, "ENABLED", True)
    item = analysis(passed=True)
    item.rollout.case.input = {
        "user_query": "Cancel reservations XEHM4B and 59XX6W and communicate total 1628."
    }

    monkeypatch.setattr(
        ExperienceGradientEstimator,
        "_run_extract_loop",
        mock_extract_loop(
            operations(
                content=(
                    'dag = workflow("scope")\n'
                    'r = tell("Only modify reservations explicitly selected in the current '
                    'request; communicate total 1628")'
                )
            )
        ),
    )

    proposals = await estimate(ExperienceGradientEstimator(), item)
    assert len(proposals) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid", ["rename", "extra", "delete", "cross_case", "supersedes", "errors"]
)
async def test_invalid_target_changes_fail_explicitly(monkeypatch, invalid):
    ops = operations()
    if invalid == "rename":
        ops.upsert_operations[0].memory_fields["experience_name"] = "other"
    if invalid == "extra":
        ops.upsert_operations.append(operations("other").upsert_operations[0])
    if invalid == "delete":
        ops.delete_file_contents.append(MemoryFile(uri=f"{ROOT}/case_a.md", content="old"))
    if invalid == "cross_case":
        ops.upsert_operations[0].uris = [f"{ROOT}/other.md"]
    if invalid == "supersedes":
        ops.upsert_operations[0].memory_fields["supersedes"] = "other"
    if invalid == "errors":
        ops.errors.append("invalid SDK")
    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", mock_extract_loop(ops))
    with pytest.raises(ValueError):
        await estimate(ExperienceGradientEstimator(), analysis())


@pytest.mark.asyncio
async def test_concurrent_sessions_propose_same_case_without_waiting_for_trials(monkeypatch):
    entered = 0
    ready = asyncio.Event()

    async def run(self, provider, context):
        nonlocal entered
        entered += 1
        if entered == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 1)
        return await validated_operations(provider, operations())

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", run)
    batches = await asyncio.gather(
        estimate(ExperienceGradientEstimator(), analysis()),
        estimate(ExperienceGradientEstimator(), analysis(session=SESSION.replace("s1", "s2"))),
    )
    assert batches[0][0].target_uri == batches[1][0].target_uri
    assert batches[0][0].links[0].to_uri != batches[1][0].links[0].to_uri


@pytest.mark.asyncio
async def test_estimator_wires_session_provider_into_extract_loop(monkeypatch):
    from types import SimpleNamespace

    from openviking.server.identity import RequestContext, Role
    from openviking.session.train.components import gradient_estimator as module
    from openviking_cli.session.user_id import UserIdentifier

    recorded = []

    class Loop:
        def __init__(self, **kwargs):
            recorded.append(kwargs)

        async def run(self):
            return await validated_operations(recorded[0]["context_provider"], operations()), []

    monkeypatch.setattr(module, "ExtractLoop", Loop)
    fs = SimpleNamespace()
    ctx = RequestContext(user=UserIdentifier("default", "u"), role=Role.ROOT)
    resolver = SimpleNamespace(get_vlm=AsyncMock(return_value=SimpleNamespace(model="test")))
    estimator = ExperienceGradientEstimator(viking_fs=fs, vlm_resolver=resolver)
    result = await estimator.estimate(
        analysis(), ExperienceSet(ROOT, []), ExperienceGradientContext(ctx, [])
    )
    provider = recorded[0]["context_provider"]
    assert [schema.memory_type for schema in provider.get_memory_schemas(ctx)] == ["experiences"]
    assert provider.case.name == result[0].target_name
    assert provider.source_session_uri == SESSION
    assert provider._viking_fs is fs
    resolver.get_vlm.assert_awaited_once_with("default")


@pytest.mark.asyncio
@pytest.mark.parametrize("as_dict", [False, True])
async def test_patch_proposal_materializes_valid_dag_and_retains_observed_base(
    monkeypatch, as_dict
):
    from openviking.session.memory.merge_op import SearchReplaceBlock, StrPatch

    source = 'dag = workflow("review")\nr = tell("old reminder")'
    patch = StrPatch(blocks=[SearchReplaceBlock(search="old reminder", replace="new reminder")])
    value = patch.model_dump() if as_dict else patch
    ops = operations(content=value)
    old = MemoryFile(
        uri=f"{ROOT}/case_a.md",
        memory_type="experiences",
        content=source,
        extra_fields={"experience_name": "case_a", "version": 3},
    )
    ops.upsert_operations[0].old_memory_file_content = old
    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", mock_extract_loop(ops))
    (proposal,) = await estimate(ExperienceGradientEstimator(), analysis())
    expected = source.replace("old reminder", "new reminder")
    assert proposal.before_file.content == source
    assert proposal.after_file.content == expected
    assert proposal.metadata["memory_fields"]["content"] == expected
    assert proposal.base_version == 3
    assert ops.upsert_operations[0].memory_fields["content"] is value
