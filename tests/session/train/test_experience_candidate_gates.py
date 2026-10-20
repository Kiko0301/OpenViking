from __future__ import annotations

import json
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_case_experience_pipeline import setup_observation as setup_observation
from test_gradient_estimator_component import ROOT, analysis, operations

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.train import ExperienceGradientEstimator, MemoryFilePolicyUpdater
from openviking.session.train.components.policy_optimizer import case_experience_plan_items
from openviking.session.train.domain import PolicyUpdatePlan
from openviking.session.train.gates.errors import ExperienceGateRejected
from openviking.session.train.gates.identifiers import ExperienceIdentifierGate

TARGET = ROOT + "/case_a.md"
BAD = 'dag = workflow("review")\nr = tell("Cancel XEHM4B")'
GOOD = 'dag = workflow("review")\nr = tell("Verify the reservation selected by this user")'


def create(source):
    return f"sdk.create_experiences(experience_name='case_a', content={source!r})\nsdk.commit()"


def harness(setup, monkeypatch, replies, *, gates=(), old=None):
    fs, ctx, model, _analyzer, context, policies = setup
    if old is not None:
        fs.files[TARGET] = MemoryFileUtils.write(
            MemoryFile(
                uri=TARGET,
                memory_type="experiences",
                content=old,
                extra_fields={"experience_name": "case_a", "case_name": "case_a", "version": 3},
            )
        )
    replies = iter(replies)

    async def complete(**kwargs):
        assert not fs.policy_lock.locked(), "model calls must not hold the write lock"
        return next(replies)

    model.get_completion_async = AsyncMock(side_effect=complete)
    config = SimpleNamespace(
        memory=SimpleNamespace(link_enabled=False, extraction_output_format="python")
    )
    monkeypatch.setattr(
        "openviking.session.memory.extract_loop.get_openviking_config", lambda: config
    )
    item = analysis()
    item.rollout.case.input = {"query": "Cancel XEHM4B"}
    estimator = ExperienceGradientEstimator(viking_fs=fs, vlm=model, gates=list(gates))
    return estimator, item, context, policies, model, fs


@pytest.mark.asyncio
@pytest.mark.parametrize("repaired", [True, False])
async def test_identifier_gate_uses_real_extract_loop_bounded_repair(
    setup_observation, monkeypatch, repaired
):
    monkeypatch.setattr(ExperienceIdentifierGate, "ENABLED", True)
    estimator, item, context, policies, model, fs = harness(
        setup_observation, monkeypatch, [create(BAD), create(GOOD if repaired else BAD)]
    )
    if repaired:
        (proposal,) = await estimator.estimate(item, policies, context)
        assert proposal.after_file.content == GOOD
        assert proposal.metadata["experience_proposal_gate"]["gates"] == [
            "ExperienceIdentifierGate"
        ]
        assert (
            proposal.metadata["proposal_source_sessions"][0]["source_session_uri"]
            == item.metadata["source_session_uri"]
        )
        assert proposal.links[0].to_uri == item.trajectories[0].uri
        plan = PolicyUpdatePlan(
            items=case_experience_plan_items(operations(content=GOOD), [proposal], policies)
        )
        async with policies.lock() as lease:
            applied = await MemoryFilePolicyUpdater(viking_fs=fs).apply(
                plan, policies, context.request_context, transaction_handle=lease
            )
        assert not applied.errors
        assert MemoryFileUtils.read(fs.files[TARGET], uri=TARGET).content == GOOD
    else:
        with pytest.raises(ValueError, match="Provider operation validation failed after repair"):
            await estimator.estimate(item, policies, context)
        assert TARGET not in fs.files
    assert model.get_completion_async.await_count == 2
    messages = model.get_completion_async.await_args_list[1].kwargs["messages"]
    assert {"role": "assistant", "content": create(BAD)} in messages
    feedback = next(m["content"] for m in messages if "Errors: [{" in m["content"])
    assert "ExperienceIdentifierGate" in feedback and "XEHM4B" in feedback
    assert "experience_synthesis_noop" not in item.metadata


@pytest.mark.asyncio
async def test_gate_checks_whole_materialized_patch_not_only_edit(setup_observation, monkeypatch):
    monkeypatch.setattr(ExperienceIdentifierGate, "ENABLED", True)
    old = BAD + '\nother = tell("old reminder")'
    bad = 'experience.content.edit(search="old reminder", replace="new reminder")\nsdk.commit()'
    good = (
        'experience.content.edit(search="Cancel XEHM4B", replace="Verify the selected reservation")\n'
        'experience.content.edit(search="old reminder", replace="new reminder")\nsdk.commit()'
    )
    estimator, item, context, policies, model, fs = harness(
        setup_observation, monkeypatch, [bad, good], old=old
    )
    (proposal,) = await estimator.estimate(item, policies, context)
    assert "XEHM4B" not in proposal.after_file.content
    assert "new reminder" in proposal.after_file.content
    assert proposal.before_file.content == old and proposal.base_version == 3
    assert MemoryFileUtils.read(fs.files[TARGET], uri=TARGET).content == old
    assert model.get_completion_async.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["sdk.commit()", create(GOOD)])
async def test_explicit_noop_and_generalized_candidate_need_no_repair(
    setup_observation, monkeypatch, reply
):
    estimator, item, context, policies, model, _fs = harness(
        setup_observation, monkeypatch, [reply]
    )
    proposals = await estimator.estimate(item, policies, context)
    assert bool(proposals) is (reply != "sdk.commit()")
    assert model.get_completion_async.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reminder", ["Cancel XEHM4B", "Check the 24-hour cancellation window"])
@pytest.mark.parametrize("patch", [False, True])
async def test_identifier_gate_is_disabled_by_default(
    setup_observation, monkeypatch, reminder, patch
):
    assert ExperienceIdentifierGate.ENABLED is False
    assert "ENABLED" not in {f.name for f in fields(ExperienceIdentifierGate)}
    source = f'dag = workflow("review")\nr = tell({reminder!r})'
    old = source + '\nother = tell("old reminder")' if patch else None
    reply = (
        'experience.content.edit(search="old reminder", replace="new reminder")\nsdk.commit()'
        if patch
        else create(source)
    )
    estimator, item, context, policies, model, _fs = harness(
        setup_observation, monkeypatch, [reply], old=old
    )
    item.rollout.case.input = {"query": reminder}
    (proposal,) = await estimator.estimate(item, policies, context)
    assert proposal.after_file.content == (
        old.replace("old reminder", "new reminder") if patch else source
    )
    assert proposal.metadata["experience_proposal_gate"]["gates"] == []
    model.get_completion_async.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("repairable", [True, False])
async def test_custom_gate_uses_existing_repair_path_without_quality_replay(
    setup_observation, monkeypatch, repairable
):
    class TestGate:
        ENABLED = True
        calls = 0

        async def validate(self, plan, *args):
            self.calls += 1
            assert not setup_observation[0].policy_lock.locked()
            if self.calls == 1:
                if not repairable:
                    raise RuntimeError("checker unavailable")
                raise ExperienceGateRejected(
                    [{"gate": type(self).__name__, "error": "unjustified restriction"}]
                )
            return plan

    gate = TestGate()
    estimator, item, context, policies, model, _fs = harness(
        setup_observation, monkeypatch, [create(GOOD), create(GOOD)], gates=[gate]
    )
    if repairable:
        (proposal,) = await estimator.estimate(item, policies, context)
        assert proposal.metadata["experience_proposal_gate"]["gates"] == ["TestGate"]
        assert gate.calls == model.get_completion_async.await_count == 2
        assert "unjustified restriction" in json.dumps(
            model.get_completion_async.await_args.kwargs["messages"]
        )
    else:
        with pytest.raises(RuntimeError, match="checker unavailable"):
            await estimator.estimate(item, policies, context)
        assert gate.calls == model.get_completion_async.await_count == 1


@pytest.mark.asyncio
async def test_invalid_dag_is_not_silently_discarded_after_repair(setup_observation, monkeypatch):
    invalid = 'dag = workflow("review")\nr = tell("verify")\nmissing.then(r)'
    estimator, item, context, policies, model, fs = harness(
        setup_observation, monkeypatch, [create(invalid), create(invalid)]
    )
    with pytest.raises(ValueError, match="Provider operation validation failed after repair"):
        await estimator.estimate(item, policies, context)
    assert model.get_completion_async.await_count == 2
    assert TARGET not in fs.files


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["valid", "cannot_repair", "invalid_dag", "identifier_rejected"]
)
@pytest.mark.parametrize("loop_repaired", [True, False])
async def test_edit_repair_runs_before_provider_retry_and_revalidates_all_gates(
    setup_observation, monkeypatch, outcome, loop_repaired
):
    monkeypatch.setattr(ExperienceIdentifierGate, "ENABLED", True)
    old = 'dag = workflow("review")\nr = tell("old reminder")'
    bad = 'experience.content.edit(search="missing reminder", replace="new reminder")\nsdk.commit()'
    replacement = {
        "valid": "new reminder",
        "invalid_dag": "invalid\nstring",
        "identifier_rejected": "Cancel XEHM4B",
    }.get(outcome, "new reminder")
    repair = (
        {"status": "cannot_repair", "reason": "Unclear intended edit"}
        if outcome == "cannot_repair"
        else {
            "status": "repaired",
            "repairs": [{"edit_index": 0, "search": "old reminder", "replace": replacement}],
        }
    )
    good = 'experience.content.edit(search="old reminder", replace="new reminder")\nsdk.commit()'
    responses = [bad, json.dumps(repair)]
    if outcome != "valid":
        responses.append(good if loop_repaired else bad)
        if not loop_repaired:
            responses.append(json.dumps(repair))  # The new draft gets its own local repair.
    estimator, item, context, policies, model, fs = harness(
        setup_observation, monkeypatch, responses, old=old
    )
    if outcome == "valid" or loop_repaired:
        (proposal,) = await estimator.estimate(item, policies, context)
        assert proposal.before_file.content == old
        assert proposal.after_file.content == old.replace("old reminder", "new reminder")
        assert proposal.base_version == 3
        assert proposal.links[0].to_uri == item.trajectories[0].uri
        assert proposal.metadata["experience_proposal_gate"]["gates"] == [
            "ExperienceIdentifierGate"
        ]
    else:
        with pytest.raises(ValueError, match="Provider operation validation failed after repair"):
            await estimator.estimate(item, policies, context)
    # Each failed draft gets isolated repair; the existing loop bounds SDK retries.
    assert model.get_completion_async.await_count == len(responses)
    if outcome != "valid":
        fallback = model.get_completion_async.await_args_list[2].kwargs["messages"]
        assert {"role": "assistant", "content": bad} in fallback
        feedback = "\n".join(m["content"] or "" for m in fallback)
        assert {
            "cannot_repair": "cannot_repair: Unclear intended edit",
            "invalid_dag": "invalid_repair:",
            "identifier_rejected": "candidate_validation_failed:",
        }[outcome] in feedback
        if outcome == "identifier_rejected":
            assert "ExperienceIdentifierGate" in feedback
    request = model.get_completion_async.await_args_list[1].kwargs
    assert len(request["messages"]) == 2 and request["tools"] == []
    payload = json.loads(request["messages"][1]["content"])
    assert payload["targets"][0]["original_text"] == old
    assert payload["errors"][0]["gate"] == "candidate_content"
    assert "XEHM4B" not in request["messages"][1]["content"]
    assert MemoryFileUtils.read(fs.files[TARGET], uri=TARGET).content == old


@pytest.mark.asyncio
async def test_default_candidate_validation_preserves_sources_without_quality_evaluators(
    setup_observation, monkeypatch
):
    import openviking.session.train as train

    assert not hasattr(train, "ExperienceImprovementGate")

    def unexpected(*args, **kwargs):
        pytest.fail("Candidate validation must not initialize quality replay evaluators")

    monkeypatch.setattr("openviking.models.jev.JevClient", unexpected)
    monkeypatch.setattr(
        "openviking.service.experience_dag_decider.ExperienceDagDecider", unexpected
    )
    estimator, item, context, policies, model, _fs = harness(
        setup_observation, monkeypatch, [create(GOOD)]
    )
    (proposal,) = await estimator.estimate(item, policies, context)
    assert proposal.metadata["experience_proposal_gate"]["gates"] == []
    (source,) = proposal.metadata["proposal_source_sessions"]
    assert source["source_session_uri"] == item.metadata["source_session_uri"]
    assert source["case_name"] == item.rollout.case.name
    assert "evidence" not in source and "experience_execution" not in source
    model.get_completion_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_gate_rejected_repair_does_not_prevent_repair_of_the_next_draft(
    setup_observation, monkeypatch
):
    monkeypatch.setattr(ExperienceIdentifierGate, "ENABLED", True)
    old = 'dag = workflow("review")\nr = tell("old reminder")'
    bad = 'experience.content.edit(search="missing reminder", replace="new reminder")\nsdk.commit()'

    def repair(replacement):
        return json.dumps(
            {
                "status": "repaired",
                "repairs": [{"edit_index": 0, "search": "old reminder", "replace": replacement}],
            }
        )

    estimator, item, context, policies, model, fs = harness(
        setup_observation,
        monkeypatch,
        [bad, repair("Cancel XEHM4B"), bad, repair("new reminder")],
        old=old,
    )
    (proposal,) = await estimator.estimate(item, policies, context)
    assert proposal.before_file.content == old
    assert proposal.after_file.content == old.replace("old reminder", "new reminder")
    assert model.get_completion_async.await_count == 4
    calls = model.get_completion_async.await_args_list
    assert "ExperienceIdentifierGate" in "\n".join(
        m["content"] for m in calls[2].kwargs["messages"]
    )
    for i in (1, 3):
        request = calls[i].kwargs
        assert len(request["messages"]) == 2 and request["tools"] == []
        assert json.loads(request["messages"][1]["content"])["targets"][0]["original_text"] == old
    assert MemoryFileUtils.read(fs.files[TARGET], uri=TARGET).content == old
