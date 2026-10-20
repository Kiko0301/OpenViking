from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_case_experience_pipeline import observe, presented_trajectories
from test_case_experience_pipeline import setup_observation as setup_observation
from test_gradient_estimator_component import ROOT, analysis, operations, validated_operations

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.memory_type_registry import get_default_registry
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.train import (
    ExperienceGradientEstimator,
    MemoryFilePolicyUpdater,
    PatchMergePolicyOptimizerContext,
)
from openviking.session.train.components.trajectory_lifecycle import archive_trajectory
from openviking.session.train.components.trajectory_optimizer import (
    TrajectoryExperienceOptimizer,
)


def ann_result(*runs):
    return SimpleNamespace(
        memories=[
            SimpleNamespace(uri=run.trajectories[0].uri, score=1.0 - index / 100)
            for index, run in enumerate(runs)
        ]
    )


DAG = 'dag = workflow("cancel")\nc = check("duplicate requested")\nr = tell("verify duplicate identity")\nc.then(r)'
TARGET = f"{ROOT}/case_a.md"


async def make_run(setup, *, task="a", passed=False, session="current", state="applied"):
    fs, _ctx, _model, analyzer, context, policies = setup
    run = analysis(session=f"viking://session/{session}/archives/001", passed=passed)
    spec = asdict(run.rollout.case)
    spec["task_signature"] = f"task:{task}"
    spec["input"] = {"user_query": f"request {task}", "initial_state": {"version": 1}}
    run.metadata["task_spec"] = spec
    signals = await observe(analyzer, run, policies, context)
    memory = run.metadata["trajectory_memory"]
    memory.extra_fields["experience_update"]["state"] = state
    fs.files[memory.uri] = MemoryFileUtils.write(memory)
    if state in {"applied", "no_op"}:
        archived = await archive_trajectory(
            memory, viking_fs=fs, ctx=_ctx, registry=get_default_registry(), vikingdb=None
        )
        memory.uri = archived.uri
        run.trajectories[0].uri = archived.uri
    return run, signals


async def existing_policy(setup):
    fs, _ctx, _model, _analyzer, _context, policies = setup
    fs.files[TARGET] = MemoryFileUtils.write(
        MemoryFile(
            uri=TARGET,
            content=DAG,
            memory_type="experiences",
            extra_fields={
                "experience_name": "case_a",
                "case_name": "case_a",
                "version": 1,
            },
        )
    )
    return await policies.reload()


async def make_plan(setup, signals, policies):
    fs, ctx, *_ = setup
    estimator = SimpleNamespace(estimate=AsyncMock(return_value=[]))
    plan = await TrajectoryExperienceOptimizer(fs, estimator).plan(
        signals, policies, PatchMergePolicyOptimizerContext(ctx)
    )
    estimator.estimate.assert_awaited_once()
    return plan, estimator.estimate.await_args.args[0]


@pytest.mark.asyncio
async def test_unrelated_current_success_does_not_hide_missing_task_comparison(setup_observation):
    setup = setup_observation
    fs = setup[0]
    history_b, _ = await make_run(setup, task="b", session="history-b", passed=True)
    history_c, _ = await make_run(setup, task="c", session="history-c", passed=True)
    policies = await existing_policy(setup)
    success_a, a = await make_run(setup, task="a", session="success-a", passed=True)
    failed_b, b = await make_run(setup, task="b", session="failed-b")
    fs.find.return_value = ann_result(history_b, history_c)
    fs.ls = AsyncMock(wraps=fs.ls)
    fs.read_file = AsyncMock(wraps=fs.read_file)

    _plan, selected = await make_plan(setup, [*a, *b], policies)

    assert [t.uri for t in selected.trajectories] == [
        success_a.trajectories[0].uri,
        failed_b.trajectories[0].uri,
        history_b.trajectories[0].uri,
    ]
    assert selected.metadata["current_trajectory_uris"] == [
        success_a.trajectories[0].uri,
        failed_b.trajectories[0].uri,
    ]
    assert {call.args[0] for call in fs.read_file.await_args_list} == {
        history_b.trajectories[0].uri,
    }
    fs.ls.assert_not_awaited()
    fs.find.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_ann_keeps_current_success_and_failure_without_historical_reads(
    setup_observation,
):
    setup = setup_observation
    history, _ = await make_run(setup, session="history", passed=True)
    policies = await existing_policy(setup)
    success, a = await make_run(setup, session="success", passed=True)
    failure, b = await make_run(setup, session="failure")
    setup[0].read_file = AsyncMock(side_effect=AssertionError("Current success is sufficient"))

    _plan, selected = await make_plan(setup, [*a, *b], policies)

    assert [t.uri for t in selected.trajectories] == [
        success.trajectories[0].uri,
        failure.trajectories[0].uri,
    ]
    setup[0].read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_noop_success_is_discovered_after_reload_without_rewriting_experience(
    setup_observation,
):
    setup = setup_observation
    fs = setup[0]
    policies = await existing_policy(setup)
    success, signals = await make_run(setup, session="success", passed=True, state="no_op")
    before = dict(fs.files)

    plan, _selected = await make_plan(setup, signals, policies)
    assert plan.items == []
    assert fs.files == before

    policies = await policies.reload()
    fs.find.return_value = ann_result(success)
    failure, signals = await make_run(setup, session="next-failure")
    plan, selected = await make_plan(setup, signals, policies)
    assert plan.items == []
    assert [t.uri for t in selected.trajectories] == [
        failure.trajectories[0].uri,
        success.trajectories[0].uri,
    ]


@pytest.mark.asyncio
async def test_new_experience_has_no_success_index_or_source_links(setup_observation, monkeypatch):
    setup = setup_observation
    fs, ctx, _model, _analyzer, _context, policies = setup
    success, signals = await make_run(setup, session="success", passed=True)

    async def synthesize(self, provider, context):
        provider.read_file = AsyncMock(return_value=None)
        payload = await presented_trajectories(provider)
        assert len(payload) == 1 and payload[0]["outcome"] == "success"
        return await validated_operations(provider, operations(content=DAG))

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    plan = await TrajectoryExperienceOptimizer(fs, ExperienceGradientEstimator(viking_fs=fs)).plan(
        signals, policies, PatchMergePolicyOptimizerContext(ctx)
    )
    async with policies.lock() as lease:
        result = await MemoryFilePolicyUpdater(viking_fs=fs).apply(
            plan, policies, ctx, transaction_handle=lease
        )
    assert result.errors == []
    stored = MemoryFileUtils.read(fs.files[TARGET], uri=TARGET)
    assert stored.extra_fields["case_name"] == "case_a"
    assert "success_trajectory_refs" not in stored.extra_fields
    assert stored.links == []
    assert "success_trajectory_refs" not in stored.content

    # Actual prefetch presentation must hide the bookkeeping, not just the DAG body.
    from openviking.session.memory.experience.experience_context import (
        AgentExperienceContextProvider,
    )

    provider = AgentExperienceContextProvider(
        messages=[],
        case=success.rollout.case,
        target_uri=TARGET,
        source_session_uri=success.metadata["source_session_uri"],
        trajectories=success.trajectories,
    )
    provider._ctx, provider._viking_fs = ctx, fs
    messages = provider.format_prefetch_messages(await provider.prefetch())
    assert not any("success_trajectory_refs" in message["content"] for message in messages)


@pytest.mark.asyncio
async def test_current_success_does_not_scan_or_create_empty_experience(setup_observation):
    setup = setup_observation
    success, signals = await make_run(setup, passed=True)
    setup[0].ls = AsyncMock(side_effect=AssertionError("No directory scans"))
    plan, selected = await make_plan(setup, signals, setup[-1])
    assert plan.items == []
    assert selected.trajectories == success.trajectories
    assert TARGET not in setup[0].files


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "failure",
        "wrong_case",
        "wrong_format",
        "pending",
        "failed_update",
        "missing_update_state",
        "bad_source",
        "cross_user",
        "cross_case",
        "hidden",
    ],
)
async def test_unusable_history_does_not_enter_comparison(setup_observation, defect):
    setup = setup_observation
    fs = setup[0]
    history, _ = await make_run(setup, session="history", passed=True)
    memory = history.metadata["trajectory_memory"]
    original_uri = memory.uri
    if defect == "missing":
        del fs.files[memory.uri]
    elif defect in {"cross_user", "cross_case", "hidden"}:
        del fs.files[original_uri]
        memory.uri = {
            "cross_user": original_uri.replace("/user/u/", "/user/other/"),
            "cross_case": original_uri.replace("/case_a/", "/other/"),
            "hidden": original_uri.rsplit("/", 1)[0] + "/.hidden.md",
        }[defect]
        fs.files[memory.uri] = MemoryFileUtils.write(memory)
        # Even a stale ANN hit must not cause out-of-directory/hidden reads.
        fs.ls = AsyncMock(side_effect=AssertionError("No directory scan"))
    else:
        if defect == "failure":
            memory.extra_fields["outcome"] = "failure"
        elif defect == "wrong_case":
            memory.extra_fields["case_name"] = "other"
        elif defect == "wrong_format":
            memory.extra_fields["trajectory_format"] = "unsupported"
        elif defect == "pending":
            memory.extra_fields["experience_update"]["state"] = "pending"
        elif defect == "failed_update":
            memory.extra_fields["experience_update"]["state"] = "failed"
        elif defect == "missing_update_state":
            memory.extra_fields.pop("experience_update")
        elif defect == "bad_source":
            memory.links = []
        fs.files[memory.uri] = MemoryFileUtils.write(memory)
    policies = await existing_policy(setup)
    failure, signals = await make_run(setup, session="failure")
    fs.find.return_value = SimpleNamespace(memories=[SimpleNamespace(uri=memory.uri, score=0.9)])
    fs.read_file = AsyncMock(wraps=fs.read_file)
    _plan, selected = await make_plan(setup, signals, policies)
    assert [t.uri for t in selected.trajectories] == [failure.trajectories[0].uri]
    if defect in {"cross_user", "cross_case", "hidden"}:
        fs.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_history_read_errors_are_not_hidden(setup_observation):
    setup = setup_observation
    history, _ = await make_run(setup, session="history", passed=True)
    policies = await existing_policy(setup)
    _failure, signals = await make_run(setup, session="failure")
    setup[0].find.return_value = ann_result(history)
    setup[0].read_file = AsyncMock(side_effect=PermissionError("denied"))
    with pytest.raises(PermissionError, match="denied"):
        await make_plan(setup, signals, policies)


@pytest.mark.asyncio
async def test_ann_does_not_require_task_spec(setup_observation):
    setup = setup_observation
    history, _ = await make_run(setup, session="history", passed=True)
    policies = await existing_policy(setup)
    _failure, signals = await make_run(setup, session="failure")
    signals[0].analysis.trajectories[0].metadata.pop("task_spec")
    setup[0].find.return_value = ann_result(history)
    _plan, selected = await make_plan(setup, signals, policies)
    assert len(selected.trajectories) == 2
    setup[0].find.assert_awaited_once()


@pytest.mark.asyncio
async def test_ann_ranking_wins_over_recency_after_processed_state_validation(setup_observation):
    setup = setup_observation
    policies = await existing_policy(setup)
    older, _ = await make_run(setup, session="older", passed=True)
    newer, _ = await make_run(setup, session="newer", passed=True, state="no_op")
    pending, _ = await make_run(setup, session="pending", passed=True, state="pending")
    for day, run in enumerate((older, newer), 1):
        memory = run.metadata["trajectory_memory"]
        memory.extra_fields["created_at"] = f"2026-01-0{day}T00:00:00+00:00"
        setup[0].files[memory.uri] = MemoryFileUtils.write(memory)
    setup[0].find.return_value = ann_result(pending, older, newer)
    failure, signals = await make_run(setup, session="failure")
    _plan, selected = await make_plan(setup, signals, policies)
    assert [t.uri for t in selected.trajectories] == [
        failure.trajectories[0].uri,
        older.trajectories[0].uri,
    ]


@pytest.mark.asyncio
async def test_individual_ann_queries_deduplicate_shared_successes(setup_observation):
    setup = setup_observation
    first, _ = await make_run(setup, task="a", session="success-a", passed=True)
    second, _ = await make_run(setup, task="b", session="success-b", passed=True)
    policies = await existing_policy(setup)
    failure_a, a = await make_run(setup, task="a", session="failure-a")
    failure_b, b = await make_run(setup, task="b", session="failure-b")
    again_a, again = await make_run(setup, task="a", session="another-a")
    setup[0].find.side_effect = [ann_result(first), ann_result(second), ann_result(first)]
    _plan, selected = await make_plan(setup, [*a, *b, *again], policies)
    assert [t.uri for t in selected.trajectories] == [
        failure_a.trajectories[0].uri,
        failure_b.trajectories[0].uri,
        again_a.trajectories[0].uri,
        first.trajectories[0].uri,
        second.trajectories[0].uri,
    ]

    assert setup[0].find.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [FileNotFoundError("missing"), PermissionError("denied")])
async def test_missing_history_directory_is_empty_but_access_errors_propagate(
    setup_observation, failure
):
    setup = setup_observation
    current, signals = await make_run(setup, session="current")
    setup[0].find = AsyncMock(side_effect=failure)
    if isinstance(failure, FileNotFoundError):
        plan, selected = await make_plan(setup, signals, setup[-1])
        assert plan.items == []
        assert selected.trajectories == current.trajectories
    else:
        with pytest.raises(PermissionError, match="denied"):
            await make_plan(setup, signals, setup[-1])


@pytest.mark.asyncio
async def test_history_needs_no_experience_file_or_reference_index(setup_observation):
    setup = setup_observation
    history, _ = await make_run(setup, session="success", passed=True, state="no_op")
    current, signals = await make_run(setup, session="current")
    setup[0].find.return_value = ann_result(history)
    plan, selected = await make_plan(setup, signals, setup[-1])
    assert plan.items == []
    assert TARGET not in setup[0].files
    assert [t.uri for t in selected.trajectories] == [
        current.trajectories[0].uri,
        history.trajectories[0].uri,
    ]


@pytest.mark.asyncio
async def test_ann_skips_duplicate_and_disappearing_hits(setup_observation):
    setup = setup_observation
    fs = setup[0]
    history, _ = await make_run(setup, session="success", passed=True)
    current, signals = await make_run(setup, session="current")
    uri = history.trajectories[0].uri
    missing = uri.rsplit("/", 1)[0] + "/gone.md"
    fs.find.return_value = SimpleNamespace(
        memories=[SimpleNamespace(uri=value, score=0.9) for value in [missing, missing, uri, uri]]
    )
    fs.ls = AsyncMock(side_effect=AssertionError("No directory scan"))
    fs.read_file = AsyncMock(wraps=fs.read_file)
    _plan, selected = await make_plan(setup, signals, setup[-1])
    assert [t.uri for t in selected.trajectories] == [current.trajectories[0].uri, uri]
    assert [call.args[0] for call in fs.read_file.await_args_list] == [missing, uri]
    fs.ls.assert_not_awaited()
