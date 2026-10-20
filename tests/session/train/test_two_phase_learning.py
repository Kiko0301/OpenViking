# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import asyncio
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from test_external_case_learning import cases as cases
from test_external_case_learning import noop_result
from test_external_case_learning import setup_observation as setup_observation
from test_train_framework import _case, _policy_set

from openviking.session.train import (
    ExperienceSet,
    ListCaseLoader,
    OfflinePolicyOptimizationPipeline,
    PipelineContext,
    PolicyApplyResult,
    PolicyUpdatePlan,
    RolloutTrainingResult,
)


def pipeline(trainer, learner):
    return OfflinePolicyOptimizationPipeline(
        snapshotter=SimpleNamespace(),
        rollout_executor=SimpleNamespace(),
        policy_trainer=trainer,
        experience_learner=learner,
    )


def result(policies=None, **metadata):
    return RolloutTrainingResult(
        analyses=[],
        gradients=[],
        plan=PolicyUpdatePlan(),
        apply_result=PolicyApplyResult(
            updated_policy_set=policies or ExperienceSet("", []),
        ),
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_pipeline_learns_once_after_all_epoch_commit_batches():
    order = []
    policies = _policy_set()
    batches = []

    async def collect(rollouts, policy_set, context):
        order.append("commit")
        batch = result(policy_set)
        batches.append(batch)
        return batch

    async def learn(collected, policy_set, context):
        assert order == ["commit", "commit"]
        assert collected == batches
        order.append("learn")
        return result(policy_set)

    runner = pipeline(SimpleNamespace(train_rollouts=collect), SimpleNamespace(learn=learn))
    runner._rollout_batch = AsyncMock(return_value=([], "snapshot"))
    epoch = await runner._run_training_epoch(
        epoch=0,
        case_loader=ListCaseLoader([_case(), _case()], batch_size=1),
        policy_set=policies,
        ctx=PipelineContext(),
    )
    assert order == ["commit", "commit", "learn"]
    assert epoch.metadata["session_commit_cost_seconds"] >= 0
    assert epoch.metadata["experience_learn_cost_seconds"] >= 0


@pytest.mark.asyncio
async def test_external_rollouts_use_both_pipeline_phases():
    first = result()
    second = result()
    collector = SimpleNamespace(train_rollouts=AsyncMock(return_value=first))
    learner = SimpleNamespace(learn=AsyncMock(return_value=second))
    runner = pipeline(collector, learner)
    policies, context = _policy_set(), PipelineContext()
    assert await runner.train_from_rollouts([], policies, context) is second
    learner.learn.assert_awaited_once_with([first], policies, context)


@pytest.mark.asyncio
async def test_server_progress_counts_terminal_cases_not_trajectories(cases):
    entered, finish = asyncio.Event(), asyncio.Event()
    snapshots = []

    async def work(**kwargs):
        entered.set()
        await finish.wait()
        if kwargs["case_name"] == "case_0":
            raise ValueError("bad Case")
        return await noop_result(cases, **kwargs)

    async def progress(snapshot):
        snapshots.append(snapshot)

    cases.compressor._learn_one_trajectory_case = AsyncMock(side_effect=work)
    task = asyncio.create_task(
        cases.compressor.learn_pending_experiences(
            ctx=cases.ctx,
            trajectory_uris=[r.uri for r in cases.records],
            on_progress=progress,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert snapshots[0]["total"] == 6
        assert snapshots[0]["completed"] == 0
        assert snapshots[-1]["running"] > 0
        assert snapshots[-1]["completed"] == 0
        finish.set()
        learned = await task
        assert learned["failed_count"] == 1
        assert snapshots[-1] == {
            "total": 6,
            "completed": 6,
            "running": 0,
            "applied": 0,
            "no_op": 5,
            "failed": 1,
        }
        assert [p["completed"] for p in snapshots] == sorted(p["completed"] for p in snapshots)
    finally:
        finish.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_remote_learner_renders_real_counts_and_keeps_commit_errors(monkeypatch, capsys):
    from openviking.session.train.components import experience_learner as module

    updates = []

    class Printer:
        def __init__(self, *, total, **kwargs):
            assert total == 3

        def update_counts(self, **kwargs):
            updates.append(kwargs)

        def finish(self):
            updates.append("finished")

    monkeypatch.setattr(module, "ProgressPrinter", Printer)
    snapshots = [
        {"total": 3, "completed": 0, "running": 2, "applied": 0, "no_op": 0, "failed": 0},
        {"total": 3, "completed": 1, "running": 2, "applied": 1, "no_op": 0, "failed": 0},
        {"total": 3, "completed": 3, "running": 0, "applied": 1, "no_op": 1, "failed": 1},
    ]
    outcome = {
        "case_count": 3,
        "trajectory_count": 4,
        "applied_count": 1,
        "no_op_count": 1,
        "failed_count": 1,
        "written_experience_uris": ["exp-a"],
        "results": [{"case_name": "bad", "state": "failed", "error": "invalid DAG"}],
    }
    task_updates = [{"status": "running", "meta": {"progress": p}} for p in snapshots[:2]] + [
        {"status": "completed", "meta": {"progress": snapshots[-1]}, "result": outcome}
    ]
    client = SimpleNamespace(
        learn_experiences=AsyncMock(return_value={"task_id": "learn-1"}),
        get_task=AsyncMock(side_effect=task_updates),
    )
    batch = result()
    batch.apply_result.metadata["commit_results"] = [
        {"trajectory_uris": ["traj-a", "traj-a"]},
        {"trajectory_uris": ["do-not-learn"], "error": "commit failed"},
    ]
    batch.apply_result.errors = ["commit failed"]
    learner = module.RemoteExperienceLearner(client=client, poll_interval_seconds=0.001)
    learned = await learner.learn([batch], _policy_set())
    assert client.learn_experiences.await_args.kwargs["trajectory_uris"] == ["traj-a"]
    assert client.learn_experiences.await_args.kwargs["wait"] is False
    assert updates == [
        {"completed": 0, "failed": 0, "running": 2},
        {"completed": 1, "failed": 0, "running": 2},
        {"completed": 3, "failed": 0, "running": 0},
        "finished",
    ]
    assert learned.apply_result.written_uris == ["exp-a"]
    assert learned.apply_result.errors == ["commit failed"]
    assert learned.metadata["experience_learning"]["failed_count"] == 1
    assert learned.metadata["experience_learning"]["results"][0]["state"] == "failed"
    assert learned.metadata["experience_learning"]["task_id"] == "learn-1"
    output = capsys.readouterr().out
    assert "skipped=1" in output
    assert "invalid DAG" not in output
    assert "failed=" not in output


@pytest.mark.asyncio
async def test_empty_phase_one_does_not_scan_unrelated_pending_trajectories():
    from openviking.session.train.components.experience_learner import RemoteExperienceLearner

    client = SimpleNamespace(learn_experiences=AsyncMock(), get_task=AsyncMock())
    learned = await RemoteExperienceLearner(client).learn([result()], _policy_set())
    client.learn_experiences.assert_not_called()
    assert learned.metadata["experience_learning"]["case_count"] == 0


@pytest.mark.asyncio
async def test_poll_read_errors_do_not_resubmit_learning(monkeypatch):
    import httpx

    from openviking.session.train.components import session_commit
    from openviking.session.train.components.experience_learner import RemoteExperienceLearner

    monkeypatch.setattr(session_commit.asyncio, "sleep", AsyncMock())
    client = SimpleNamespace(
        learn_experiences=AsyncMock(return_value={"task_id": "learn-1"}),
        get_task=AsyncMock(
            side_effect=[httpx.ReadError("")] * 10
            + [
                {"status": "completed", "result": {"case_count": 1, "written_experience_uris": []}},
            ]
        ),
    )
    batch = result()
    batch.apply_result.metadata["commit_results"] = [{"trajectory_uris": ["traj-a"]}]
    learned = await RemoteExperienceLearner(client).learn([batch], _policy_set())
    assert learned.apply_result.errors == []
    client.learn_experiences.assert_awaited_once()
    assert client.get_task.await_count == 11


@pytest.mark.asyncio
async def test_wait_cancellation_does_not_fake_a_complete_bar(monkeypatch):
    from openviking.session.train.components import experience_learner as module

    finished = []

    class Printer:
        def __init__(self, **kwargs):
            pass

        def update_counts(self, **kwargs):
            finished.append(kwargs)

        def finish(self):
            finished.append("finished")

    monkeypatch.setattr(module, "ProgressPrinter", Printer)
    client = SimpleNamespace(
        learn_experiences=AsyncMock(return_value={"task_id": "learn-1"}),
        get_task=AsyncMock(
            side_effect=[
                {
                    "status": "running",
                    "meta": {
                        "progress": {
                            "total": 2,
                            "completed": 0,
                            "running": 1,
                            "applied": 0,
                            "no_op": 0,
                            "failed": 0,
                        }
                    },
                },
                asyncio.CancelledError(),
            ]
        ),
    )
    batch = result()
    batch.apply_result.metadata["commit_results"] = [{"trajectory_uris": ["traj-a"]}]
    events = SimpleNamespace(record=AsyncMock())
    learner = module.RemoteExperienceLearner(
        client, poll_interval_seconds=0.001, event_recorder=events
    )
    with pytest.raises(asyncio.CancelledError):
        await learner.learn([batch], _policy_set())
    assert finished == [{"completed": 0, "failed": 0, "running": 1}, "finished"]
    done = events.record.await_args
    assert done.args == ("experience_learn_done",)
    assert done.kwargs["task_id"] == "learn-1"
    assert done.kwargs["incomplete"] is True
    assert done.kwargs["trajectory_count"] is None
    assert done.kwargs["trigger_trajectory_count"] == 1
    assert done.kwargs["selection_mode"] == "case_inbox"
    assert done.kwargs["error"].startswith("CancelledError:")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "timeout", "completed"])
async def test_incomplete_remote_task_aborts_epoch_and_retains_diagnostics(
    monkeypatch, status, capsys
):
    from openviking.session.train.components import experience_learner as module

    progress = {"total": 3, "completed": 1, "running": 2, "applied": 1, "no_op": 0, "failed": 0}

    async def wait(*args, **kwargs):
        await kwargs["on_update"]({"meta": {"progress": progress}})
        return {"status": status, "error": "stopped", "result": None}

    monkeypatch.setattr(module, "wait_for_task", wait)
    client = SimpleNamespace(learn_experiences=AsyncMock(return_value={"task_id": "learn-1"}))
    batch = result()
    commits = [{"trajectory_uris": ["traj-a"]}]
    batch.apply_result.metadata["commit_results"] = commits
    events = SimpleNamespace(record=AsyncMock())
    trainer = SimpleNamespace(train_rollouts=AsyncMock(return_value=batch))
    runner = pipeline(trainer, module.RemoteExperienceLearner(client, event_recorder=events))
    runner._rollout_batch = AsyncMock(return_value=([], "snapshot"))
    runner._run_epoch_evaluation_pass = AsyncMock()
    expected_type = ValueError if status == "completed" else RuntimeError
    with pytest.raises(expected_type, match="learn-1"):
        await runner.train(
            ListCaseLoader([_case()]),
            _policy_set(),
            PipelineContext(max_epochs=2, lifecycle_hooks=[]),
        )

    trainer.train_rollouts.assert_awaited_once()  # Do not enter the next epoch either.
    runner._run_epoch_evaluation_pass.assert_not_awaited()
    assert batch.apply_result.metadata["commit_results"] == commits
    assert batch.apply_result.errors == []
    done = events.record.await_args
    assert done.args == ("experience_learn_done",)
    learning = done.kwargs
    assert learning["task_id"] == "learn-1"
    assert learning["incomplete"] is True
    assert learning["case_count"] == 3
    assert learning["applied_count"] == 1
    assert learning["failed_count"] == 0
    assert learning["progress"] == progress
    assert learning["error"].startswith(f"{expected_type.__name__}: learn-1:")
    assert "stopped" not in capsys.readouterr().out


@pytest.mark.parametrize("timeout", [None, 1.0])
def test_learning_wait_accepts_no_deadline_or_a_positive_timeout(timeout):
    from openviking.session.train.components.experience_learner import RemoteExperienceLearner

    learner = RemoteExperienceLearner(SimpleNamespace(), timeout_seconds=timeout)
    assert learner.timeout_seconds == timeout


@pytest.mark.parametrize("field", ["timeout_seconds", "poll_interval_seconds"])
@pytest.mark.parametrize("value", [0, -1])
def test_learning_wait_rejects_nonpositive_limits(field, value):
    from openviking.session.train.components.experience_learner import RemoteExperienceLearner

    with pytest.raises(ValueError, match=field):
        RemoteExperienceLearner(SimpleNamespace(), **{field: value})


@pytest.mark.asyncio
async def test_default_learning_wait_passes_old_deadline_and_blocks_evaluation(monkeypatch):
    from openviking.session.train.components import session_commit
    from openviking.session.train.components.experience_learner import RemoteExperienceLearner

    now = 0.0
    polling, complete = asyncio.Event(), asyncio.Event()
    outcome = {"case_count": 1, "trajectory_count": 1, "written_experience_uris": ["exp-a"]}

    async def query(task_id):
        nonlocal now
        assert task_id == "learn-1"
        now += 2000.0  # Virtual time: pass the old 1800s limit without real waiting.
        if now == 2000.0:
            return {"status": "running"}
        polling.set()
        await complete.wait()
        return {"status": "completed", "result": outcome}

    monkeypatch.setattr(
        session_commit,
        "asyncio",
        SimpleNamespace(
            get_running_loop=lambda: SimpleNamespace(time=lambda: now), sleep=AsyncMock()
        ),
    )
    client = SimpleNamespace(
        learn_experiences=AsyncMock(return_value={"task_id": "learn-1"}),
        get_task=AsyncMock(side_effect=query),
    )
    learner = RemoteExperienceLearner(client)
    assert learner.timeout_seconds is None
    batch = result()
    batch.apply_result.metadata["commit_results"] = [{"trajectory_uris": ["traj-a"]}]
    runner = pipeline(SimpleNamespace(train_rollouts=AsyncMock(return_value=batch)), learner)
    runner._rollout_batch = AsyncMock(return_value=([], "snapshot"))
    runner._run_epoch_evaluation_pass = AsyncMock(return_value=None)
    task = asyncio.create_task(
        runner.train(ListCaseLoader([_case()]), _policy_set(), PipelineContext(lifecycle_hooks=[]))
    )
    try:
        await asyncio.wait_for(polling.wait(), timeout=1)
        assert now > 1800
        assert not task.done()
        runner._run_epoch_evaluation_pass.assert_not_awaited()
        complete.set()
        learned = await asyncio.wait_for(task, timeout=1)
    finally:
        complete.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert learned.apply_result.written_uris == ["exp-a"]
    runner._run_epoch_evaluation_pass.assert_awaited_once()
    client.learn_experiences.assert_awaited_once()
    assert client.get_task.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["epoch", "external_rollouts"])
@pytest.mark.parametrize("stage", ["submission", "polling"])
@pytest.mark.parametrize(
    "exception_type", [httpx.ReadTimeout, httpx.ConnectError, asyncio.CancelledError]
)
async def test_learning_request_errors_and_cancellation_propagate(entry, stage, exception_type):
    from openviking.session.train.components.experience_learner import RemoteExperienceLearner

    error = exception_type("interrupted")
    client = SimpleNamespace(
        learn_experiences=AsyncMock(return_value={"task_id": "learn-1"}),
        get_task=AsyncMock(),
    )
    failed_call = client.learn_experiences if stage == "submission" else client.get_task
    failed_call.side_effect = error
    events = SimpleNamespace(record=AsyncMock())
    learner = RemoteExperienceLearner(client, event_recorder=events)
    batch = result()
    batch.apply_result.metadata["commit_results"] = [{"trajectory_uris": ["traj-a"]}]
    trainer = SimpleNamespace(train_rollouts=AsyncMock(return_value=batch))
    runner = pipeline(trainer, learner)
    runner._rollout_batch = AsyncMock(return_value=([], "snapshot"))
    runner._run_epoch_evaluation_pass = AsyncMock()
    context = PipelineContext(max_epochs=2, lifecycle_hooks=[])

    with pytest.raises(exception_type) as caught:
        if entry == "epoch":
            await runner.train(ListCaseLoader([_case()]), _policy_set(), context)
        else:
            await runner.train_from_rollouts([], _policy_set(), context)

    assert caught.value is error
    runner._run_epoch_evaluation_pass.assert_not_awaited()
    trainer.train_rollouts.assert_awaited_once()
    client.learn_experiences.assert_awaited_once()
    assert client.get_task.await_count == (1 if stage == "polling" else 0)
    done = events.record.await_args
    assert done.args == ("experience_learn_done",)
    assert done.kwargs["incomplete"] is True
    assert done.kwargs["task_id"] == ("learn-1" if stage == "polling" else None)
    assert done.kwargs["error"] == f"{exception_type.__name__}: interrupted"


@pytest.mark.asyncio
async def test_pipeline_empty_epoch_does_not_launch_learning():
    collector = SimpleNamespace(train_rollouts=AsyncMock())
    learner = SimpleNamespace(learn=AsyncMock())
    runner = pipeline(collector, learner)
    await runner._run_training_epoch(
        epoch=0,
        case_loader=ListCaseLoader([]),
        policy_set=_policy_set(),
        ctx=PipelineContext(),
    )
    collector.train_rollouts.assert_not_awaited()
    learner.learn.assert_not_awaited()


def test_phase_two_errors_and_timings_are_visible_in_epoch_report():
    from openviking.session.train.components.report_builder import PipelineReportBuilder
    from openviking.session.train.domain import PipelineEpochResult

    phase_one = [{"error": "commit failed"}]
    apply_result = result().apply_result
    apply_result.errors = ["commit failed", "experience learning failed"]
    apply_result.metadata = {
        "commit_results": phase_one,
        "experience_learning": {"failed_count": 1},
    }
    epoch = PipelineEpochResult(
        epoch=0,
        analyses=[],
        gradients=[],
        plan=PolicyUpdatePlan(),
        apply_result=apply_result,
        policy_snapshot_ids=[],
        metadata={"session_commit_cost_seconds": 2.0, "experience_learn_cost_seconds": 3.0},
    )
    report = PipelineReportBuilder().train_epoch_report(epoch)
    assert report["errors"] == ["commit failed", "experience learning failed"]
    assert report["session_commit_cost_seconds"] == 2.0
    assert report["experience_learn_cost_seconds"] == 3.0
    assert report["experience_learning"] == {"failed_count": 1}


@pytest.mark.asyncio
async def test_commit_summary_aggregates_batches_and_does_not_use_last_batch_counts(capsys):
    from openviking.session.train.components.experience_learner import RemoteExperienceLearner
    from openviking.session.train.components.report_builder import PipelineReportBuilder
    from openviking.session.train.domain import PipelineEpochResult

    first, second = result(), result()
    first.apply_result.metadata = {
        "commit_results": [
            {"task_status": "completed", "trajectory_uris": ["traj-a", "traj-b"]},
            {"task_status": "completed", "trajectory_uris": []},
            {"task_status": "failed", "error": "same error", "trajectory_uris": ["bad"]},
        ],
        "commit_summary": {"completed": 900},
    }
    second.apply_result.metadata = {
        "commit_results": [
            {"task_status": "completed", "trajectory_uris": ["traj-a"]},
            {"task_status": "completed", "trajectory_uris": []},
            {"task_status": "skipped", "trajectory_uris": []},
            {"task_status": "failed", "error": "same error", "trajectory_uris": []},
        ],
        "commit_summary": {"completed": 1000},
    }
    first.apply_result.errors = ["same error"]
    second.apply_result.errors = ["same error"]
    client = SimpleNamespace(
        learn_experiences=AsyncMock(return_value={"task_id": "learn-1"}),
        get_task=AsyncMock(
            return_value={
                "status": "completed",
                "result": {
                    "case_count": 1,
                    "trajectory_count": 2,
                    "applied_count": 0,
                    "no_op_count": 1,
                    "failed_count": 0,
                },
            }
        ),
    )

    learned = await RemoteExperienceLearner(client).learn([first, second], _policy_set())

    expected = {"completed": 4, "with_traj": 2, "without_traj": 2, "errors": 2}
    assert learned.apply_result.metadata["commit_summary"] == expected
    assert learned.metadata["commit_summary"] == expected
    assert learned.apply_result.errors == ["same error", "same error"]
    assert client.learn_experiences.await_args.kwargs["trajectory_uris"] == ["traj-a", "traj-b"]
    epoch = PipelineEpochResult(
        epoch=0,
        analyses=[],
        gradients=[],
        plan=learned.plan,
        apply_result=learned.apply_result,
        policy_snapshot_ids=["a", "b"],
    )
    report = PipelineReportBuilder().train_epoch_report(epoch)
    assert report["commit_summary"] == expected
    assert report["committed_rollout_count"] == 6
    assert report["skipped_rollout_count"] == 1
    assert report["errors"] == ["same error"]
    assert "[TRAIN_COMMIT]" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_progress_is_terminal_only_after_state_writes(cases, monkeypatch):
    from test_external_case_learning import states

    cases.compressor._learn_one_trajectory_case = AsyncMock(side_effect=partial(noop_result, cases))
    snapshots = []

    async def progress(p):
        terminal = sum(s["state"] in {"applied", "no_op", "failed"} for s in states(cases))
        assert p["completed"] * 2 <= terminal
        assert p["running"] <= 4
        snapshots.append(p)

    await cases.compressor.learn_pending_experiences(
        ctx=cases.ctx,
        trajectory_uris=[r.uri for r in cases.records],
        max_cases=2,
        on_progress=progress,
    )
    assert snapshots[0]["total"] == 2
    assert snapshots[-1]["completed"] == 2
    assert sum(s["state"] == "pending" for s in states(cases)) == 8
    snapshots.clear()
    await cases.compressor.learn_pending_experiences(
        ctx=cases.ctx,
        trajectory_uris=[r.uri for r in cases.records[:4]],
        on_progress=progress,
    )
    assert snapshots == [
        {"total": 0, "completed": 0, "running": 0, "applied": 0, "no_op": 0, "failed": 0}
    ]


@pytest.mark.asyncio
async def test_task_deadline_bounds_a_hung_poll_request():
    from openviking.session.train.components.session_commit import wait_for_task

    cancelled = asyncio.Event()

    async def query(task_id):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = await asyncio.wait_for(
        wait_for_task(
            SimpleNamespace(get_task=query),
            "learn-1",
            timeout_seconds=0.01,
            poll_interval_seconds=0.001,
        ),
        timeout=1,
    )
    assert task["status"] == "timeout"
    assert cancelled.is_set()
