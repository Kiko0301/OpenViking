# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import asyncio
import json
from collections import Counter
from unittest.mock import AsyncMock

import pytest
from test_case_experience_pipeline import setup_observation as setup_observation
from test_external_case_learning import cases as cases
from test_external_case_learning import learn, states
from test_gradient_estimator_component import ROOT, operations, validated_operations

from openviking.session.train import ExperienceGradientEstimator
from openviking.session.train.components.trajectory_lifecycle import trajectory_archive_uri


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 240)], indirect=True)
async def test_second_batch_failure_leaves_208_trajectories_unclaimed(cases, monkeypatch):
    cases.compressor.streaming_trainer_config.max_gradients_per_update = 16
    untouched = {r.uri: cases.fs.files[r.uri] for r in cases.records[32:]}
    batches = []

    async def synthesize(estimator, provider, context):
        batches.append(set(provider.current_trajectory_uris))
        current = states(cases)
        offset = (len(batches) - 1) * 16
        assert batches[-1] == {r.uri for r in cases.records[offset : offset + 16]}
        assert Counter(s["state"] for s in current) == (
            {"processing": 16, "pending": 224}
            if len(batches) == 1
            else {"applied": 16, "processing": 16, "pending": 208}
        )
        assert all(s["attempts"] == 0 and s["batch_id"] is None for s in current[offset + 16 :])
        if len(batches) == 1:
            return await validated_operations(
                provider,
                operations(
                    name="case_0",
                    content='dag = workflow("case")\nr = tell("chunk zero completed")',
                ),
            )
        messages = provider.format_prefetch_messages(await provider.prefetch())
        assert "chunk zero completed" in json.dumps(messages)
        raise ValueError("Could not find the exact text")

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    result = await asyncio.wait_for(learn(cases), timeout=10)
    assert len(batches) == 2
    assert result["failed_count"] == 1
    assert result["trajectory_count"] == 32
    assert result["written_experience_uris"] == [f"{ROOT}/case_0.md"]
    assert result["results"][0]["plan_metadata"]["chunk_count"] == 1
    current = states(cases)
    assert [s["state"] for s in current] == ["applied"] * 16 + ["failed"] * 16 + ["pending"] * 208
    assert [s["attempts"] for s in current] == [1] * 32 + [0] * 208
    assert len({s["batch_id"] for s in current[:16]}) == 1
    assert len({s["batch_id"] for s in current[16:32]}) == 1
    assert current[0]["batch_id"] != current[16]["batch_id"]
    assert all(cases.fs.files[uri] == before for uri, before in untouched.items())
    assert all(r.uri not in cases.fs.files for r in cases.records[:16])
    assert all(trajectory_archive_uri(r.uri) in cases.fs.files for r in cases.records[:16])
    cases.fs.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 5)], indirect=True)
async def test_resubmission_processes_pending_tail_not_failed_batch(cases, monkeypatch):
    calls = []

    async def synthesize(estimator, provider, context):
        calls.append(set(provider.current_trajectory_uris))
        if len(calls) == 2:
            raise ValueError("second batch failed")
        return None

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    first = await asyncio.wait_for(learn(cases), timeout=3)
    assert first["trajectory_count"] == 4 and first["failed_count"] == 1
    assert [s["state"] for s in states(cases)] == ["no_op", "no_op", "failed", "failed", "pending"]
    second = await asyncio.wait_for(learn(cases), timeout=3)
    assert second["trajectory_count"] == 1 and second["no_op_count"] == 1
    assert calls[-1] == {cases.records[-1].uri}
    assert [s["attempts"] for s in states(cases)] == [1] * 5
    assert [s["state"] for s in states(cases)] == ["no_op", "no_op", "failed", "failed", "no_op"]
    # Retry failed work only when explicitly requested; do not relearn completed work.
    third = await asyncio.wait_for(learn(cases, retry_failed=True), timeout=3)
    assert third["trajectory_count"] == 2 and third["no_op_count"] == 1
    assert calls[-1] == {r.uri for r in cases.records[2:4]}
    assert [s["attempts"] for s in states(cases)] == [1, 1, 2, 2, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 5)], indirect=True)
async def test_cancelled_batch_does_not_claim_the_tail(cases, monkeypatch):
    ready = asyncio.Event()
    finish = asyncio.Event()
    calls = 0

    async def synthesize(estimator, provider, context):
        nonlocal calls
        calls += 1
        if calls == 2:
            ready.set()
            await finish.wait()
        return None

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    task = asyncio.create_task(learn(cases))
    try:
        await asyncio.wait_for(ready.wait(), timeout=3)
        assert [s["state"] for s in states(cases)] == [
            "no_op",
            "no_op",
            "processing",
            "processing",
            "pending",
        ]
    finally:
        task.cancel()
        # Existing trainer.close() drains an active batch rather than abandoning
        # an in-flight write. Release it and verify cancellation never claims more.
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
    assert [s["state"] for s in states(cases)] == ["no_op"] * 4 + ["pending"]
    assert states(cases)[-1]["attempts"] == 0
    assert not cases.compressor._experience_learning_lock.locked()
    assert not cases.fs.policy_lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 5)], indirect=True)
async def test_archive_failure_does_not_claim_tail_or_relearn_terminal_batch(cases, monkeypatch):
    synthesize = AsyncMock(return_value=None)
    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    original_cp = cases.fs.cp
    failed_source = cases.records[2].uri

    async def copy(old_uri, new_uri, **kwargs):
        if old_uri == failed_source:
            raise OSError("archive unavailable")
        return await original_cp(old_uri, new_uri, **kwargs)

    cases.fs.cp = copy
    first = await asyncio.wait_for(learn(cases), timeout=3)
    assert first["failed_count"] == 1 and first["trajectory_count"] == 4
    assert [s["state"] for s in states(cases)] == ["no_op"] * 4 + ["pending"]
    assert [s["attempts"] for s in states(cases)] == [1] * 4 + [0]
    assert synthesize.await_count == 2
    cases.fs.cp = original_cp
    second = await asyncio.wait_for(learn(cases), timeout=3)
    assert second["no_op_count"] == 1 and second["trajectory_count"] == 1
    assert synthesize.await_count == 3
    assert [s["state"] for s in states(cases)] == ["no_op"] * 5
    assert [s["attempts"] for s in states(cases)] == [1] * 5


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 5)], indirect=True)
async def test_partial_claim_failure_does_not_touch_unclaimed_records(cases, monkeypatch):
    synthesize = AsyncMock(return_value=None)
    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    original = cases.compressor._write_trajectory_update_state
    blocked_uri = cases.records[3].uri

    async def write_state(**kwargs):
        if kwargs["state"] == "processing" and kwargs["memory"].uri == blocked_uri:
            raise OSError("claim unavailable")
        return await original(**kwargs)

    cases.compressor._write_trajectory_update_state = write_state
    result = await asyncio.wait_for(learn(cases), timeout=3)
    assert result["failed_count"] == 1 and result["trajectory_count"] == 3
    assert synthesize.await_count == 1
    assert [s["state"] for s in states(cases)] == ["no_op", "no_op", "failed", "pending", "pending"]
    assert [s["attempts"] for s in states(cases)] == [1, 1, 1, 0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(2, 5)], indirect=True)
async def test_failure_stops_only_its_case_and_short_tail_flushes_immediately(cases, monkeypatch):
    calls = Counter()
    sizes = {"case_0": [], "case_1": []}

    async def synthesize(estimator, provider, context):
        name = provider.trajectories[0].metadata["case_name"]
        calls[name] += 1
        sizes[name].append(len(provider.current_trajectory_uris))
        if name == "case_0" and calls[name] == 2:
            raise ValueError("case zero second batch failed")
        return None

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    result = await asyncio.wait_for(learn(cases), timeout=3)
    assert result["failed_count"] == result["no_op_count"] == 1
    assert sizes == {"case_0": [2, 2], "case_1": [2, 2, 1]}
    assert [s["state"] for s in states(cases)] == [
        "no_op",
        "no_op",
        "failed",
        "failed",
        "pending",
    ] + ["no_op"] * 5
    successful = next(r for r in result["results"] if r["case_name"] == "case_1")
    assert successful["plan_metadata"]["chunk_count"] == 3
    assert successful["plan_metadata"]["chunk_item_counts"] == [0, 0, 0]
    assert all(
        m["optimizer"] == "trajectory_comparison" for m in successful["plan_metadata"]["chunks"]
    )
    assert cases.compressor.streaming_trainer_config.max_gradients_per_update == 2
    assert cases.compressor.streaming_trainer_config.max_wait_seconds == 3600
