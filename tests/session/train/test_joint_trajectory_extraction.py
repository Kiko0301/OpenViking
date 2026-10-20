# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import asyncio
import copy
import json
from unittest.mock import AsyncMock

import pytest
from test_case_experience_pipeline import setup_observation as setup_observation
from test_case_experience_pipeline import trajectory_program
from test_gradient_estimator_component import analysis

from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.train.components.session_analyzer import SessionAnalyzerContext


def rollouts(*names, source="viking://user/u/sessions/joint/history/archive_001"):
    first = analysis(name=names[0] if names else "case_a", session=source)
    first.rollout.metadata.update(first.metadata)
    first.rollout.evaluation = first.evaluation
    result = []
    for name in names:
        rollout = copy.deepcopy(first.rollout)
        rollout.case.name = name
        result.append(rollout)
    return result


def program(*names):
    return (
        "\n".join(
            trajectory_program(f"1. `user` Fact for {name}", case_name=name).removesuffix(
                "\nsdk.commit()"
            )
            for name in names
        )
        + "\nsdk.commit()"
    )


def context(setup):
    return SessionAnalyzerContext(request_context=setup[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
async def test_one_joint_request_creates_exactly_one_correctly_bound_trajectory_per_case(
    setup_observation, monkeypatch, output_format
):
    from openviking_cli.utils.config import get_openviking_config

    monkeypatch.setattr(get_openviking_config().memory, "extraction_output_format", output_format)
    fs, _, model, analyzer, _, _ = setup_observation
    inputs = rollouts("case_a", "case_b")
    response = program("case_b", "case_a")  # Binding is by Case name, not output order.
    if output_format == "json":
        response = json.dumps(
            {
                "trajectories": [
                    {
                        "page_id": 100 + i,
                        "case_name": name,
                        "context_rules": "",
                        "evaluation_criteria": "[]",
                        "action_scores": "[]",
                        "steps": f"1. `user` Fact for {name}",
                    }
                    for i, name in enumerate(["case_b", "case_a"])
                ]
            }
        )
    model.get_completion_async.return_value = response
    fs.ls = AsyncMock(side_effect=AssertionError("Creation must not scan trajectory history"))
    results = await analyzer.analyze_many(inputs, context(setup_observation))
    model.get_completion_async.assert_awaited_once()
    request = model.get_completion_async.await_args.kwargs["messages"]
    payload = next(
        json.loads(m["content"]) for m in request if m["content"].startswith('{"cases":')
    )
    assert [item["case"]["name"] for item in payload["cases"]] == ["case_a", "case_b"]
    assert "source_steps" in payload
    assert all("source_steps" not in item for item in payload["cases"])
    assert "another Case's exclusive actions" in request[0]["content"]
    assert "Shared facts may" in request[0]["content"]
    assert len(results) == 2
    for result, name in zip(results, ["case_a", "case_b"], strict=True):
        (trajectory,) = result.trajectories
        assert f"/trajectories/{name}/" in trajectory.uri
        assert f"Fact for {name}" in trajectory.content
        stored = MemoryFileUtils.read(fs.files[trajectory.uri], uri=trajectory.uri)
        assert stored.extra_fields["case_name"] == name
        assert stored.links[0]["to_uri"] == inputs[0].metadata["source_session_uri"]
    fs.ls.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("names", "error"),
    [
        (("case_a",), "missing_cases"),
        (("case_a", "case_a"), "duplicate_cases"),
        (("case_a", "other"), "unexpected_cases"),
        (("case_a", "case_b", "other"), "unexpected_cases"),
    ],
)
@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_case_set_validation_retries_before_any_write(
    setup_observation, names, error, repair_succeeds
):
    fs, _, model, analyzer, _, _ = setup_observation
    answers = iter(
        [program(*names), program("case_a", "case_b") if repair_succeeds else program(*names)]
    )

    async def complete(**kwargs):
        assert not any("/trajectories/" in uri for uri in fs.files)
        return next(answers)

    model.get_completion_async.side_effect = complete
    if repair_succeeds:
        results = await analyzer.analyze_many(
            rollouts("case_a", "case_b"), context(setup_observation)
        )
        assert len(results) == 2
    else:
        with pytest.raises(ValueError, match="Provider operation validation failed after repair"):
            await analyzer.analyze_many(rollouts("case_a", "case_b"), context(setup_observation))
        assert not fs.files
    assert model.get_completion_async.await_count == 2
    assert any(
        error in m["content"] for m in model.get_completion_async.await_args.kwargs["messages"]
    )


@pytest.mark.asyncio
async def test_no_cases_make_no_model_or_storage_call(setup_observation):
    fs, _, model, analyzer, _, _ = setup_observation
    assert await analyzer.analyze_many([], context(setup_observation)) == []
    model.get_completion_async.assert_not_awaited()
    assert not fs.files


@pytest.mark.asyncio
async def test_joint_input_rejects_mixed_archives_or_duplicate_cases(setup_observation):
    _, _, model, analyzer, _, _ = setup_observation
    inputs = rollouts("case_a", "case_b")
    inputs[1].metadata["source_session_uri"] += "_other"
    with pytest.raises(ValueError, match="same archive"):
        await analyzer.analyze_many(inputs, context(setup_observation))
    with pytest.raises(ValueError, match="unique Case"):
        await analyzer.analyze_many(rollouts("case_a", "case_a"), context(setup_observation))
    model.get_completion_async.assert_not_awaited()


def keyed_locks(fs):
    locks = {}
    acquired = []

    async def acquire(paths, **kwargs):
        acquired.append(tuple(paths))
        held = []
        try:
            for path in paths:
                lock = locks.setdefault(path, asyncio.Lock())
                await lock.acquire()
                held.append(lock)
            return held
        except BaseException:
            for lock in reversed(held):
                lock.release()
            raise

    async def release(held):
        for lock in reversed(held):
            lock.release()

    fs._async_agfs.pathlock_acquire_exact_batch = acquire
    fs._async_agfs.pathlock_release = release
    fs._async_agfs.pathlock_acquire_tree = AsyncMock(
        side_effect=AssertionError("No Case tree lock")
    )
    return locks, acquired


@pytest.mark.asyncio
async def test_slow_writer_does_not_block_another_source_in_same_case(setup_observation):
    fs, _, _, analyzer, _, _ = setup_observation
    locks, acquired = keyed_locks(fs)
    started, release = asyncio.Event(), asyncio.Event()
    write = fs.write_file
    first = True

    async def delayed_write(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            started.set()
            await release.wait()
        return await write(*args, **kwargs)

    fs.write_file = delayed_write
    task = asyncio.create_task(
        analyzer.analyze_many(
            rollouts("case_a", source="viking://session/a/history/1"), context(setup_observation)
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 3)
        other = await asyncio.wait_for(
            analyzer.analyze_many(
                rollouts("case_a", source="viking://session/b/history/1"),
                context(setup_observation),
            ),
            3,
        )
        assert len(other[0].trajectories) == 1
    finally:
        release.set()
        await task
    assert all(not lock.locked() for lock in locks.values())
    assert all(len(paths) == 1 and not paths[0].endswith(".overview.md") for paths in acquired)


@pytest.mark.asyncio
async def test_cancelled_writer_releases_file_lease(setup_observation):
    fs, _, _, analyzer, _, _ = setup_observation
    locks, _ = keyed_locks(fs)
    started = asyncio.Event()

    async def wait_forever(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    fs.write_file = wait_forever
    task = asyncio.create_task(
        analyzer.analyze_many(rollouts("case_a"), context(setup_observation))
    )
    await asyncio.wait_for(started.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(not lock.locked() for lock in locks.values())


@pytest.mark.asyncio
async def test_collision_at_file_lock_changes_name_without_overwriting(
    setup_observation, monkeypatch
):
    fs, _, model, analyzer, _, _ = setup_observation
    keyed_locks(fs)
    original_acquire = fs._async_agfs.pathlock_acquire_exact_batch
    collision_uri = None

    async def acquire(paths, **kwargs):
        nonlocal collision_uri
        lease = await original_acquire(paths, **kwargs)
        if collision_uri is None:
            # A competing creator won after extraction validation, before our file lease.
            collision_uri = paths[0]
            fs.files[collision_uri] = "other creator's immutable data"
        return lease

    # The fake maps URI to path unchanged, making the injected collision observable by exists().
    fs._uri_to_path = lambda uri, **kwargs: uri
    fs._async_agfs.pathlock_acquire_exact_batch = acquire
    results = await analyzer.analyze_many(rollouts("case_a"), context(setup_observation))
    assert results[0].trajectories[0].uri != collision_uri
    assert fs.files[collision_uri] == "other creator's immutable data"
    model.get_completion_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_200_new_sources_do_not_read_or_list_existing_history(setup_observation):
    fs, _, model, analyzer, _, _ = setup_observation
    locks, _ = keyed_locks(fs)
    history = {
        f"viking://user/u/memories/trajectories/case_a/old/{i}.md": "old" for i in range(1000)
    }
    fs.files.update(history)
    read = fs.read_file

    async def checked_read(uri, **kwargs):
        assert uri not in history, "Creation must not read historical trajectories"
        return await read(uri, **kwargs)

    fs.read_file = checked_read
    fs.ls = AsyncMock(side_effect=AssertionError("Creation must not list historical trajectories"))
    results = await asyncio.gather(
        *[
            analyzer.analyze_many(
                rollouts("case_a", source=f"viking://session/s{i}/history/1"),
                context(setup_observation),
            )
            for i in range(200)
        ]
    )
    assert len({items[0].trajectories[0].uri for items in results}) == 200
    assert model.get_completion_async.await_count == 200
    assert all(not lock.locked() for lock in locks.values())
    fs.ls.assert_not_awaited()


@pytest.mark.asyncio
async def test_name_claimed_during_model_call_is_reallocated_before_create_guard(setup_observation):
    fs, _, model, analyzer, _, _ = setup_observation
    candidates = []
    exists = fs.exists
    collision_uri = None

    async def remember_candidate(uri, **kwargs):
        candidates.append(uri)
        return await exists(uri, **kwargs)

    async def complete(**kwargs):
        nonlocal collision_uri
        collision_uri = candidates[0]
        fs.files[collision_uri] = "other creator's immutable data"
        return trajectory_program()

    fs.exists = remember_candidate
    model.get_completion_async.side_effect = complete
    results = await analyzer.analyze_many(rollouts("case_a"), context(setup_observation))
    model.get_completion_async.assert_awaited_once()
    assert results[0].trajectories[0].uri != collision_uri
    assert fs.files[collision_uri] == "other creator's immutable data"


@pytest.mark.asyncio
@pytest.mark.parametrize("case_count", [0, 1, 2])
async def test_compressor_uses_one_joint_extraction_not_a_per_case_model_loop(
    setup_observation, monkeypatch, case_count
):
    from openviking.session.compressor_v3 import SessionCompressorV3

    fs, ctx, model, analyzer, _, _ = setup_observation
    inputs = rollouts(*[f"case_{i}" for i in range(max(1, case_count))])
    monkeypatch.setattr("openviking.session.compressor_v3.get_viking_fs", lambda: fs)
    monkeypatch.setattr(
        "openviking.session.memory.account_templates.resolve_account_memory_registry",
        AsyncMock(side_effect=lambda fs, account_id, registry: registry),
    )
    model.get_completion_async.return_value = program(*[r.case.name for r in inputs])
    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=analyzer)
    compressor._session_skill_extraction_enabled = lambda: False
    analyze_many = AsyncMock(wraps=analyzer.analyze_many)
    monkeypatch.setattr(type(analyzer), "analyze_many", analyze_many)
    result = await compressor.train_from_extracted_cases(
        cases=[r.case for r in inputs] if case_count else [],
        messages=inputs[0].messages,
        ctx=ctx,
        archive_uri=inputs[0].metadata["source_session_uri"],
        collect_memory_diff=True,
    )
    assert result["case_count"] == case_count
    assert result["submitted"] == case_count
    assert len(result.get("trajectory_uris", [])) == case_count
    assert model.get_completion_async.await_count == (1 if case_count else 0)
    assert analyze_many.await_count == (1 if case_count else 0)
    if case_count:
        assert len(analyze_many.await_args.args[0]) == case_count


@pytest.mark.asyncio
async def test_joint_validation_uses_each_cases_own_rubric():
    from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
    from openviking.session.memory.experience.trajectory_context import (
        AgentTrajectoryContextProvider,
        TrajectoryTarget,
    )

    targets = [
        TrajectoryTarget(
            case={
                "name": name,
                "rubric": {"criteria": [{"name": "check", "description": name + " requirement"}]},
            },
            target_uri=f"viking://user/u/memories/trajectories/{name}/new.md",
        )
        for name in ["case_a", "case_b"]
    ]
    provider = AgentTrajectoryContextProvider(messages=[], source_steps=[], targets=targets)
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="trajectories",
                uris=[],
                memory_fields={
                    "case_name": target.case_name,
                    "context_rules": "",
                    "steps": "Observed fact.",
                    "evaluation_criteria": [
                        {"id": "check", "passed": True, "evidence": "Observed"}
                    ],
                },
            )
            for target in targets
        ],
        delete_file_contents=[],
        errors=[],
    )
    assert await provider.validate_resolved_operations(operations) == []
    for operation, target in zip(operations.upsert_operations, targets, strict=True):
        assert (
            operation.memory_fields["evaluation_criteria"][0]["description"]
            == target.case_name + " requirement"
        )
        assert operation.uris == [target.target_uri] and operation.create_only


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
@pytest.mark.parametrize("repair_succeeds", [True, False])
@pytest.mark.parametrize("bad_result", ["missing", "null", "true", "score"])
async def test_external_judgment_mismatch_uses_existing_retry_before_writing(
    setup_observation, monkeypatch, output_format, repair_succeeds, bad_result
):
    from openviking.session.memory.experience.experience_context import _trajectory_model_content
    from openviking.session.train.domain import CriterionResult, RubricEvaluation
    from openviking_cli.utils.config import get_openviking_config

    monkeypatch.setattr(get_openviking_config().memory, "extraction_output_format", output_format)
    fs, _, model, analyzer, _, _ = setup_observation
    inputs = rollouts("case_a")
    inputs[0].evaluation = RubricEvaluation(
        passed=False,
        score=0.0,
        criterion_results=[
            CriterionResult("external_reward", False, 0.0, [], ["Database mismatch."])
        ],
        feedback=[],
    )
    expected = {
        "id": "external_reward",
        "passed": False,
        "score": 0.0,
        "evidence": "External database check failed despite the tool succeeding.",
    }
    invalid = dict(expected)
    if bad_result in {"null", "true"}:
        invalid["passed"] = None if bad_result == "null" else True
    elif bad_result == "score":
        invalid["score"] = 1.0

    def response(assessments):
        fields = {
            "case_name": "case_a",
            "context_rules": "",
            "steps": "1. `tool:update` The tool returned success.",
            "evaluation_criteria": json.dumps(assessments),
            "action_scores": "[]",
        }
        if output_format == "json":
            return json.dumps({"trajectories": [{"page_id": 100, **fields}]})
        arguments = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        return f"sdk.create_trajectories({arguments})\nsdk.commit()"

    bad = response([] if bad_result == "missing" else [invalid])
    answers = iter([bad, response([expected]) if repair_succeeds else bad])

    async def complete(**kwargs):
        assert not fs.files, "Invalid external judgments must not be persisted before repair"
        return next(answers)

    model.get_completion_async.side_effect = complete
    if repair_succeeds:
        (result,) = await analyzer.analyze_many(inputs, context(setup_observation))
        (trajectory,) = result.trajectories
        stored = MemoryFileUtils.read(fs.files[trajectory.uri], uri=trajectory.uri)
        assert stored.extra_fields["evaluation_criteria"] == [expected]
        assert "passed: false" in _trajectory_model_content(stored.content)
        assert "score: 0.0" in _trajectory_model_content(stored.content)
    else:
        with pytest.raises(ValueError, match="Provider operation validation failed after repair"):
            await analyzer.analyze_many(inputs, context(setup_observation))
        assert not fs.files
    assert model.get_completion_async.await_count == 2
    repair_messages = model.get_completion_async.await_args.kwargs["messages"]
    assert any(
        '"expected": {"id": "external_reward", "passed": false, "score": 0.0}' in m["content"]
        for m in repair_messages
        if m["role"] == "user"
    )
