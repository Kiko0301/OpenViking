# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_case_experience_pipeline import setup_observation as setup_observation
from test_external_case_learning import cases as cases
from test_external_case_learning import states
from test_two_phase_learning import _policy_set, result

from openviking.session.memory.memory_type_registry import get_default_registry
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.memory.utils.uri import generate_uri
from openviking.session.train import ExperienceGradientEstimator
from openviking.session.train.components.experience_learner import RemoteExperienceLearner
from openviking.session.train.components.trajectory_lifecycle import (
    archive_trajectory,
    trajectory_archive_uri,
)


async def candidates(cases, *, uris=None, names=None, retry_failed=False):
    return await cases.compressor._load_pending_trajectory_memories(
        viking_fs=cases.fs,
        ctx=cases.ctx,
        registry=get_default_registry(),
        trajectory_uris=uris,
        case_names=names,
        retry_failed=retry_failed,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(2, 3)], indirect=True)
async def test_new_trigger_includes_older_pending_but_not_unrelated_case(cases):
    newest = cases.records[2]
    cases.fs.ls = AsyncMock(wraps=cases.fs.ls)
    selected = await candidates(cases, uris=[newest.uri, newest.uri])
    assert [m.uri for m in selected] == [m.uri for m in cases.records[:3]]
    assert [c.args[0] for c in cases.fs.ls.await_args_list] == [newest.uri.rsplit("/", 1)[0]]
    assert all(s["state"] == "pending" and s["attempts"] == 0 for s in states(cases))


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(2, 3)], indirect=True)
async def test_remote_experiment_trigger_consumes_backlog_without_resubmitting_sessions(
    cases, monkeypatch
):
    seen = []

    async def synthesize(estimator, provider, context):
        seen.append(set(provider.current_trajectory_uris))
        return None

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    outcome = None

    async def submit(**kwargs):
        nonlocal outcome
        assert kwargs.pop("wait") is False
        outcome = await cases.compressor.learn_pending_experiences(ctx=cases.ctx, **kwargs)
        return {"task_id": "inbox-job"}

    async def task(_task_id):
        return {"status": "completed", "result": outcome}

    client = SimpleNamespace(
        learn_experiences=AsyncMock(side_effect=submit), get_task=AsyncMock(side_effect=task)
    )
    batch = result()
    batch.apply_result.metadata["commit_results"] = [
        {"task_status": "completed", "trajectory_uris": [cases.records[2].uri]}
    ]
    learned = await asyncio.wait_for(
        RemoteExperienceLearner(client=client).learn([batch], _policy_set()), timeout=3
    )
    learning = learned.apply_result.metadata["experience_learning"]
    assert learning["trigger_trajectory_count"] == 1
    assert learning["trajectory_count"] == 3
    assert learning["selection_mode"] == "case_inbox"
    assert seen == [{m.uri for m in cases.records[:2]}, {cases.records[2].uri}]
    assert [s["state"] for s in states(cases)] == ["no_op"] * 3 + ["pending"] * 3
    assert [s["attempts"] for s in states(cases)] == [1] * 3 + [0] * 3
    client.learn_experiences.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 6)], indirect=True)
async def test_state_filters_and_terminal_archival_recovery_are_preserved(cases):
    for index, updates in {
        1: {"state": "failed", "attempts": 1},
        2: {"state": "processing", "attempts": 1, "lease_until": "2099-01-01T00:00:00+00:00"},
        3: {"state": "processing", "attempts": 1, "lease_until": "2000-01-01T00:00:00+00:00"},
        4: {"state": "no_op", "attempts": 1},
    }.items():
        m = cases.records[index]
        m.extra_fields["experience_update"].update(updates)
        cases.fs.files[m.uri] = MemoryFileUtils.write(m)
    selected = await candidates(cases, uris=[cases.records[5].uri])
    assert [m.uri for m in selected] == [cases.records[i].uri for i in [0, 3, 5]]
    assert cases.records[4].uri not in cases.fs.files
    assert trajectory_archive_uri(cases.records[4].uri) in cases.fs.files
    selected = await candidates(cases, uris=[cases.records[5].uri], retry_failed=True)
    assert [m.uri for m in selected] == [cases.records[i].uri for i in [0, 1, 3, 5]]
    assert states(cases)[4]["attempts"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 3)], indirect=True)
async def test_archived_notification_wakes_inbox_without_scanning_archive(cases):
    old = cases.records[0]
    old.extra_fields["experience_update"]["state"] = "no_op"
    cases.fs.files[old.uri] = MemoryFileUtils.write(old)
    archived = await archive_trajectory(
        old, viking_fs=cases.fs, ctx=cases.ctx, registry=get_default_registry(), vikingdb=None
    )
    archive_dir = archived.uri.rsplit("/", 1)[0]
    cases.fs.files[archive_dir + "/S_unrelated.md"] = "Must not read archival history"
    original_ls, original_read = cases.fs.ls, cases.fs.read_file

    async def listing(uri, **kwargs):
        assert uri == old.uri.rsplit("/", 1)[0]
        return await original_ls(uri, **kwargs)

    async def read(uri, **kwargs):
        if "/archive/" in uri:
            assert uri == archived.uri
        return await original_read(uri, **kwargs)

    cases.fs.ls, cases.fs.read_file = listing, read
    assert [m.uri for m in await candidates(cases, uris=[old.uri])] == [
        r.uri for r in cases.records[1:]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(3, 2)], indirect=True)
async def test_named_case_and_trigger_scopes_form_a_union(cases):
    selected = await candidates(cases, uris=[cases.records[0].uri], names=["case_1"])
    assert [m.uri for m in selected] == [r.uri for r in cases.records[:4]]


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 1)], indirect=True)
async def test_empty_notifications_do_not_expand_into_unfiltered_consumption(cases):
    cases.fs.ls = AsyncMock(side_effect=AssertionError("No global scan for an empty notification"))
    assert await candidates(cases, uris=[]) == []
    cases.fs.ls.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(2, 1)], indirect=True)
async def test_scope_is_resolved_from_logical_case_name_not_portable_directory_name(cases):
    schema = get_default_registry().get("trajectories")
    logical_name = "case:quoted"
    directory = generate_uri(
        schema.model_copy(update={"filename_template": ""}),
        {"case_name": logical_name},
        user_space=cases.ctx.user.user_id,
    )
    m = cases.records[0].model_copy(deep=True)
    old_uri = m.uri
    m.uri = directory + "/inbox/" + old_uri.rsplit("/", 1)[1]
    m.extra_fields["case_name"] = logical_name
    m.extra_fields["archive_uri"] = trajectory_archive_uri(m.uri)
    assert directory.rsplit("/", 1)[-1] != logical_name
    cases.fs.files[m.uri] = MemoryFileUtils.write(m)
    del cases.fs.files[old_uri]
    selected = await candidates(cases, uris=[m.uri])
    assert [r.uri for r in selected] == [m.uri]
    assert selected[0].extra_fields["case_name"] == logical_name


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(2, 1)], indirect=True)
@pytest.mark.parametrize("failure", ["other_user", "missing", "wrong_case", "permission"])
async def test_bad_trigger_never_widens_scope_or_swallows_errors(cases, failure):
    uri = cases.records[0].uri
    cases.fs.ls = AsyncMock(side_effect=AssertionError("Invalid trigger must not scan any inbox"))
    before = dict(cases.fs.files)
    if failure == "other_user":
        uri = uri.replace("/user/u/", "/user/someone_else/")
        error = ValueError
    elif failure == "missing":
        del cases.fs.files[uri]
        error = FileNotFoundError
    elif failure == "wrong_case":
        m = cases.records[0].model_copy(deep=True)
        m.extra_fields["case_name"] = "case_1"
        cases.fs.files[uri] = MemoryFileUtils.write(m)
        error = ValueError
    else:
        cases.fs.read_file = AsyncMock(side_effect=PermissionError("denied"))
        error = PermissionError
    with pytest.raises(error):
        await candidates(cases, uris=[uri])
    cases.fs.ls.assert_not_awaited()
    assert cases.fs.files[cases.records[1].uri] == before[cases.records[1].uri]


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 5)], indirect=True)
async def test_next_round_new_uri_consumes_previous_unclaimed_tail(
    cases, setup_observation, monkeypatch
):
    from test_case_experience_pipeline import observe, trajectory_program
    from test_gradient_estimator_component import analysis

    batches = []

    async def synthesize(estimator, provider, context):
        batches.append(set(provider.current_trajectory_uris))
        if len(batches) == 2:
            raise ValueError("current batch failed")
        return None

    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    first = await asyncio.wait_for(
        cases.compressor.learn_pending_experiences(
            ctx=cases.ctx, trajectory_uris=[cases.records[0].uri]
        ),
        timeout=3,
    )
    assert first["failed_count"] == 1 and first["trajectory_count"] == 4
    assert [s["state"] for s in states(cases)] == ["no_op", "no_op", "failed", "failed", "pending"]

    fs, ctx, model, analyzer, context, policies = setup_observation
    model.get_completion_async.return_value = trajectory_program(case_name="case_0")
    fresh = analysis(name="case_0", session="viking://session/next_round/history/archive_001")
    await observe(analyzer, fresh, policies, context)
    new_memory = fresh.metadata["trajectory_memory"]
    cases.records.append(new_memory)
    second = await asyncio.wait_for(
        cases.compressor.learn_pending_experiences(ctx=cases.ctx, trajectory_uris=[new_memory.uri]),
        timeout=3,
    )
    assert second["no_op_count"] == 1 and second["trajectory_count"] == 2
    assert batches[-1] == {cases.records[4].uri, new_memory.uri}
    assert [s["state"] for s in states(cases)] == [
        "no_op",
        "no_op",
        "failed",
        "failed",
        "no_op",
        "no_op",
    ]
    assert [s["attempts"] for s in states(cases)] == [1] * 6


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 3)], indirect=True)
async def test_creation_order_is_not_notification_or_filename_order(cases):
    dates = ["2026-10-20T01:00:00+00:00", "2026-10-20T10:00:00+08:00", "2026-10-20T00:30:00Z"]
    for m, date in zip(cases.records, dates, strict=True):
        m.extra_fields["created_at"] = date
        cases.fs.files[m.uri] = MemoryFileUtils.write(m)
    selected = await candidates(cases, uris=[cases.records[1].uri, cases.records[0].uri])
    assert [m.uri for m in selected] == [cases.records[i].uri for i in [2, 0, 1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 2)], indirect=True)
@pytest.mark.parametrize("created_at", [None, "invalid timestamp"])
async def test_invalid_creation_metadata_is_not_silently_ordered_or_claimed(cases, created_at):
    m = cases.records[0]
    m.extra_fields["created_at"] = created_at
    cases.fs.files[m.uri] = MemoryFileUtils.write(m)
    with pytest.raises(ValueError, match="Invalid Trajectory created_at"):
        await candidates(cases, uris=[cases.records[1].uri])
    assert all(s["state"] == "pending" and s["attempts"] == 0 for s in states(cases))
