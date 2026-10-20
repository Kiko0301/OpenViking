from __future__ import annotations

import asyncio
import copy
import re
from unittest.mock import AsyncMock

import pytest
from test_case_experience_pipeline import observe
from test_case_experience_pipeline import setup_observation as setup_observation
from test_gradient_estimator_component import analysis

from openviking.message import Message, TextPart
from openviking.session.memory.memory_type_registry import get_default_registry
from openviking.session.memory.memory_updater import ExtractContext
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.memory.utils.uri import generate_uri
from openviking.session.train.components.trajectory_observation import (
    list_trajectory_uris,
    trajectory_source_session_uri,
)


def context():
    return ExtractContext(
        [
            Message(
                id="1",
                role="user",
                parts=[TextPart(text="request")],
                created_at="2024-05-15T15:00:00-05:00",
            )
        ]
    )


@pytest.mark.parametrize("outcome,code", [("success", "S"), ("failure", "F"), ("unknown", "U")])
def test_schema_uses_dated_flat_inbox_and_stable_short_random_id(outcome, code):
    ctx = context()
    schema = get_default_registry().get("trajectories")
    fields = {"case_name": "case_a", "outcome": outcome}
    first = generate_uri(schema, fields, user_space="u", extract_context=ctx)
    assert re.fullmatch(
        rf"viking://user/u/memories/trajectories/case_a/inbox/20240515_{code}_[a-z0-9]{{6}}\.md",
        first,
    )
    assert all(
        generate_uri(schema, fields, user_space="u", extract_context=ctx) == first for _ in range(3)
    )
    assert first.endswith("_" + ctx.get_random_id() + ".md")
    ctx.messages[0].created_at = "2030-01-02T00:00:00Z"
    assert generate_uri(schema, fields, user_space="u", extract_context=ctx) == first


@pytest.mark.asyncio
async def test_concurrent_and_repeated_same_source_create_new_trajectories(setup_observation):
    fs, _, model, analyzer, ctx, policies = setup_observation
    first = analysis(session="viking://session/same/archives/1", passed=True)
    second = copy.deepcopy(first)
    await asyncio.gather(
        observe(analyzer, first, policies, ctx), observe(analyzer, second, policies, ctx)
    )
    assert first.trajectories[0].uri != second.trajectories[0].uri
    uri = first.trajectories[0].uri
    assert len([p for p in fs.files if "/trajectories/" in p and p.endswith(".md")]) == 2
    saved = MemoryFileUtils.read(fs.files[uri], uri=uri)
    assert trajectory_source_session_uri(saved) == first.metadata["source_session_uri"]
    assert re.search(r"/\d{8}_S_[a-z0-9]{6}\.md$", uri)
    previous_uri = second.trajectories[0].uri
    await observe(analyzer, second, policies, ctx)
    assert second.trajectories[0].uri not in {uri, previous_uri}
    assert model.get_completion_async.await_count == 3


@pytest.mark.asyncio
async def test_short_name_collision_does_not_overwrite_another_source(
    setup_observation, monkeypatch
):
    fs, _, _, analyzer, ctx, policies = setup_observation

    def short_id(self, *, refresh=False):
        if self._random_id is None or refresh:
            self._random_id = "bbbbbb" if refresh else "aaaaaa"
        return self._random_id

    monkeypatch.setattr(ExtractContext, "get_random_id", short_id)
    first = analysis(session="viking://session/first/archives/1", passed=True)
    second = analysis(session="viking://session/second/archives/1", passed=True)
    await observe(analyzer, first, policies, ctx)
    first_uri = first.trajectories[0].uri
    before = fs.files[first_uri]
    await observe(analyzer, second, policies, ctx)
    assert first_uri.endswith("_S_aaaaaa.md")
    assert second.trajectories[0].uri.endswith("_S_bbbbbb.md")
    assert fs.files[first_uri] == before
    assert (
        trajectory_source_session_uri(second.metadata["trajectory_memory"])
        == second.metadata["source_session_uri"]
    )


@pytest.mark.asyncio
async def test_pending_discovery_only_lists_flat_inbox(setup_observation):
    fs, req_ctx, _, analyzer, ctx, policies = setup_observation
    for day in ("2024-05-15", "2024-06-16"):
        run = analysis(session=f"viking://session/{day}/archives/1", passed=True)
        for msg in run.rollout.messages:
            msg.created_at = day + "T15:00:00-05:00"
        await observe(analyzer, run, policies, ctx)
        memory = run.metadata["trajectory_memory"]
        memory.extra_fields["experience_update"]["state"] = "applied"
        fs.files[memory.uri] = MemoryFileUtils.write(memory)
    history = await list_trajectory_uris(
        fs, req_ctx, "viking://user/u/memories/trajectories/case_a/inbox"
    )
    assert len(history) == 2
    assert all(uri.rsplit("/", 1)[0].endswith("/inbox") for uri in history)


@pytest.mark.asyncio
async def test_directory_scan_pages_without_losing_trajectories():
    root = "viking://user/u/memories/trajectories/case_a"
    entries = [{"uri": f"{root}/{i}.md", "isDir": False} for i in range(1001)]
    fs = type("FS", (), {})()
    fs.ls = AsyncMock(
        side_effect=lambda uri, **kw: entries[kw["offset"] : kw["offset"] + kw["node_limit"]]
    )
    assert len(await list_trajectory_uris(fs, None, root)) == 1001
    assert [call.kwargs["offset"] for call in fs.ls.await_args_list] == [0, 1000]


@pytest.mark.asyncio
async def test_repeated_source_gets_new_name_and_keeps_original_file(setup_observation):
    fs, _, model, analyzer, ctx, policies = setup_observation
    first = analysis(session="viking://session/retry/archives/1", passed=False)
    first.rollout.messages[0].created_at = "2024-05-15T00:00:00Z"
    await observe(analyzer, first, policies, ctx)
    uri = first.trajectories[0].uri
    before = fs.files[uri]
    first.rollout.messages[0].created_at = "2025-06-16T00:00:00Z"
    await observe(analyzer, first, policies, ctx)
    assert first.trajectories[0].uri != uri
    assert first.trajectories[0].uri.rsplit("/", 1)[0].endswith("/inbox")
    assert first.trajectories[0].uri.rsplit("/", 1)[-1][:8] == uri.rsplit("/", 1)[-1][:8]
    assert fs.files[uri] == before
    assert model.get_completion_async.await_count == 2


@pytest.mark.asyncio
async def test_collision_exhaustion_is_safe(setup_observation, monkeypatch):
    fs, _, _, analyzer, ctx, policies = setup_observation
    monkeypatch.setattr(ExtractContext, "get_random_id", lambda self, **kwargs: "aaaaaa")
    first = analysis(session="viking://session/first/archives/1", passed=True)
    second = analysis(session="viking://session/second/archives/1", passed=True)
    await observe(analyzer, first, policies, ctx)
    before = dict(fs.files)
    with pytest.raises(RuntimeError, match="unused Trajectory filename"):
        await observe(analyzer, second, policies, ctx)
    assert fs.files == before
    assert not fs.policy_lock.locked()
