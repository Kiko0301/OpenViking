# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_case_experience_pipeline import observe
from test_case_experience_pipeline import setup_observation as setup_observation
from test_external_case_learning import cases as cases
from test_external_case_learning import learn, states
from test_gradient_estimator_component import analysis

from openviking.session.compressor_v3 import SessionCompressorV3
from openviking.session.memory.memory_type_registry import get_default_registry
from openviking.session.memory.memory_updater import ExtractContext
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.train import ExperienceGradientEstimator
from openviking.session.train.components.trajectory_lifecycle import (
    archive_trajectory,
    trajectory_archive_uri,
)

ROOT = "viking://user/u/memories/trajectories/case_a"


@pytest.mark.parametrize("code", ["S", "F", "U"])
def test_archive_moves_date_from_filename_to_directories(code):
    inbox = f"{ROOT}/inbox/20261019_{code}_abcdef.md"
    archive = f"{ROOT}/archive/2026/10/19/{code}_abcdef.md"
    assert trajectory_archive_uri(inbox) == archive
    assert trajectory_archive_uri(archive) == archive


@pytest.mark.parametrize(
    "path",
    [
        "inbox/20260230_S_abcdef.md",
        "inbox/20261019_S_../a.md",
        "inbox/2026/10/19/S_abcdef.md",
        "inbox/S_abcdef.md",
        "archive/2026/13/01/S_abcdef.md",
        "inbox/../archive/2026/10/19/S_abcdef.md",
    ],
)
def test_invalid_paths_are_rejected(path):
    with pytest.raises(ValueError):
        trajectory_archive_uri(f"{ROOT}/{path}")


async def observation(setup, state="applied"):
    fs, _, _, analyzer, context, policies = setup
    run = analysis(passed=True)
    run.rollout.messages[0].created_at = "2020-01-01T00:00:00Z"
    await observe(analyzer, run, policies, context)
    memory = run.metadata["trajectory_memory"]
    memory.extra_fields["experience_update"].update(state=state, attempts=1, batch_id="batch")
    fs.files[memory.uri] = MemoryFileUtils.write(memory)
    return memory


async def archive(setup, memory, db=None):
    return await archive_trajectory(
        memory, viking_fs=setup[0], ctx=setup[1], registry=get_default_registry(), vikingdb=db
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["applied", "no_op"])
async def test_archive_keeps_creation_date_body_source_and_terminal_state(setup_observation, state):
    setup = setup_observation
    memory = await observation(setup, state)
    created = memory.extra_fields["created_at"]
    if isinstance(created, str):
        created = datetime.fromisoformat(created)
    assert f"/inbox/{created:%Y%m%d}_" in memory.uri
    assert "/2020/" not in memory.uri
    fs = setup[0]
    fs.ls = AsyncMock(side_effect=AssertionError("No directory scans during archive"))
    archived = await archive(setup, memory)
    assert memory.uri not in fs.files
    assert archived.uri == trajectory_archive_uri(memory.uri)
    assert f"/archive/{created:%Y/%m/%d}/" in archived.uri
    stored = MemoryFileUtils.read(fs.files[archived.uri], uri=archived.uri)
    assert stored.content == memory.content
    assert stored.links[0]["to_uri"] == memory.links[0]["to_uri"]
    assert stored.links[0]["from_uri"] == archived.uri
    assert stored.extra_fields["experience_update"]["state"] == state
    assert stored.extra_fields["experience_update"]["attempts"] == 1
    # Repeating archival with the original inbox URI is safe.
    assert (await archive(setup, memory)).uri == archived.uri


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "processing", "failed"])
async def test_unfinished_records_stay_in_inbox(setup_observation, state):
    memory = await observation(setup_observation, state)
    before = dict(setup_observation[0].files)
    with pytest.raises(ValueError, match="unfinished"):
        await archive(setup_observation, memory)
    assert setup_observation[0].files == before
    assert not setup_observation[0].policy_lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["copy", "enqueue_false", "enqueue_error", "delete"])
async def test_archive_failures_keep_recovery_record_without_relearning(setup_observation, failure):
    setup = setup_observation
    memory = await observation(setup)
    fs = setup[0]
    db = SimpleNamespace(enqueue_embedding_msg=AsyncMock(return_value=True))
    original_cp, original_rm = fs.cp, fs.rm
    if failure == "copy":
        fs.cp = AsyncMock(side_effect=OSError("copy failed"))
    elif failure == "enqueue_false":
        db.enqueue_embedding_msg.return_value = False
    elif failure == "enqueue_error":
        db.enqueue_embedding_msg.side_effect = OSError("queue failed")
    else:
        fs.rm = AsyncMock(side_effect=OSError("delete failed"))
    with pytest.raises((OSError, RuntimeError)):
        await archive(setup, memory, db)
    assert memory.uri in fs.files
    assert (
        MemoryFileUtils.read(fs.files[memory.uri]).extra_fields["experience_update"]["state"]
        == "applied"
    )
    assert not fs.policy_lock.locked()
    fs.cp, fs.rm = original_cp, original_rm
    db.enqueue_embedding_msg.return_value = True
    db.enqueue_embedding_msg.side_effect = None
    calls_before_retry = db.enqueue_embedding_msg.await_count
    archived = await archive(setup, memory, db)
    assert memory.uri not in fs.files
    assert archived.extra_fields["experience_update"]["attempts"] == 1
    if failure == "delete":
        assert db.enqueue_embedding_msg.await_count == calls_before_retry
    message = db.enqueue_embedding_msg.await_args.args[0]
    assert message.context_data["uri"] == archived.uri
    assert message.context_data["search_tags"] == ["trajectory_outcome=success"]
    assert archived.extra_fields["experience_update"]["embedding_enqueued"] is True


@pytest.mark.asyncio
async def test_archive_collision_never_overwrites_other_evidence(setup_observation):
    memory = await observation(setup_observation)
    target = trajectory_archive_uri(memory.uri)
    other = memory.model_copy(deep=True, update={"uri": target, "content": "different evidence"})
    fs = setup_observation[0]
    fs.files[target] = MemoryFileUtils.write(other)
    before = dict(fs.files)
    with pytest.raises(ValueError, match="collision"):
        await archive(setup_observation, memory)
    assert fs.files == before


@pytest.mark.asyncio
async def test_new_creation_checks_archive_name_collisions(setup_observation, monkeypatch):
    def short_id(self, *, refresh=False):
        if self._random_id is None or refresh:
            self._random_id = "bbbbbb" if refresh else "aaaaaa"
        return self._random_id

    monkeypatch.setattr(ExtractContext, "get_random_id", short_id)
    first = await observation(setup_observation)
    archived = await archive(setup_observation, first)
    before = setup_observation[0].files[archived.uri]
    second = await observation(setup_observation)
    assert second.uri.endswith("_S_bbbbbb.md")
    assert setup_observation[0].files[archived.uri] == before


@pytest.mark.asyncio
async def test_discovery_visits_only_inboxes_and_resolves_old_submission_uris(setup_observation):
    setup = setup_observation
    archived_source = await observation(setup)
    archived = await archive(setup, archived_source)
    pending = await observation(setup, "pending")
    fs = setup[0]
    original_ls = fs.ls

    async def inbox_only(uri, **kwargs):
        assert uri in {ROOT.rsplit("/", 1)[0], ROOT + "/inbox"}
        return await original_ls(uri, **kwargs)

    fs.ls = AsyncMock(side_effect=inbox_only)
    fs.read_file = AsyncMock(wraps=fs.read_file)
    compressor = SessionCompressorV3(vikingdb=None, rollout_analyzer=SimpleNamespace())
    kwargs = {
        "viking_fs": fs,
        "ctx": setup[1],
        "registry": get_default_registry(),
        "case_names": None,
        "retry_failed": False,
    }
    memories = await compressor._load_pending_trajectory_memories(trajectory_uris=None, **kwargs)
    assert [m.uri for m in memories] == [pending.uri]
    assert all(c.args[0] != archived.uri for c in fs.read_file.await_args_list)
    fs.ls.reset_mock()
    memories = await compressor._load_pending_trajectory_memories(
        trajectory_uris=[archived_source.uri], **kwargs
    )
    assert [m.uri for m in memories] == [pending.uri]
    assert [call.args[0] for call in fs.ls.await_args_list] == [ROOT + "/inbox"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cases", [(1, 2)], indirect=True)
async def test_learning_retry_only_finishes_archival_after_copy_failure(cases, monkeypatch):
    synthesize = AsyncMock(return_value=None)
    monkeypatch.setattr(ExperienceGradientEstimator, "_run_extract_loop", synthesize)
    original_cp = cases.fs.cp
    cases.fs.cp = AsyncMock(side_effect=OSError("archive unavailable"))
    result = await asyncio.wait_for(learn(cases), timeout=5)
    assert result["failed_count"] == 1
    assert all(state["state"] == "no_op" for state in states(cases))
    synthesize.assert_awaited_once()
    cases.fs.cp = original_cp
    result = await asyncio.wait_for(learn(cases), timeout=5)
    assert result["trajectory_count"] == 0
    synthesize.assert_awaited_once()
    memory = cases.records[0]
    assert memory.uri not in cases.fs.files
    assert trajectory_archive_uri(memory.uri) in cases.fs.files
    assert states(cases)[0]["attempts"] == 1


@pytest.mark.asyncio
async def test_concurrent_archive_is_idempotent_and_locks_only_two_files(setup_observation):
    memory = await observation(setup_observation)
    fs = setup_observation[0]
    db = SimpleNamespace(enqueue_embedding_msg=AsyncMock(return_value=True))
    fs._async_agfs.pathlock_acquire_exact_batch = AsyncMock(
        wraps=fs._async_agfs.pathlock_acquire_exact_batch
    )
    results = await asyncio.gather(
        archive(setup_observation, memory, db), archive(setup_observation, memory, db)
    )
    assert {r.uri for r in results} == {trajectory_archive_uri(memory.uri)}
    db.enqueue_embedding_msg.assert_awaited_once()
    for call in fs._async_agfs.pathlock_acquire_exact_batch.await_args_list:
        assert set(call.args[0]) == {
            fs._uri_to_path(memory.uri, ctx=setup_observation[1]),
            fs._uri_to_path(results[0].uri, ctx=setup_observation[1]),
        }
    assert not fs.policy_lock.locked()


@pytest.mark.asyncio
async def test_cancelled_archive_keeps_terminal_source_and_releases_locks(setup_observation):
    memory = await observation(setup_observation)
    ready = asyncio.Event()

    async def enqueue(_message):
        ready.set()
        await asyncio.Event().wait()

    db = SimpleNamespace(enqueue_embedding_msg=AsyncMock(side_effect=enqueue))
    task = asyncio.create_task(archive(setup_observation, memory, db))
    try:
        await asyncio.wait_for(ready.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    fs = setup_observation[0]
    assert memory.uri in fs.files
    assert not fs.policy_lock.locked()
    db.enqueue_embedding_msg.side_effect = None
    db.enqueue_embedding_msg.return_value = True
    assert (await archive(setup_observation, memory, db)).uri == trajectory_archive_uri(memory.uri)
    assert memory.uri not in fs.files
