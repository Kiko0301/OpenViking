"""End-to-end incremental Compile against the real ragfs binding and git.

Runtime is not called: the test drives the provider hooks the queue worker would
run, so it checks the kernel half (registry, snapshots, change plan, finish).
"""

import asyncio
import base64
from datetime import datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service.compile_scheduler import TZ, current_slot, next_slot
from openviking.service.compile_service import CompileConfigUpdate, CompileRequest
from openviking.service.compile_targets import CompileStore, CompileVersions, parse_schedule
from openviking.service.external_task_service import ExternalTaskSnapshot
from openviking.service.task_tracker import get_task_tracker
from openviking.storage.content_write import ContentWriteCoordinator
from openviking.storage.queuefs import get_queue_manager
from openviking_cli.exceptions import ConflictError, InvalidArgumentError
from openviking_cli.session.user_id import UserIdentifier

pytest.importorskip("ragfs_python")

SRC = "viking://resources/compile_src"
TO = "viking://resources/compile_wiki"
SKILL = "viking://user/test_user/skills/wiki"


@pytest.fixture(autouse=True)
def _reindex_submission(service, monkeypatch):
    monkeypatch.setattr("openviking.server.dependencies.get_service", lambda: service)
    monkeypatch.setattr("openviking.service.reindex_executor.get_service", lambda: service)
    monkeypatch.setattr(service, "reindex", AsyncMock(return_value={"status": "accepted"}))


@pytest.fixture
async def compile_task(service):
    """Create a task with existing source/output files before its start snapshot."""
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.USER)
    for uri in (f"{SRC}/a.md", f"{TO}/page.md", f"{SKILL}/SKILL.md"):
        await _write(service, uri, "before", ctx)
    service.compile.configure_local_backend("http://127.0.0.1:1", "")
    task = await service.compile.create(
        CompileRequest.model_validate({"from": [SRC], "to": TO, "skill": SKILL}),
        connection={},
        ctx=ctx,
    )
    return ctx, task


async def _write(service, uri, text, ctx):
    await service.viking_fs.write_file(uri, text, ctx=ctx)


@pytest.fixture(params=["single", "batch"])
def compile_write(service, request):
    """Exercise both execution adapters against the same lifecycle assertions."""

    async def write(ctx, uri, content):
        if request.param == "single":
            return await service.fs.write(uri, content, ctx, edit_source="compile", wait=True)
        return await service.fs.batch_write(
            root_uri=TO,
            operations=[{"uri": uri, "content": content}],
            ctx=ctx,
            defer_processing=True,
            edit_source="compile",
        )

    return write


async def _run(
    service, ctx, *, write_page=None, to=TO, scheduled=False, status="completed", **request_fields
):
    compile_service = service.compile
    compile_service.configure_local_backend("http://127.0.0.1:1", "")
    if scheduled:
        store = compile_service.lifecycle.store
        record = await store.get_by_to(ctx.account_id, to)
        async with store.edit(ctx.account_id, record["id"]) as record:
            task = await compile_service.create_scheduled(
                record, ctx=ctx, connection={}, idempotency_key=f"schedule-test-{uuid4().hex}"
            )
    else:
        request = CompileRequest.model_validate(
            {"from": [SRC], "to": to, "skill": SKILL, **request_fields}
        )
        task = await compile_service.create(request, connection={}, ctx=ctx)
    task = await get_task_tracker().get(task.task_id, ctx.account_id, ctx.user.user_id)
    started = await compile_service.on_start(task)
    changes = None
    if started == "proceed":
        changes = await compile_service.lifecycle.runtime_changes(
            task.meta["request"], ctx.account_id, task.task_id
        )
        if write_page:
            await _write(service, f"{to}/page.md", write_page, ctx)
    await compile_service._tasks._apply_snapshot(
        started
        if isinstance(started, ExternalTaskSnapshot)
        else ExternalTaskSnapshot(status=status),
        task_id=task.task_id,
        account_id=ctx.account_id,
        user_id=ctx.user.user_id,
    )
    return started, changes


def test_schedule_slots():
    def at(day, hour, minute):
        return datetime(2026, 10, day, hour, minute, tzinfo=TZ)

    daily = parse_schedule({"frequency": "daily", "hour": 2, "minute": 30})
    assert current_slot(daily, at(13, 2, 29)) == at(12, 2, 30)
    assert current_slot(daily, at(13, 2, 30)) == at(13, 2, 30)
    assert next_slot(daily, at(13, 2, 30)) == at(14, 2, 30)
    # 2026-10-13 is a Tuesday; weekly defaults to Monday 00:00.
    assert current_slot(parse_schedule({"frequency": "weekly"}), at(13, 1, 0)) == at(12, 0, 0)
    sunday = parse_schedule({"frequency": "weekly", "weekday": 7, "hour": 23, "minute": 59})
    assert current_slot(sunday, at(13, 1, 0)) == at(11, 23, 59)
    assert next_slot(sunday, at(13, 1, 0)) == at(18, 23, 59)


@pytest.mark.asyncio
async def test_operation_history_pagination(service, monkeypatch):
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.USER)
    store = service.compile.lifecycle.store
    target_id = await store.register(ctx.account_id, TO)
    operations = [
        {"operation_id": "op_b", "created_at": 2},
        {"operation_id": "op_missing"},
        {"operation_id": "op_latest", "created_at": 3},
        {"operation_id": "op_a", "created_at": 2},
        {"operation_id": "op_none", "created_at": None},
        {"operation_id": "op_old", "created_at": 0},
    ]
    for operation in operations:
        await store.put_operation(ctx.account_id, target_id, operation)
    expected = ["op_latest", "op_b", "op_a", "op_old", "op_none", "op_missing"]
    all_operations = await store.list_operations(ctx.account_id, target_id)
    assert [item["operation_id"] for item in all_operations] == expected
    page = await service.compile.list_operations(TO, ctx, limit=4, cursor=None)
    assert [item["operation_id"] for item in page["operations"]] == expected[:4]
    assert page["next_cursor"] == expected[3]
    page = await service.compile.list_operations(TO, ctx, limit=4, cursor=page["next_cursor"])
    assert [item["operation_id"] for item in page["operations"]] == expected[4:]
    assert page["next_cursor"] is None
    read_json = AsyncMock(wraps=store.read_json)
    monkeypatch.setattr(store, "read_json", read_json)
    for cursor in (expected[-1], "unknown", "../state"):
        assert await service.compile.list_operations(TO, ctx, limit=2, cursor=cursor) == {
            "operations": [],
            "next_cursor": None,
        }
    assert all("/../" not in call.args[0] for call in read_json.await_args_list)


@pytest.mark.asyncio
async def test_file_version_metadata(service):
    """Legacy reads and appended versions use nested metadata."""
    store = service.compile.lifecycle.store
    versions = CompileVersions(Mock(spec_set=CompileStore, wraps=store), service.viking_fs)
    uri, target_id = f"{TO}/legacy.md", "target_legacy"
    defaults = {"task_id": None, "restored_from": None, "after_compile_snapshot": False}
    meta = {"task_id": "cmp_running", "restored_from": None, "after_compile_snapshot": True}
    legacy = {
        "uri": uri,
        "next": 2,
        "source_roots": [SRC],
        "versions": [{"version_id": "v1", "source": "manual", **meta}],
    }
    await store.put_version_index("default", target_id, legacy)
    loaded = await versions.load("default", target_id, uri)
    assert set(loaded) == {"uri", "next", "versions"}
    assert await versions._indexes("default", target_id) == [loaded]
    assert versions._ordinary_version(loaded, "cmp_running")["meta"] == meta
    await versions.append(
        "default", target_id, uri, "snapshot_2", {"source": "external", "actor": None}
    )
    stored = await store.get_version_index("default", target_id, uri)
    assert set(stored) == {"uri", "next", "versions"}
    assert [version["meta"] for version in stored["versions"]] == [meta, defaults]
    for version in stored["versions"]:
        assert not (set(defaults) & version.keys())


@pytest.mark.asyncio
async def test_incremental_compile_on_real_git(service):
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.ROOT)
    for uri, text in [(f"{SRC}/a.md", "a"), (f"{SRC}/b.md", "b"), (f"{SKILL}/SKILL.md", "s")]:
        await _write(service, uri, text, ctx)

    started, changes = await _run(service, ctx, write_page="v1")
    assert started == "proceed"
    assert changes["scopes"] == [{"uri": SRC, "change_type": "added", "recursive": True}]
    updated_at = (await service.compile.get_config(TO, ctx))["content_updated_at"]
    assert isinstance(updated_at, float)

    started, _ = await _run(service, ctx)
    assert isinstance(started, ExternalTaskSnapshot) and started.status == "completed"
    assert (await service.compile.get_config(TO, ctx))["content_updated_at"] == updated_at

    await _write(service, f"{SRC}/a.md", "a2", ctx)
    started, changes = await _run(service, ctx, write_page="v2")
    assert started == "proceed"
    assert changes["scopes"] == [
        {"uri": f"{SRC}/a.md", "change_type": "modified", "recursive": False}
    ]
    config = await service.compile.get_config(TO, ctx)
    assert config["initialized"] is True and config["last_run"]["status"] == "completed"
    assert config["content_updated_at"] > updated_at
    runs = await service.compile.list_operations(TO, ctx, limit=10, cursor=None)
    assert [run["result"] for run in runs["operations"] if run["result"]][0]["updated"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["instruction", "clear", "skill_uri", "content", "attachment"])
async def test_compile_inputs_and_skill_snapshots(service, changed):
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.USER)
    for uri in (f"{SRC}/a.md", f"{SKILL}/SKILL.md"):
        await _write(service, uri, "original", ctx)
    params = {"instruction": "refresh" if changed == "clear" else None}
    await _run(service, ctx, write_page="v1", **params)
    if changed in {"instruction", "clear"}:
        params["instruction"] = " \n " if changed == "clear" else " refresh "
    else:
        if changed == "skill_uri":
            params["skill"] = f"{SKILL}-other"
        path = "references/rules.md" if changed == "attachment" else "SKILL.md"
        await _write(service, f"{params.get('skill', SKILL)}/{path}", "changed", ctx)
    started, changes = await _run(service, ctx, write_page="v2", **params)
    assert started == "proceed" and changes["execution"] == {"mode": "full"}
    assert changes["scopes"] == [{"uri": SRC, "change_type": "added", "recursive": True}]
    snapshot = changes["skill_snapshot"]
    assert set(snapshot) == {"uri", "snapshot_id"}
    assert await service.viking_fs.show(
        snapshot["snapshot_id"], path=f"{snapshot['uri']}/SKILL.md", ctx=ctx
    ) == (b"changed" if changed in {"skill_uri", "content"} else b"original")
    operation = (await service.compile.list_operations(TO, ctx, limit=1, cursor=None))[
        "operations"
    ][0]
    assert operation["skill_snapshot"] == snapshot
    assert operation["params"]["args"]["mode"] == "incremental" and operation["head"]
    versions = await service.compile.file_versions(f"{TO}/page.md", ctx, limit=10, before=None)
    assert [v["source"] for v in versions["versions"]] == ["compile", "compile"]
    for instruction in [None, "", " \t\n "] if changed == "clear" else [params["instruction"]]:
        started, _ = await _run(service, ctx, **{**params, "instruction": instruction})
        assert isinstance(started, ExternalTaskSnapshot) and started.status == "completed"
    if changed == "clear":
        started, _ = await _run(service, ctx)
        assert isinstance(started, ExternalTaskSnapshot) and started.status == "completed"
    await _write(service, f"{SRC}/a.md", "a2", ctx)
    started, changes = await _run(service, ctx, **params)
    assert started == "proceed" and changes["execution"]["mode"] == "incremental"
    assert changes["scopes"] == [
        {"uri": f"{SRC}/a.md", "change_type": "modified", "recursive": False}
    ]


@pytest.mark.asyncio
async def test_scheduled_compile_keeps_its_config_and_successful_baseline(service, monkeypatch):
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.USER)
    other_skill = f"{SKILL}-other"
    for uri in (f"{SRC}/a.md", f"{SRC}_other/a.md", f"{SKILL}/SKILL.md", f"{other_skill}/SKILL.md"):
        await _write(service, uri, "original", ctx)
    lifecycle = service.compile.lifecycle
    monkeypatch.setattr(lifecycle, "save_credential", AsyncMock(return_value=({}, "credential")))
    await _run(service, ctx, instruction="scheduled", args={"auto_update": True})
    started, changes = await _run(service, ctx, scheduled=True)
    assert started == "proceed" and changes["execution"]["mode"] == "full"
    before = await lifecycle.store.get_by_to(ctx.account_id, TO)
    await _run(service, ctx, skill=other_skill, instruction="manual", **{"from": [f"{SRC}_other"]})
    after = await lifecycle.store.get_by_to(ctx.account_id, TO)
    assert after["from"] == [SRC]
    assert (after["skill"], after["instruction"], after["scheduled_progress"]) == (
        SKILL,
        "scheduled",
        before["scheduled_progress"],
    )
    started, _ = await _run(service, ctx, scheduled=True)
    assert isinstance(started, ExternalTaskSnapshot) and started.status == "completed"
    for change in ("skill", "instruction"):
        if change == "skill":
            await _write(service, f"{SKILL}/SKILL.md", "new skill", ctx)
        else:
            await service.compile.update_config(
                CompileConfigUpdate.model_validate(
                    {
                        "to": TO,
                        "from": [SRC],
                        "skill": SKILL,
                        "instruction": "new instruction",
                        "auto_update": True,
                        "update_schedule": {"frequency": "daily"},
                    }
                ),
                ctx,
            )
        started, changes = await _run(service, ctx, scheduled=True, write_page=change)
        assert started == "proceed" and changes["execution"]["mode"] == "full"
        operation = (await service.compile.list_operations(TO, ctx, limit=1, cursor=None))[
            "operations"
        ][0]
        assert operation["operation_type"] == "schedule_compile" and operation["head"]
        started, _ = await _run(service, ctx, scheduled=True)
        assert isinstance(started, ExternalTaskSnapshot) and started.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "legacy"])
async def test_compile_retries_and_missing_successful_baseline(service, status):
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.USER)
    for uri in (f"{SRC}/a.md", f"{SKILL}/SKILL.md"):
        await _write(service, uri, "original", ctx)
    await _run(service, ctx)
    store = service.compile.lifecycle.store
    record = await store.get_by_to(ctx.account_id, TO)
    first = await get_task_tracker().get(
        record["last_run"]["task_id"], ctx.account_id, ctx.user.user_id
    )
    if status == "legacy":
        async with store.edit(ctx.account_id, record["id"]) as record:
            record["progress"].pop("skill_snapshot")
    else:
        await _run(service, ctx, instruction="refresh", status=status)
        assert (await store.get_by_to(ctx.account_id, TO))["progress"] == record["progress"]
    instruction = None if status == "legacy" else "refresh"
    started, changes = await _run(service, ctx, instruction=instruction)
    assert started == "proceed" and changes["execution"]["mode"] == "full"
    successful = (await store.get_by_to(ctx.account_id, TO))["progress"]
    await service.compile.on_finish(first, await service.compile.on_start(first))
    assert (await store.get_by_to(ctx.account_id, TO))["progress"] == successful
    started, _ = await _run(service, ctx, instruction=instruction)
    assert isinstance(started, ExternalTaskSnapshot) and started.status == "completed"


@pytest.mark.asyncio
async def test_first_compile_created_at(service, compile_task):
    ctx, first = compile_task
    store = service.compile.lifecycle.store
    field = "first_compile_created_at"
    initial = await store.get_operation(
        ctx.account_id, first.meta["request"]["compile"]["target_id"], first.task_id
    )
    assert initial["operation_type"] == "initial" and initial["conflicts_num"] == 0
    assert await store.get_outputs(
        ctx.account_id, first.meta["request"]["compile"]["target_id"], first.task_id
    ) == {"outputs": []}
    assert (await service.compile.get_config(TO, ctx))[field] == first.created_at
    assert (await store.list_summaries(ctx.account_id))[0][field] == first.created_at
    request = CompileRequest.model_validate({"from": [SRC], "to": TO, "skill": SKILL})
    params = {"connection": {}, "ctx": ctx, "idempotency_key": "first-created-retry"}
    task = await service.compile.create(request, **params)
    assert task.task_id != first.task_id
    operation = await store.get_operation(
        ctx.account_id, first.meta["request"]["compile"]["target_id"], task.task_id
    )
    assert operation["operation_type"] == "manual_compile"
    assert (await service.compile.create(request, **params)).task_id == task.task_id
    assert (await service.compile.get_config(TO, ctx))[field] == first.created_at
    assert (await service.compile.list_configs(ctx))[0][field] == first.created_at

    target_id = first.meta["request"]["compile"]["target_id"]
    async with store.edit(ctx.account_id, target_id) as record:
        scheduled = await service.compile.create_scheduled(
            record, ctx=ctx, connection={}, idempotency_key="operation-schedule-test"
        )
    operation = await store.get_operation(ctx.account_id, target_id, scheduled.task_id)
    assert operation["operation_type"] == "schedule_compile"
    assert await store.get_outputs(ctx.account_id, target_id, scheduled.task_id) == {"outputs": []}
    async with store.edit(ctx.account_id, target_id) as record:
        record.pop(field)
    await service.compile.create(request, **params)
    assert (await service.compile.get_config(TO, ctx))[field] is None
    assert (await service.compile.list_configs(ctx))[0][field] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("locked_uri", [f"{SRC}/a.md", f"{TO}/page.md"])
async def test_busy_start_requeues_until_path_unlocks(service, monkeypatch, locked_uri):
    """Source or target file locks defer the same task without submitting Runtime work."""
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.USER)
    for uri in (f"{SRC}/a.md", f"{TO}/page.md", f"{SKILL}/SKILL.md"):
        await _write(service, uri, "content", ctx)
    compile_service = service.compile
    compile_service.configure_local_backend("http://127.0.0.1:1", "")
    submit = AsyncMock(return_value="runtime-id")
    monkeypatch.setattr(compile_service, "submit", submit)
    monkeypatch.setattr(
        compile_service, "get", AsyncMock(return_value=ExternalTaskSnapshot(status="completed"))
    )
    task = await compile_service.create(
        CompileRequest.model_validate({"from": [SRC], "to": TO, "skill": SKILL}),
        connection={},
        ctx=ctx,
    )
    queue = get_queue_manager().get_queue("ExternalTask")
    tracker = get_task_tracker()
    agfs = service.viking_fs._async_agfs
    lease = await agfs.pathlock_acquire_exact(service.viking_fs._uri_to_path(locked_uri, ctx=ctx))
    try:
        for _ in range(2):
            await asyncio.wait_for(queue.dequeue(), timeout=2)
            current = await tracker.get(task.task_id, ctx.account_id, ctx.user.user_id)
            assert current.status == "pending" and current.error is None
            status = await queue.get_status()
            assert (status.pending, status.in_progress, status.requeue_count) == (1, 0, _ + 1)
            submit.assert_not_awaited()
    finally:
        await agfs.pathlock_release(lease)
    await queue.dequeue()
    assert (await tracker.get(task.task_id, ctx.account_id, ctx.user.user_id)).status == "completed"
    submit.assert_awaited_once()
    assert (await queue.get_status()).is_complete


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["pending", "requeue"])
async def test_queued_cancel_finishes_history_without_releasing_running_task(
    service, compile_task, monkeypatch, cancel_at
):
    """Cancel B before dispatch or during rotation while A retains the shared target."""
    ctx, first = compile_task
    compile_service = service.compile
    tracker = get_task_tracker()
    manager = get_queue_manager()
    queue = manager.get_queue("ExternalTask")
    entered, release = asyncio.Event(), asyncio.Event()

    async def poll(task_id, *args, **kwargs):
        if task_id == first.task_id:
            entered.set()
            await release.wait()
        return ExternalTaskSnapshot(status="completed")

    submit = AsyncMock(side_effect=lambda task_id, *args, **kwargs: task_id)
    monkeypatch.setattr(compile_service, "submit", submit)
    monkeypatch.setattr(compile_service, "get", poll)
    worker = asyncio.create_task(queue.dequeue())
    await asyncio.wait_for(entered.wait(), 5)
    store = compile_service.lifecycle.store
    target_id = first.meta["request"]["compile"]["target_id"]
    await service.fs.rm(f"{TO}/page.md", ctx, wait=True)
    protection = await store.get_outputs(ctx.account_id, target_id, first.task_id)
    try:
        request = CompileRequest.model_validate(
            {"from": [SRC], "to": TO, "skill": SKILL, "instruction": "refresh"}
        )
        queued = await compile_service.create(request, connection={}, ctx=ctx)
        if cancel_at == "pending":
            await tracker.cancel(
                queued.task_id, account_id=ctx.account_id, user_id=ctx.user.user_id
            )
        else:
            enqueue = manager.enqueue

            async def cancel_before_rotation(queue_name, data, **kwargs):
                if queue_name == manager.EXTERNAL_TASK and data.get("task_id") == queued.task_id:
                    await tracker.record_cancelled(queued.task_id, ctx.account_id, ctx.user.user_id)
                return await enqueue(queue_name, data, **kwargs)

            monkeypatch.setattr(manager, "enqueue", cancel_before_rotation)
        await queue.dequeue()
        current = await tracker.get(queued.task_id, ctx.account_id, ctx.user.user_id)
        assert current.status == "cancelled"
        run = await store.get_operation(ctx.account_id, target_id, queued.task_id)
        assert run["status"] == "cancelled" and run["finished_at"] is not None
        assert run["started_at"] is None
        assert (await store.get(ctx.account_id, target_id))["active_task_id"] == first.task_id
        assert await store.get_outputs(ctx.account_id, target_id, first.task_id) == protection
    finally:
        release.set()
        await asyncio.wait_for(worker, 10)
    assert (
        await tracker.get(first.task_id, ctx.account_id, ctx.user.user_id)
    ).status == "completed"
    assert (await store.get(ctx.account_id, target_id))["active_task_id"] is None
    submit.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_waits_for_output_lock_before_releasing_target(
    service, compile_task, monkeypatch
):
    """Cancellation retains target ownership until rollback can preserve ordinary edits."""
    from openviking.storage.errors import ResourceBusyError

    ctx, task = compile_task
    compile_service = service.compile
    lifecycle = compile_service.lifecycle
    tracker = get_task_tracker()
    queue = get_queue_manager().get_queue("ExternalTask")
    polling, busy = asyncio.Event(), asyncio.Event()

    async def poll(*args, **kwargs):
        polling.set()
        await asyncio.Event().wait()

    finish = lifecycle.on_finish

    async def finish_with_lock_check(*args, **kwargs):
        try:
            return await finish(*args, **kwargs)
        except ResourceBusyError:
            busy.set()
            raise

    monkeypatch.setattr(compile_service, "submit", AsyncMock(return_value="runtime-task"))
    monkeypatch.setattr(compile_service, "get", poll)
    monkeypatch.setattr(
        compile_service, "cancel", AsyncMock(return_value=ExternalTaskSnapshot(status="cancelled"))
    )
    monkeypatch.setattr(lifecycle, "on_finish", finish_with_lock_check)
    worker = asyncio.create_task(queue.dequeue())
    await asyncio.wait_for(polling.wait(), 5)
    uri = f"{TO}/page.md"
    await service.fs.write(uri, "ordinary", ctx, processing_mode="vectors_only", wait=True)
    await _write(service, uri, "compile output", ctx)
    agfs = service.viking_fs._async_agfs
    lease = await agfs.pathlock_acquire_exact(service.viking_fs._uri_to_path(uri, ctx=ctx))
    try:
        await tracker.cancel(task.task_id, account_id=ctx.account_id, user_id=ctx.user.user_id)
        await asyncio.wait_for(busy.wait(), 5)
        await asyncio.sleep(0.05)
        current = await tracker.get(task.task_id, ctx.account_id, ctx.user.user_id)
        assert current.status == "cancelling"
        record = await lifecycle.store.get(
            ctx.account_id, task.meta["request"]["compile"]["target_id"]
        )
        run = await lifecycle.store.get_operation(ctx.account_id, record["id"], task.task_id)
        assert run["finished_at"] is None and record["active_task_id"] == task.task_id
    finally:
        await agfs.pathlock_release(lease)
        await asyncio.wait_for(worker, 10)
    assert (await tracker.get(task.task_id, ctx.account_id, ctx.user.user_id)).status == "cancelled"
    record = await lifecycle.store.get(ctx.account_id, record["id"])
    run = await lifecycle.store.get_operation(ctx.account_id, record["id"], task.task_id)
    assert run["status"] == "cancelled" and run["finished_at"] is not None
    assert record["active_task_id"] is None
    assert await service.viking_fs.read_file(uri, ctx=ctx) == "ordinary"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["commit", "reindex", "outputs", "git_disabled", "release"])
async def test_finish_commits_then_reindexes_and_resumes(
    service, monkeypatch, compile_task, failure
):
    from openviking.pyagfs.exceptions import AGFSNotSupportedError

    ctx, task = compile_task
    compile_service = service.compile
    queue = get_queue_manager().get_queue("ExternalTask")
    delivery = await queue.dequeue_raw()
    if failure == "git_disabled":
        monkeypatch.setattr(
            service.viking_fs, "commit", AsyncMock(side_effect=AGFSNotSupportedError("disabled"))
        )
    await compile_service.on_start(task)
    lifecycle = compile_service.lifecycle
    write_json = AsyncMock(wraps=lifecycle.store.write_json)
    monkeypatch.setattr(lifecycle.store, "write_json", write_json)
    for content in ("before", "published", "before", "published"):
        await service.fs.write(f"{TO}/page.md", content, ctx, edit_source="compile", wait=True)
    assert (await compile_service.get_config(TO, ctx))["content_updated_at"] is None
    assert not any(
        call.args[0].endswith(("/state.json", "/target_folders.json"))
        for call in write_json.await_args_list
    )
    if failure != "git_disabled":
        write_json.assert_awaited_once()
    commit_spy = AsyncMock(wraps=lifecycle._commit)
    monkeypatch.setattr(lifecycle, "_commit", commit_spy)
    merge_outputs = lifecycle.store.merge_outputs
    release = lifecycle._release
    if failure == "commit":
        commit_spy.side_effect = OSError("commit failed")
    elif failure == "outputs":
        monkeypatch.setattr(
            lifecycle.store, "merge_outputs", AsyncMock(side_effect=OSError("outputs failed"))
        )
    elif failure == "release":
        monkeypatch.setattr(lifecycle, "_release", AsyncMock(side_effect=OSError("release failed")))
    else:
        service.reindex.side_effect = OSError("reindex failed")
    snapshot = ExternalTaskSnapshot(status="completed", meta={"commit_oid": "ignored"})
    with pytest.raises(OSError):
        await compile_service.on_finish(task, snapshot)
    run = await lifecycle.store.get_operation(
        ctx.account_id, task.meta["request"]["compile"]["target_id"], task.task_id
    )
    assert (run["finished_at"] is not None) == (failure == "release")
    resumed = await compile_service.on_start(task)
    assert isinstance(resumed, ExternalTaskSnapshot) and resumed.status == "completed"
    if failure == "commit":
        service.reindex.assert_not_awaited()
    else:
        assert run["snapshot_committed"] is True
    if failure == "release":
        await service.fs.write(f"{TO}/page.md", "ordinary", ctx, edit_source="manual", wait=True)
        ordinary_updated_at = (await compile_service.get_config(TO, ctx))["content_updated_at"]
    commit_spy.side_effect = None
    monkeypatch.setattr(lifecycle.store, "merge_outputs", merge_outputs)
    monkeypatch.setattr(lifecycle, "_release", release)
    service.reindex.side_effect = None
    await compile_service.on_finish(task, snapshot)
    updated_at = (await compile_service.get_config(TO, ctx))["content_updated_at"]
    assert isinstance(updated_at, float)
    if failure == "release":
        assert updated_at == ordinary_updated_at
        assert await service.viking_fs.read_file(f"{TO}/page.md", ctx=ctx) == "ordinary"
    await compile_service.on_finish(task, snapshot)
    assert (await compile_service.get_config(TO, ctx))["content_updated_at"] == updated_at
    assert commit_spy.await_count == (2 if failure == "commit" else 1)
    kwargs = service.reindex.await_args.kwargs
    assert {k: kwargs[k] for k in ("uri", "mode", "force", "recursive", "wait")} == {
        "uri": TO,
        "mode": "semantic_and_vectors",
        "force": False,
        "recursive": True,
        "wait": False,
    }
    run = await lifecycle.store.get_operation(
        ctx.account_id, task.meta["request"]["compile"]["target_id"], task.task_id
    )
    assert run["status"] == "completed" and run["reindex_task_id"] == kwargs["task_id"]
    outputs = await compile_service.operation_outputs(TO, task.task_id, ctx, limit=10, cursor=None)
    assert len(outputs["outputs"]) == 1
    if failure == "git_disabled":
        assert outputs["versions_available"] is False
        assert outputs["outputs"][0]["version_id"] is None
    if failure != "git_disabled":
        versions = await compile_service.file_versions(f"{TO}/page.md", ctx, limit=5, before=None)
        assert sum(v["source"] == "compile" for v in versions["versions"]) == 1
    await queue.ack(delivery["id"], delivery)


@pytest.mark.parametrize(
    "git_enabled,status", [(True, "completed"), (False, "failed"), (False, "cancelled")]
)
async def test_compile_timestamp_defers_round_trip_writes(
    service, compile_task, compile_write, monkeypatch, git_enabled, status
):
    """Reverted writes retain deferred timestamps across task recovery, with or without Git."""
    from openviking.pyagfs.exceptions import AGFSNotSupportedError
    from openviking.service.compile_targets import CompileLifecycle

    ctx, task = compile_task
    queue = get_queue_manager().get_queue("ExternalTask")
    delivery = await queue.dequeue_raw()
    if not git_enabled:
        monkeypatch.setattr(
            service.viking_fs, "commit", AsyncMock(side_effect=AGFSNotSupportedError("disabled"))
        )
    await service.compile.on_start(task)
    for content in ("changed", "before"):
        await compile_write(ctx, f"{TO}/page.md", content)
    assert (await service.compile.get_config(TO, ctx))["content_updated_at"] is None
    restarted = CompileLifecycle(service.compile.lifecycle.store, service.viking_fs)
    await restarted.on_finish(task, ExternalTaskSnapshot(status=status))
    assert isinstance((await service.compile.get_config(TO, ctx))["content_updated_at"], float)
    await queue.ack(delivery["id"], delivery)


@pytest.mark.asyncio
async def test_finish_accepts_async_reindex_before_completing_task(
    service, monkeypatch, compile_task
):
    from openviking.service.core import OpenVikingService
    from openviking.service.task_work_index import extract_task_metadata

    monkeypatch.setattr(service, "reindex", OpenVikingService.reindex.__get__(service))
    ctx, task = compile_task
    queue = get_queue_manager().get_queue("ExternalTask")
    delivery = await queue.dequeue_raw()
    await service.compile.on_start(task)
    await service.fs.batch_write(
        root_uri=TO,
        operations=[{"uri": f"{TO}/page.md", "content": "published"}],
        ctx=ctx,
        defer_processing=True,
        edit_source="compile",
    )
    assert await service.viking_fs.read_file(f"{TO}/page.md", ctx=ctx) == "published"
    await _write(service, f"{TO}/page.md", "published", ctx)
    identity = {"account_id": ctx.account_id, "user_id": ctx.user.user_id}
    tracker = get_task_tracker()
    await tracker.start(task.task_id, **identity)
    persist = service.compile.lifecycle.store.put_operation

    async def fail_final_record(account, target, run):
        if run["finished_at"] is not None:
            raise OSError("final record interrupted")
        await persist(account, target, run)

    monkeypatch.setattr(service.compile.lifecycle.store, "put_operation", fail_final_record)
    with pytest.raises(OSError, match="final record interrupted"):
        await service.compile._tasks._apply_snapshot(
            ExternalTaskSnapshot(status="completed"), task_id=task.task_id, **identity
        )
    assert (await tracker.get(task.task_id, **identity)).status == "running"
    monkeypatch.setattr(service.compile.lifecycle.store, "put_operation", persist)
    await service.compile._tasks._apply_snapshot(
        ExternalTaskSnapshot(status="completed"), task_id=task.task_id, **identity
    )
    await queue.ack(delivery["id"], delivery)
    metadata = extract_task_metadata(delivery)
    assert metadata is not None
    await tracker.mark_work_done(task.task_id, metadata.work_id, **identity)
    assert (await tracker.get(task.task_id, **identity)).status == "completed"
    run = (await service.compile.list_operations(TO, ctx, limit=1, cursor=None))["operations"][0]
    reindex = await tracker.get(run["reindex_task_id"], **identity)
    assert reindex.task_type == "admin_reindex" and reindex.resource_id == TO


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_unsuccessful_compile_preserves_ordinary_versions(
    service, compile_task, compile_write, status
):
    ctx, task = compile_task
    await _write(service, f"{TO}/compile-only.md", "before", ctx)
    await service.compile.on_start(task)
    for name in ("page.md", "ordinary-new.md"):
        await service.fs.write(f"{TO}/{name}", "ordinary", ctx, edit_source="manual", wait=True)
    for name in ("page.md", "ordinary-new.md", "compile-only.md", "new.md"):
        await compile_write(ctx, f"{TO}/{name}", "compiled")
    # Returning to start bytes must still restore the latest ordinary version.
    await service.fs.batch_write(
        root_uri=TO,
        operations=[{"uri": f"{TO}/page.md", "content": "ordinary latest"}],
        ctx=ctx,
        defer_processing=True,
    )
    await compile_write(ctx, f"{TO}/page.md", "before")
    await service.fs.write(f"{TO}/last.md", "ordinary last", ctx, wait=True)
    await service.compile.on_finish(task, ExternalTaskSnapshot(status=status))
    await service.compile.on_finish(task, ExternalTaskSnapshot(status=status))
    for name in ("page.md", "ordinary-new.md"):
        uri = f"{TO}/{name}"
        expected = "ordinary latest" if name == "page.md" else "ordinary"
        assert await service.viking_fs.read_file(uri, ctx=ctx) == expected
        history = await service.compile.file_versions(uri, ctx, limit=10, before=None)
        latest = history["versions"][0]
        assert latest["source"] == ("agent" if name == "page.md" else "manual")
        assert latest["provenance"]["source_roots"] == [uri]
        assert (
            await service.viking_fs.show(latest["snapshot_id"], path=uri, ctx=ctx)
            == expected.encode()
        )
    assert await service.viking_fs.read_file(f"{TO}/compile-only.md", ctx=ctx) == "before"
    assert await service.viking_fs.read_file(f"{TO}/last.md", ctx=ctx) == "ordinary last"
    assert not await service.viking_fs.exists(f"{TO}/new.md", ctx=ctx)
    outputs = await service.compile.operation_outputs(TO, task.task_id, ctx, limit=100, cursor=None)
    assert {item["uri"] for item in outputs["outputs"]} == {
        f"{TO}/{name}" for name in ("page.md", "ordinary-new.md", "compile-only.md", "new.md")
    }
    for item in outputs["outputs"]:
        assert await service.viking_fs.show(item["snapshot_id"], path=item["uri"], ctx=ctx) in (
            b"compiled",
            b"before",
        )
    service.reindex.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("directory,ordinary", [(False, False), (True, False), (True, True)])
async def test_compile_rollback_preserves_user_deletions(
    service, compile_task, monkeypatch, status, directory, ordinary
):
    """Deleted scopes stay deleted while subsequent ordinary recreations survive rollback."""
    ctx, task = compile_task
    monkeypatch.setattr(
        service.viking_fs, "_schedule_restore_reindex_for_paths", AsyncMock(return_value=None)
    )
    scope = f"{TO}/removed" if directory else f"{TO}/page.md"
    page = f"{scope}/page.md" if directory else scope
    sibling, compiled_delete = f"{scope}-sibling.md", f"{TO}/compile-deleted.md"
    for uri in (page, sibling, compiled_delete):
        await _write(service, uri, "before", ctx)
    await service.compile.on_start(task)
    await service.fs.rm(scope + "/" if directory else scope, ctx, recursive=directory, wait=True)
    kept = f"{scope}/kept.md"
    if ordinary:
        await service.fs.write(kept, "ordinary", ctx, edit_source="manual", wait=True)
    await service.fs.batch_write(
        root_uri=TO,
        operations=[{"uri": page, "content": "before"}]
        + ([{"uri": kept, "content": "compiled"}] if ordinary else []),
        ctx=ctx,
        defer_processing=True,
        edit_source="compile",
    )
    if directory:
        await _write(service, f"{scope}/generated/.overview.md", "generated", ctx)
    await service.viking_fs.rm(compiled_delete, ctx=ctx)
    store, target_id = service.compile.lifecycle.store, task.meta["request"]["compile"]["target_id"]
    run = await store.get_operation(ctx.account_id, target_id, task.task_id)
    await service.compile.lifecycle._rollback(TO, run, ctx)
    history = await service.compile.list_operations(TO, ctx, limit=20, cursor=None)
    assert any(item["operation_type"] == "rm" for item in history["operations"])
    await service.compile.on_finish(task, ExternalTaskSnapshot(status=status))
    assert not await service.viking_fs.exists(page, ctx=ctx)
    if ordinary:
        assert await service.viking_fs.read_file(kept, ctx=ctx) == "ordinary"
        assert not await service.viking_fs.exists(f"{scope}/generated", ctx=ctx)
    else:
        assert not await service.viking_fs.exists(scope, ctx=ctx)
    assert await service.viking_fs.read_file(sibling, ctx=ctx) == "before"
    assert await service.viking_fs.read_file(compiled_delete, ctx=ctx) == "before"


@pytest.mark.asyncio
async def test_compile_delete_failure_and_cancellation(service, compile_task, monkeypatch):
    """Failed deletes have no effects; cancellation finishes successful deletion metadata."""
    from openviking.storage.errors import LockAcquisitionError

    ctx, task = compile_task
    uri = f"{TO}/page.md"
    await service.compile.on_start(task)
    store, target_id = service.compile.lifecycle.store, task.meta["request"]["compile"]["target_id"]
    with monkeypatch.context() as patch:
        patch.setattr(service.viking_fs, "rm", AsyncMock(side_effect=OSError("delete failed")))
        with pytest.raises(OSError, match="delete failed"):
            await service.fs.rm(uri, ctx)
    operations = await store.list_operations(ctx.account_id, target_id)
    failed = next(item for item in operations if item["operation_type"] == "rm")
    assert failed["status"] == "failed"
    assert await service.viking_fs.exists(uri, ctx=ctx)
    entered, proceed = asyncio.Event(), asyncio.Event()
    after_remove = service.fs.compile_versions.after_remove

    async def pause_metadata(*args):
        entered.set()
        await proceed.wait()
        await after_remove(*args)

    monkeypatch.setattr(service.fs.compile_versions, "after_remove", pause_metadata)
    deleting = asyncio.create_task(service.fs.rm(uri, ctx))
    await asyncio.wait_for(entered.wait(), 5)
    agfs = service.viking_fs._async_agfs
    with pytest.raises(LockAcquisitionError):
        await agfs.pathlock_acquire_tree(service.viking_fs._uri_to_path(TO, ctx=ctx))
    deleting.cancel()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await deleting
    operations = await store.list_operations(ctx.account_id, target_id)
    assert any(
        item["operation_type"] == "rm" and item["status"] == "completed" for item in operations
    )
    await service.compile.on_finish(task, ExternalTaskSnapshot(status="failed"))
    assert not await service.viking_fs.exists(uri, ctx=ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("operation", ["cp", "mv"])
async def test_compile_rollback_preserves_file_operations(
    service, compile_task, monkeypatch, operation, status
):
    """Transfer scopes retain their baseline and respect later ordinary edits."""
    # Summary jobs use independent leases; content rollback uses the real filesystem and Git.
    monkeypatch.setattr(service.fs, "_enqueue_copy_refresh", AsyncMock(return_value="queued"))
    monkeypatch.setattr(service.fs, "_enqueue_delete_refresh", AsyncMock())
    monkeypatch.setattr("openviking.service.fs_service.vectorize_directory_meta", AsyncMock())
    ctx, task = compile_task
    source, target = "viking://resources/ordinary", f"{TO}/imported"
    await _write(service, f"{source}/note.md", "ordinary", ctx)
    await _write(service, f"{source}/removed.md", "ordinary", ctx)
    await service.viking_fs.mkdir(f"{source}/empty", ctx=ctx)
    await service.viking_fs.mkdir(target, ctx=ctx)
    await service.compile.on_start(task)
    await _write(service, f"{target}/compile-only.md", "compiled", ctx)
    if operation == "cp":
        await service.fs.cp(source, target, ctx=ctx, recursive=True)
    else:
        await service.fs.mv(source, target, ctx)
        assert not await service.viking_fs.exists(source, ctx=ctx)
    await service.fs.rm(f"{target}/removed.md", ctx, wait=True)
    await service.fs.write(f"{target}/edited.md", "ordinary latest", ctx, wait=True)
    await service.fs.mv(f"{TO}/page.md", f"{TO}/renamed.md", ctx)
    await service.fs.mkdir(f"{TO}/empty", ctx)
    await service.fs.rm(f"{TO}/empty", ctx, recursive=True, wait=True)
    await service.fs.mkdir(f"{TO}/empty", ctx)
    await service.fs.batch_write(
        root_uri=TO,
        operations=[
            {"uri": uri, "content": "compiled"}
            for uri in (
                f"{target}/note.md",
                f"{target}/removed.md",
                f"{target}/edited.md",
                f"{target}/empty/compile-only.md",
                f"{TO}/renamed.md",
                f"{TO}/empty/compile-only.md",
            )
        ],
        ctx=ctx,
        defer_processing=True,
        edit_source="compile",
    )
    await service.compile.on_finish(task, ExternalTaskSnapshot(status=status))
    await service.compile.on_finish(task, ExternalTaskSnapshot(status=status))
    assert await service.viking_fs.read_file(f"{target}/note.md", ctx=ctx) == "ordinary"
    assert await service.viking_fs.read_file(f"{target}/edited.md", ctx=ctx) == "ordinary latest"
    assert await service.viking_fs.read_file(f"{target}/compile-only.md", ctx=ctx) == "compiled"
    assert await service.viking_fs.read_file(f"{TO}/renamed.md", ctx=ctx) == "before"
    for uri in (
        f"{TO}/page.md",
        f"{target}/removed.md",
        f"{target}/empty/compile-only.md",
        f"{TO}/empty/compile-only.md",
    ):
        assert not await service.viking_fs.exists(uri, ctx=ctx)
    for uri in (f"{target}/empty", f"{TO}/empty"):
        assert (await service.viking_fs.stat(uri, ctx=ctx, skip_count=True))["isDir"]
    history = await service.compile.list_operations(TO, ctx, limit=20, cursor=None)
    assert {"mkdir", "rm", operation, "mv"} <= {
        item["operation_type"] for item in history["operations"]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cp", "mv"])
@pytest.mark.parametrize("existing", [False, True])
async def test_partial_transfer_preserves_directory_baseline(
    service, compile_task, monkeypatch, operation, existing
):
    """A partial transfer protects its final directory, including untouched Compile files."""
    ctx, task = compile_task
    source, target = "viking://resources/ordinary", f"{TO}/imported"
    for name in ("a.md", "b.md"):
        await _write(service, f"{source}/{name}", "same", ctx)
    await service.viking_fs.mkdir(target, ctx=ctx)
    if existing:
        await _write(service, f"{target}/a.md", "before", ctx)
    await service.compile.on_start(task)
    await service.fs.batch_write(
        root_uri=TO,
        operations=[
            {"uri": f"{target}/a.md", "content": "same"},
            {"uri": f"{target}/b.md", "content": "untouched compiled"},
        ],
        ctx=ctx,
        defer_processing=True,
        edit_source="compile",
    )
    agfs = service.viking_fs._async_agfs
    copy, ls = agfs.cp, agfs.ls
    copied = []

    async def ordered_entries(*args, **kwargs):
        return sorted(await ls(*args, **kwargs), key=lambda entry: entry.get("name", ""))

    async def fail_second_file(old_path, new_path, **kwargs):
        """Fail the second physical copy only after the first write completes."""
        if old_path.endswith("/b.md"):
            assert copied == ["a.md"]
            raise OSError("transfer interrupted")
        result = await copy(old_path, new_path, **kwargs)
        copied.append(old_path.rsplit("/", 1)[-1])
        return result

    with monkeypatch.context() as patch:
        patch.setattr(agfs, "ls", ordered_entries)
        patch.setattr(agfs, "cp", fail_second_file)
        with pytest.raises(OSError, match="transfer interrupted"):
            await getattr(service.fs, operation)(
                source, target, ctx=ctx, **({"recursive": True} if operation == "cp" else {})
            )
    assert copied == ["a.md"]
    assert await service.viking_fs.read_file(f"{target}/a.md", ctx=ctx) == "same"
    assert await service.viking_fs.read_file(f"{target}/b.md", ctx=ctx) == "untouched compiled"
    await service.fs.batch_write(
        root_uri=TO,
        operations=[
            {"uri": f"{target}/{name}", "content": "compiled later"} for name in ("a.md", "b.md")
        ],
        ctx=ctx,
        defer_processing=True,
        edit_source="compile",
    )
    await service.compile.on_finish(task, ExternalTaskSnapshot(status="failed"))
    assert await service.viking_fs.read_file(f"{target}/a.md", ctx=ctx) == "same"
    assert await service.viking_fs.read_file(f"{target}/b.md", ctx=ctx) == "untouched compiled"
    for name in ("a.md", "b.md"):
        assert await service.viking_fs.read_file(f"{source}/{name}", ctx=ctx) == "same"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [".abstract.md", ".overview.md"])
async def test_copy_derived_file_does_not_block_compile_rollback(service, compile_task, name):
    """Derived files have no ordinary snapshot effect and cannot prevent business rollback."""
    ctx, task = compile_task
    source, target = f"viking://resources/ordinary/{name}", f"{TO}/{name}"
    await _write(service, source, "derived", ctx)
    await service.compile.on_start(task)
    await service.fs.batch_write(
        root_uri=TO,
        operations=[{"uri": f"{TO}/page.md", "content": "compiled"}],
        ctx=ctx,
        defer_processing=True,
        edit_source="compile",
    )
    await service.fs.cp(source, target, recursive=False, ctx=ctx)
    store, target_id = service.compile.lifecycle.store, task.meta["request"]["compile"]["target_id"]
    operation = next(
        item
        for item in await store.list_operations(ctx.account_id, target_id)
        if item["operation_type"] == "cp"
    )
    manifest = await store.get_outputs(ctx.account_id, target_id, operation["operation_id"])
    assert manifest["outputs"] == []
    assert manifest.get("effects", []) == []
    await service.compile.on_finish(task, ExternalTaskSnapshot(status="failed"))
    assert await service.viking_fs.read_file(f"{TO}/page.md", ctx=ctx) == "before"


@pytest.mark.asyncio
async def test_watch_move_and_compile_transfer_guards(service, compile_task):
    ctx, _ = compile_task
    source, target = "viking://resources/watched", "viking://resources/watched_moved"
    await _write(service, f"{source}/note.md", "ordinary", ctx)
    watch = await service.watch_scheduler.watch_manager.create_task(
        path="unused.md", to_uri=source, account_id=ctx.account_id, user_id=ctx.user.user_id
    )
    await service.fs.mv(source, target, ctx)
    assert not await service.viking_fs.exists(source, ctx=ctx)
    assert watch.to_uri == target
    with pytest.raises(ConflictError, match="Watched resources"):
        await service.fs.cp(target, f"{TO}/blocked", recursive=True, ctx=ctx)
    with pytest.raises(ConflictError, match="Watched resources"):
        await service.fs.mv(target, f"{TO}/blocked", ctx)
    assert not await service.viking_fs.exists(f"{TO}/blocked", ctx=ctx)
    assert await service.viking_fs.read_file(f"{target}/note.md", ctx=ctx) == "ordinary"
    await service.fs.cp(target, f"{TO}-sibling", recursive=True, ctx=ctx)
    assert await service.viking_fs.read_file(f"{TO}-sibling/note.md", ctx=ctx) == "ordinary"
    assert watch.to_uri == target


@pytest.mark.asyncio
@pytest.mark.parametrize("running", [False, True])
async def test_compile_destination_guards(service, compile_task, monkeypatch, running):
    ctx, task = compile_task
    alternate = "viking://resources/alternate_source"
    await _write(service, f"{alternate}/note.md", "source", ctx)
    store = service.compile.lifecycle.store
    target_id = task.meta["request"]["compile"]["target_id"]
    async with store.edit(ctx.account_id, target_id) as record:
        record["from"] = [alternate]
        record["schedule_configured"] = True
    if running:
        active = await service.compile.create(
            CompileRequest.model_validate({"from": [alternate], "to": TO, "skill": SKILL}),
            connection={},
            ctx=ctx,
        )
        assert await service.compile.on_start(active) == "proceed"
        assert await service.compile.on_start(task) == "busy"
        with pytest.raises(ConflictError):
            await service.fs.mv(TO + "/", "viking://resources/moved_target", ctx)
    for source in (SRC, alternate):
        for operation in (service.fs.cp, service.fs.mv):
            with pytest.raises((ConflictError, InvalidArgumentError)):
                await operation(
                    source,
                    f"{TO}/blocked",
                    ctx=ctx,
                    **({"recursive": True} if operation == service.fs.cp else {}),
                )
            assert await service.viking_fs.exists(source, ctx=ctx)
            assert not await service.viking_fs.exists(f"{TO}/blocked", ctx=ctx)
    with pytest.raises((ConflictError, InvalidArgumentError)):
        await service.fs.cp(f"{SRC}/a.md", f"{TO}/blocked", ctx=ctx, recursive=False)
    monkeypatch.setattr(
        "openviking.service.resource_service.effective_resource_add_target",
        AsyncMock(return_value=TO),
    )
    for target in (
        {"to": TO},
        {"parent": TO + "/"},
        {"to": f"{TO}/blocked", "add_type": "tos"},
        {},
    ):
        with pytest.raises(ConflictError, match="Compile target"):
            await service.resources.add_resource(
                "tos://bucket/docs/" if target.get("add_type") else "unused.md", ctx, **target
            )
    with pytest.raises(ConflictError, match="Compile target"):
        await service.resources._resource_processor.reserve_unique_candidate(
            candidate_uri=f"{TO}/blocked",
            ctx=ctx,
            target_validator=service.resources._ensure_resource_target_available,
        )
    assert not await service.viking_fs.exists(f"{TO}/blocked", ctx=ctx)
    await service.resources._ensure_resource_target_available(TO + "-sibling", ctx)
    await service.resources._ensure_resource_target_available(
        "viking://resources", ctx, is_parent=True
    )
    sibling, lease = await service.resources._resource_processor.reserve_unique_candidate(
        candidate_uri=TO,
        ctx=ctx,
        target_validator=service.resources._ensure_resource_target_available,
    )
    try:
        assert sibling == TO + "_1"
        await service.viking_fs.rm(sibling, recursive=True, ctx=ctx, lease_ref=lease)
    finally:
        await service.viking_fs._async_agfs.pathlock_release(lease)
    if running:
        await service.compile.on_finish(active, ExternalTaskSnapshot(status="failed"))
    await service.fs.rm(TO, ctx, recursive=True)
    await service.viking_fs.mkdir(TO, ctx=ctx)
    assert await store.register(ctx.account_id, TO) == target_id
    await service.fs.mv(SRC, f"{TO}/imported", ctx=ctx)
    assert await service.viking_fs.read_file(f"{TO}/imported/a.md", ctx=ctx) == "before"


@pytest.mark.parametrize(
    "edits,latest_source",
    [
        ([("manual", "A"), ("compile", "B")], "compile"),
        ([("compile", "B"), ("manual", "A")], "manual"),
        ([("manual", "A"), ("compile", "A")], "manual"),
        ([("manual", "A"), ("compile", "before")], "compile"),
        ([("compile", "B"), ("manual", "before")], "manual"),
    ],
)
async def test_finish_compares_final_content(
    service, compile_task, compile_write, edits, latest_source
):
    ctx, task = compile_task
    await service.compile.on_start(task)
    uri = f"{TO}/page.md"
    for source, content in edits:
        if source == "compile":
            await compile_write(ctx, uri, content)
        else:
            await service.fs.write(uri, content, ctx, edit_source=source, wait=True)
    before = await service.compile.file_versions(uri, ctx, limit=10, before=None)
    assert [v["source"] for v in before["versions"]] == ["manual", "external"]
    updated_at = (await service.compile.get_config(TO, ctx))["content_updated_at"]
    for _ in range(2):
        await service.compile.on_finish(task, ExternalTaskSnapshot(status="completed"))
    final_updated_at = (await service.compile.get_config(TO, ctx))["content_updated_at"]
    if edits == [("manual", "A"), ("compile", "A")]:
        assert final_updated_at == updated_at
    else:
        assert final_updated_at > updated_at
    history = (await service.compile.file_versions(uri, ctx, limit=10, before=None))["versions"]
    assert history[0]["source"] == latest_source
    outputs = await service.compile.operation_outputs(TO, task.task_id, ctx, limit=10, cursor=None)
    assert len(outputs["outputs"]) == (1 if latest_source == "compile" else 0)
    if outputs["outputs"]:
        assert outputs["outputs"][0]["version_id"] == history[0]["version_id"]
    assert history[-2:] == before["versions"]
    assert len(history) == (3 if latest_source == "compile" else 2)
    final = edits[-1][1]
    assert await service.viking_fs.read_file(uri, ctx=ctx) == final
    assert (
        await service.viking_fs.show(history[0]["snapshot_id"], path=uri, ctx=ctx) == final.encode()
    )


@pytest.mark.parametrize("edit_source", ["manual", "agent"])
async def test_write_sources_before_first_compile(
    service, compile_task, compile_write, monkeypatch, edit_source
):
    ctx, task = compile_task
    uri = f"{TO}/nested/new.md"
    commit = AsyncMock(wraps=service.compile.lifecycle.versions._commit)
    monkeypatch.setattr(service.compile.lifecycle.versions, "_commit", commit)
    assert (await service.compile.list_configs(ctx))[0]["content_updated_at"] is None
    await compile_write(ctx, uri, "compiled")
    updated_at = (await service.compile.get_config(TO, ctx))["content_updated_at"]
    assert isinstance(updated_at, float)
    assert (await service.compile.list_configs(ctx))[0]["content_updated_at"] == updated_at
    await compile_write(ctx, uri, "compiled")
    assert (await service.compile.get_config(TO, ctx))["content_updated_at"] == updated_at
    commit.assert_not_awaited()
    assert await service.viking_fs.read_file(uri, ctx=ctx) == "compiled"
    assert (await service.compile.file_versions(uri, ctx, limit=10, before=None))["versions"] == []
    await service.fs.write(uri, "ordinary", ctx, edit_source=edit_source, wait=True)
    commit.assert_awaited_once()
    history = await service.compile.file_versions(uri, ctx, limit=10, before=None)
    assert [v["source"] for v in history["versions"]] == [edit_source]
    operations = (await service.compile.list_operations(TO, ctx, limit=10, cursor=None))[
        "operations"
    ]
    assert [item["operation_type"] for item in operations] == [f"{edit_source}_revise", "initial"]
    await service.compile.on_start(task)
    await service.compile.on_finish(task, ExternalTaskSnapshot(status="completed"))
    assert await service.compile.file_versions(uri, ctx, limit=10, before=None) == history


@pytest.mark.parametrize("status", ["completed", "cancelled"])
async def test_finish_defers_while_ordinary_version_is_registering(
    service, compile_task, monkeypatch, status
):
    from openviking.storage.errors import ResourceBusyError

    ctx, task = compile_task
    await service.compile.on_start(task)
    versions = service.compile.lifecycle.versions
    append = versions.append
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_append(*args, **kwargs):
        entered.set()
        await release.wait()
        return await append(*args, **kwargs)

    monkeypatch.setattr(versions, "append", blocked_append)
    uri = f"{TO}/page.md"
    writing = asyncio.create_task(service.fs.write(uri, "ordinary", ctx, wait=True))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(ResourceBusyError):
            await service.compile.on_finish(task, ExternalTaskSnapshot(status=status))
    finally:
        release.set()
        await writing
    await service.compile.on_finish(task, ExternalTaskSnapshot(status=status))
    history = await service.compile.file_versions(uri, ctx, limit=10, before=None)
    assert [v["source"] for v in history["versions"]] == ["agent", "external"]
    assert await service.viking_fs.read_file(uri, ctx=ctx) == "ordinary"


@pytest.mark.parametrize(
    "interrupt_at,status",
    [
        ("before_versions", "completed"),
        ("after_versions", "completed"),
        ("after_versions", "cancelled"),
    ],
)
async def test_retry_does_not_publish_saved_snapshot_over_new_ordinary_version(
    service, compile_task, compile_write, monkeypatch, interrupt_at, status
):
    from openviking.service.compile_targets import CompileLifecycle

    ctx, task = compile_task
    compile_only = f"{TO}/compile-only.md"
    await _write(service, compile_only, "before", ctx)
    await service.compile.on_start(task)
    uri = f"{TO}/page.md"
    await compile_write(ctx, uri, "compiled")
    await compile_write(ctx, compile_only, "compiled")
    versions = service.compile.lifecycle.versions
    record_compile = versions.record_compile

    async def interrupted(*args, **kwargs):
        if interrupt_at == "after_versions":
            await record_compile(*args, **kwargs)
        raise OSError("interrupted")

    monkeypatch.setattr(versions, "record_compile", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        await service.compile.on_finish(task, ExternalTaskSnapshot(status="completed"))
    await service.fs.write(uri, "ordinary", ctx, edit_source="manual", wait=True)
    restarted = CompileLifecycle(service.compile.lifecycle.store, service.viking_fs)
    await restarted.on_finish(task, ExternalTaskSnapshot(status=status))
    history = await service.compile.file_versions(uri, ctx, limit=10, before=None)
    assert history["versions"][0]["meta"]["after_compile_snapshot"] is True
    expected = ["manual", "external"]
    if interrupt_at == "after_versions" and status == "completed":
        expected.insert(1, "compile")
    assert [v["source"] for v in history["versions"]] == expected
    assert await service.viking_fs.read_file(uri, ctx=ctx) == "ordinary"
    assert await service.viking_fs.read_file(compile_only, ctx=ctx) == (
        "compiled" if status == "completed" else "before"
    )


@pytest.mark.asyncio
async def test_restore_binary_file_version(service, compile_task):
    """Restore PNG bytes from a version retained across Compile publication."""
    ctx, task = compile_task
    uri, original = f"{TO}/diagram.png", b"\x89PNG\r\n\x1a\nv1"
    await service.viking_fs.write_file_bytes(uri, original, ctx=ctx)
    await service.compile.on_start(task)
    await service.fs.batch_write(
        root_uri=TO,
        operations=[
            {"uri": uri, "content_base64": base64.b64encode(original + b" changed").decode()}
        ],
        ctx=ctx,
        defer_processing=True,
        edit_source="compile",
    )
    await service.compile.on_finish(task, ExternalTaskSnapshot(status="completed"))

    restored = await service.compile.restore_file_version(uri, "v1", "v2", ctx)
    await service.resources.wait_processed()
    assert await service.fs.read_file_bytes(uri, ctx) == original
    assert (restored["version_id"], restored["meta"]["restored_from"]) == ("v3", "v1")


@pytest.mark.asyncio
async def test_versions_edit_and_single_file_restore(service, monkeypatch):
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.USER)
    page = f"{TO}/page.md"
    for uri, text in [(f"{SRC}/a.md", "a"), (f"{SKILL}/SKILL.md", "s")]:
        await _write(service, uri, text, ctx)
    await _run(service, ctx, write_page="v1")
    versions = await service.compile.file_versions(page, ctx, limit=20, before=None)
    assert [(v["version_id"], v["source"]) for v in versions["versions"]] == [("v1", "compile")]
    assert versions["versions"][0]["provenance"]["source_roots"] == [SRC]

    await _write(service, f"{SRC}/a.md", "a2", ctx)
    _, changes = await _run(service, ctx, write_page="v2")
    assert set(changes) == {"base", "head", "scopes", "skill_snapshot", "execution"}

    first = await service.compile.file_versions(page, ctx, limit=1, before=None)
    older = await service.compile.file_versions(
        page, ctx, limit=1, before=first["next_before_version"]
    )
    assert first["total"] == older["total"] == 2
    assert [v["version_id"] for v in older["versions"]] == ["v1"]

    expected = ("v3", "manual", b"edited")
    observed = []
    enqueue = ContentWriteCoordinator._enqueue_lock_coordinator

    async def check_version_before_refresh(self, **kwargs):
        """Refresh observes committed content and version."""
        if kwargs["root_uri"] == page:
            history = await service.compile.file_versions(page, ctx, limit=20, before=None)
            latest = history["versions"][0]
            assert (latest["version_id"], latest["source"]) == expected[:2]
            body = await service.viking_fs.show(
                latest["snapshot_id"], path=latest["snapshot_path"], ctx=ctx
            )
            assert body == expected[2]
            observed.append(latest["source"])
        return await enqueue(self, **kwargs)

    monkeypatch.setattr(
        ContentWriteCoordinator, "_enqueue_lock_coordinator", check_version_before_refresh
    )

    await service.fs.write(page, "edited", ctx, edit_source="manual", wait=True)
    versions = await service.compile.file_versions(page, ctx, limit=20, before=None)
    assert [(v["version_id"], v["source"]) for v in versions["versions"]][:2] == [
        ("v3", "manual"),
        ("v2", "compile"),
    ]
    await _write(service, f"{SRC}/a.md", "a3", ctx)
    _, changes = await _run(service, ctx)
    assert set(changes) == {"base", "head", "scopes", "skill_snapshot", "execution"}

    # Single-file restore writes v1 back as a new version; a stale expectation conflicts.
    from openviking_cli.exceptions import ConflictError

    with pytest.raises(ConflictError) as conflict:
        await service.compile.restore_file_version(page, "v1", "v2", ctx)
    assert conflict.value.details["resource"] == "VERSION_CONFLICT"
    assert await service.viking_fs.read_file(page, ctx=ctx) == "edited"
    expected = ("v4", "restore", b"v1")
    restored = await service.compile.restore_file_version(page, "v1", "v3", ctx)
    assert (restored["version_id"], restored["meta"]["restored_from"]) == ("v4", "v1")
    assert (await service.viking_fs.read_file(page, ctx=ctx)).strip() == "v1"
    await service.resources.wait_processed()
    updated_at = (await service.compile.get_config(TO, ctx))["content_updated_at"]

    unchanged = await service.compile.restore_file_version(page, "v1", "v4", ctx)
    assert (unchanged["version_id"], unchanged["meta"]["restored_from"]) == ("v4", None)
    assert unchanged["result"] == "noop"
    assert (await service.compile.get_config(TO, ctx))["content_updated_at"] == updated_at
    await service.resources.wait_processed()
    versions = await service.compile.file_versions(page, ctx, limit=20, before=None)
    assert versions["current_version"] == "v4"
    assert len(versions["versions"]) == 4
    assert observed == ["manual", "restore"]
    operations = (await service.compile.list_operations(TO, ctx, limit=20, cursor=None))[
        "operations"
    ]
    restores = [item for item in operations if item["operation_type"] == "restore"]
    assert [item["status"] for item in restores] == ["completed", "completed", "failed"]
    assert restores[0]["result"]["result"] == "noop"
    assert sum(item["operation_type"] == "manual_revise" for item in operations) == 1
    for operation, count in zip(restores, (0, 1, 0), strict=True):
        outputs = await service.compile.operation_outputs(
            TO, operation["operation_id"], ctx, limit=10, cursor=None
        )
        assert len(outputs["outputs"]) == count
        for item in outputs["outputs"]:
            assert item["version_id"] == "v4"
            assert (
                await service.viking_fs.show(item["snapshot_id"], path=item["uri"], ctx=ctx)
                == b"v1"
            )
    manual = next(item for item in operations if item["operation_type"] == "manual_revise")
    frozen = await service.compile.operation_outputs(
        TO, manual["operation_id"], ctx, limit=10, cursor=None
    )
    monkeypatch.setattr("openviking.service.compile_targets.MAX_VERSIONS", 1)
    monkeypatch.setattr(ContentWriteCoordinator, "_enqueue_lock_coordinator", enqueue)
    await service.fs.write(page, "latest", ctx, wait=True)
    await service.fs.mv(page, f"{TO}/moved.md", ctx)
    assert (
        await service.compile.operation_outputs(
            TO, manual["operation_id"], ctx, limit=10, cursor=None
        )
        == frozen
    )
    (item,) = frozen["outputs"]
    assert await service.viking_fs.show(item["snapshot_id"], path=item["uri"], ctx=ctx) == b"edited"


@pytest.mark.asyncio
async def test_writes_outside_compile_target_do_not_snapshot(service, monkeypatch):
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.USER)
    for uri, text in [(f"{SRC}/a.md", "a"), (f"{SKILL}/SKILL.md", "s")]:
        await _write(service, uri, text, ctx)
    await _run(service, ctx, write_page="v1")
    await _write(service, f"{TO}/.overview.md", "derived", ctx)
    updated_at = (await service.compile.get_config(TO, ctx))["content_updated_at"]

    async def unexpected_snapshot(*args, **kwargs):
        pytest.fail("Only writes inside an initialized Compile target create versions")

    monkeypatch.setattr(service.compile.lifecycle.versions, "_commit", unexpected_snapshot)
    for uri in [f"{SRC}/a.md", f"{TO}_sibling/page.md"]:
        await service.fs.write(uri, "outside target", ctx, wait=True)
        assert await service.viking_fs.read_file(uri, ctx=ctx) == "outside target"
        assert (await service.compile.get_config(TO, ctx))["content_updated_at"] == updated_at
    await service.fs.write(f"{TO}/.overview.md", "derived edit", ctx, wait=True)
    assert (await service.compile.get_config(TO, ctx))["content_updated_at"] == updated_at


@pytest.mark.asyncio
async def test_concurrent_restores_of_different_files_in_one_target(service):
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.USER)
    for uri, text in [(f"{SRC}/a.md", "a"), (f"{SKILL}/SKILL.md", "s")]:
        await _write(service, uri, text, ctx)
    await _run(service, ctx, write_page="untouched")
    pages = [f"{TO}/concurrent-{index}.md" for index in range(5)]
    for index, uri in enumerate(pages):
        await service.fs.write(uri, f"old {index}", ctx, wait=True)
        await service.fs.write(uri, f"new {index}", ctx, wait=True)

    for source, expected, created, body in [
        ("v1", "v2", "v3", "old"),
        ("v2", "v3", "v4", "new"),
    ]:
        results = await asyncio.gather(
            *(service.compile.restore_file_version(uri, source, expected, ctx) for uri in pages),
            return_exceptions=True,
        )
        assert [
            (result["uri"], result["version_id"], result["meta"]["restored_from"])
            for result in results
        ] == [(uri, created, source) for uri in pages]
        await service.resources.wait_processed()
        for index, uri in enumerate(pages):
            content = f"{body} {index}"
            assert await service.viking_fs.read_file(uri, ctx=ctx) == content
            versions = await service.compile.file_versions(uri, ctx, limit=20, before=None)
            assert len(versions["versions"]) == int(created[1:])
            latest = versions["versions"][0]
            assert (latest["version_id"], latest["source"]) == (created, "restore")
            snapshot = await service.viking_fs.show(
                latest["snapshot_id"], path=latest["snapshot_path"], ctx=ctx
            )
            assert snapshot == content.encode()

    assert await service.viking_fs.read_file(f"{TO}/page.md", ctx=ctx) == "untouched"
    untouched = await service.compile.file_versions(f"{TO}/page.md", ctx, limit=20, before=None)
    assert untouched["current_version"] == "v1"


@pytest.mark.asyncio
async def test_moving_and_removing_target_keeps_registry_consistent(service, client):
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.ROOT)
    for uri, text in [(f"{SRC}/a.md", "a"), (f"{SKILL}/SKILL.md", "s")]:
        await _write(service, uri, text, ctx)
    await _run(service, ctx, write_page="v1")
    moved = "viking://resources/compile_moved"
    await service.fs.mv(TO, moved, ctx)
    versions = await service.compile.file_versions(f"{moved}/page.md", ctx, limit=5, before=None)
    assert versions["versions"][0]["snapshot_path"] == f"{TO}/page.md"

    store = service.compile.lifecycle.store
    target_id = (await store.get_by_to(ctx.account_id, moved))["id"]
    before = (await service.compile.get_config(moved, ctx))["content_updated_at"]
    await service.viking_fs.mkdir(TO, ctx=ctx)
    await store.register(ctx.account_id, TO)
    await service.fs.mv(f"{moved}/page.md", f"{TO}/page.md", ctx)
    assert (await service.compile.get_config(moved, ctx))["content_updated_at"] > before
    updated_at = (await service.compile.get_config(TO, ctx))["content_updated_at"]
    assert updated_at > before
    await service.fs.rm(f"{TO}/page.md", ctx)
    assert (await service.compile.get_config(TO, ctx))["content_updated_at"] > updated_at
    await service.resources.wait_processed()
    await service.fs.rm(TO, ctx, recursive=True)
    await service.fs.rm(moved, ctx, recursive=True)
    assert {item["status"] for item in await service.compile.list_configs(ctx)} == {"deleted"}
    await service.resources._ensure_resource_target_available(moved, ctx)
    history = await service.compile.list_operations(moved, ctx, limit=10, cursor=None)
    old_id = history["operations"][-1]["operation_id"]
    assert await store.list_version_indexes(ctx.account_id, target_id) == []
    assert await store.get_outputs(ctx.account_id, target_id, old_id) is None
    await service.viking_fs.mkdir(moved, ctx=ctx)
    assert (await store.get(ctx.account_id, target_id))["status"] == "deleted"
    assert await store.register(ctx.account_id, moved) == target_id
    record = await store.get(ctx.account_id, target_id)
    assert not record["initialized"] and record["progress"]["source_commit"] is None
    assert not record["progress"]["processed_roots"] and record["versions_head"] is None
    assert await service.compile.list_operations(moved, ctx, limit=10, cursor=None) == history
    await _run(service, ctx, to=moved, write_page="rebuilt")
    assert len(await service.compile.list_configs(ctx)) == 2
    latest = (await store.list_operations(ctx.account_id, target_id))[0]["operation_id"]
    outputs = await store.get_outputs(ctx.account_id, target_id, latest)
    assert outputs["outputs"]
    endpoint = f"/api/v1/compile/operations/{old_id}/outputs"
    assert (await client.get(endpoint, params={"to": moved})).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("restore_scope", ["target", "account"])
async def test_operation_history_batch_and_restore(service, client, monkeypatch, restore_scope):
    """A call has one scoped record per target, including noops, failures and restores."""
    ctx = RequestContext(user=UserIdentifier("default", "test_user"), role=Role.ROOT)
    store = service.compile.lifecycle.store
    root = "viking://resources/operation_history"
    targets = [f"{root}/one", f"{root}/two"]
    uris = [f"{to}/page.md" for to in targets]
    for to, uri in zip(targets, uris, strict=True):
        await store.register(ctx.account_id, to)
        await _write(service, uri, "original", ctx)
    snapshot = await service.fs.commit(message="operation history", paths=[root], ctx=ctx)
    writes = [{"uri": uri, "content": "compiled"} for uri in uris]
    await service.fs.batch_write(
        root_uri=root, operations=writes, ctx=ctx, defer_processing=True, edit_source="compile"
    )
    assert (await service.compile.list_operations(targets[0], ctx, limit=10, cursor=None))[
        "operations"
    ] == []
    for write in writes:
        write["content"] = "agent"
    writes.append({"uri": f"{targets[0]}/second.md", "content": "extra"})
    commit = AsyncMock(wraps=service.compile.lifecycle.versions._commit)
    monkeypatch.setattr(service.compile.lifecycle.versions, "_commit", commit)
    await service.fs.batch_write(
        root_uri=root, operations=writes, ctx=ctx, defer_processing=True, edit_source="agent"
    )
    commit.assert_awaited_once()
    pages = [
        await service.compile.list_operations(to, ctx, limit=10, cursor=None) for to in targets
    ]
    for page, uri in zip(pages, uris, strict=True):
        (operation,) = page["operations"]
        assert operation["operation_type"] == "agent_revise"
        assert operation["result"]["updated"] == [uri]
        outputs = await service.compile.operation_outputs(
            uri.rsplit("/", 1)[0], operation["operation_id"], ctx, limit=10, cursor=None
        )
        assert all(
            item["uri"].startswith(uri.rsplit("/", 1)[0] + "/") for item in outputs["outputs"]
        )
        assert all(item["version_id"] == "v1" for item in outputs["outputs"])
        assert len({item["snapshot_id"] for item in outputs["outputs"]}) == 1
    assert pages[0]["operations"][0]["operation_id"] == pages[1]["operations"][0]["operation_id"]
    operation_id = pages[0]["operations"][0]["operation_id"]
    endpoint = f"/api/v1/compile/operations/{operation_id}/outputs"
    response = await client.get(endpoint, params={"to": targets[0], "limit": 1})
    assert response.status_code == 200, response.text
    first_output = response.json()["result"]
    response = await client.get(
        endpoint, params={"to": targets[0], "cursor": first_output["next_cursor"]}
    )
    assert len(response.json()["result"]["outputs"]) == 1
    assert first_output["outputs"][0]["uri"] != response.json()["result"]["outputs"][0]["uri"]
    assert (await client.get(endpoint, params={"to": SRC})).status_code == 404
    from openviking_cli.exceptions import InvalidArgumentError, NotFoundError

    with pytest.raises(InvalidArgumentError):
        await service.compile.operation_outputs(targets[0], "../escape", ctx, limit=10, cursor=None)
    foreign = RequestContext(user=UserIdentifier("foreign", "test_user"), role=Role.ROOT)
    with pytest.raises(NotFoundError):
        await service.compile.operation_outputs(
            targets[0], operation_id, foreign, limit=10, cursor=None
        )

    for _ in range(2):
        await service.fs.write(uris[0], "manual", ctx, edit_source="manual", wait=True)
    updated_at = (await service.compile.get_config(targets[0], ctx))["content_updated_at"]
    first = await client.get("/api/v1/compile/operations", params={"to": targets[0], "limit": 1})
    assert first.status_code == 200, first.text
    page = first.json()["result"]
    assert page["operations"][0]["operation_type"] == "manual_revise"
    next_page = await client.get(
        "/api/v1/compile/operations",
        params={"to": targets[0], "limit": 2, "cursor": page["next_cursor"]},
    )
    assert [item["operation_type"] for item in next_page.json()["result"]["operations"]] == [
        "manual_revise",
        "agent_revise",
    ]
    assert (await client.get("/api/v1/compile/runs", params={"to": targets[0]})).status_code == 404
    monkeypatch.setattr(
        service.viking_fs, "_schedule_restore_reindex_for_paths", AsyncMock(return_value=None)
    )
    restore_to = targets[0] if restore_scope == "target" else None
    await service.fs.restore(
        project_dir=restore_to, source_commit=snapshot["commit_oid"], ctx=ctx, dry_run=True
    )
    await service.fs.restore(project_dir=restore_to, source_commit=snapshot["commit_oid"], ctx=ctx)
    assert await service.viking_fs.read_file(uris[0], ctx=ctx) == "original"
    assert (await service.compile.get_config(targets[0], ctx))["content_updated_at"] > updated_at
    operations = (await service.compile.list_operations(targets[0], ctx, limit=10, cursor=None))[
        "operations"
    ]
    assert len(operations) == 4 and operations[0]["operation_type"] == "restore"
    outputs = await service.compile.operation_outputs(
        targets[0], operations[0]["operation_id"], ctx, limit=10, cursor=None
    )
    (item,) = outputs["outputs"]
    assert item["uri"] == uris[0] and item["version_id"] is not None
    assert (
        await service.viking_fs.show(item["snapshot_id"], path=item["uri"], ctx=ctx) == b"original"
    )
    for failure, status in [
        (RuntimeError("write failed"), "failed"),
        (asyncio.CancelledError(), "cancelled"),
    ]:
        monkeypatch.setattr(ContentWriteCoordinator, "_write", AsyncMock(side_effect=failure))
        with pytest.raises(type(failure)):
            await service.fs.write(uris[0], "unwritten", ctx)
        latest = (await service.compile.list_operations(targets[0], ctx, limit=1, cursor=None))[
            "operations"
        ][0]
        assert latest["operation_type"] == "agent_revise" and latest["status"] == status
        assert latest["finished_at"] is not None and latest["conflicts_num"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("write failed"), asyncio.CancelledError()])
async def test_batch_outputs_preserve_partial_writes(service, compile_task, monkeypatch, failure):
    """A failed batch versions its successful writes once before releasing content locks."""
    ctx, _ = compile_task
    write = ContentWriteCoordinator._write_in_place

    async def fail_second(self, uri, *args, **kwargs):
        if uri.endswith("bad.md"):
            raise failure
        return await write(self, uri, *args, **kwargs)

    monkeypatch.setattr(ContentWriteCoordinator, "_write_in_place", fail_second)
    with pytest.raises(type(failure)):
        await service.fs.batch_write(
            root_uri=TO,
            ctx=ctx,
            edit_source="manual",
            defer_processing=True,
            operations=[
                {"uri": f"{TO}/{name}.md", "content": name.split("_", 1)[1]}
                for name in ("01_good", "02_bad", "03_unwritten")
            ],
        )
    operation = (await service.compile.list_operations(TO, ctx, limit=1, cursor=None))[
        "operations"
    ][0]
    outputs = await service.compile.operation_outputs(
        TO, operation["operation_id"], ctx, limit=10, cursor=None
    )
    (item,) = outputs["outputs"]
    assert item["uri"] == f"{TO}/01_good.md" and item["version_id"] == "v1"
    assert await service.viking_fs.show(item["snapshot_id"], path=item["uri"], ctx=ctx) == b"good"
    store = service.compile.lifecycle.store
    target = await store.get_by_to(ctx.account_id, TO)
    await store.remove(store._outputs_path(ctx.account_id, target["id"], operation["operation_id"]))
    from openviking_cli.exceptions import NotFoundError

    with pytest.raises(NotFoundError, match="not recorded"):
        await service.compile.operation_outputs(
            TO, operation["operation_id"], ctx, limit=10, cursor=None
        )
