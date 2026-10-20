"""Core regressions for account isolation and advisory Compile execution."""

import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from vikingbot.compile import entry, review
from vikingbot.compile.models import CompileLimits, SanitizedCompileRequest
from vikingbot.compile.pipeline_io import JsonModel, ModelCallError, ModelCallLimitError
from vikingbot.compile.plan import Contract, Node, ReviewDecision
from vikingbot.compile.service import BotCompileService
from vikingbot.config.schema import Config
from vikingbot.openviking_mount.ov_server import VikingClient
from vikingbot.providers.base import LLMResponse, ToolCallRequest

from openviking_cli.exceptions import InvalidArgumentError


@pytest.mark.parametrize(
    "mode,account,peak", [("api_key", "b", 2), ("api_key", "a", 1), ("dev", "b", 1)]
)
async def test_target_locks_follow_effective_account(tmp_path, mode, account, peak):
    config = Config(storage_workspace=str(tmp_path))
    config.ov_server.set_effective_auth_mode(mode)
    service = BotCompileService(agent_loop=SimpleNamespace(config=config))
    active = highest = 0

    async def execute(*args):
        nonlocal active, highest
        active += 1
        highest = max(highest, active)
        await asyncio.sleep(0)
        active -= 1

    service._execute_task = execute
    request = SimpleNamespace(to="viking://resources/wiki")
    await asyncio.gather(
        service._run_task("a", request, {"account_id": "a", "user_id": "alice"}),
        service._run_task("b", request, {"account_id": account, "user_id": "bob"}),
    )
    assert highest == peak


@pytest.mark.parametrize("failure", ["model", "record", "cancel"])
async def test_review_failures_preserve_plan_and_cancellation(failure):
    contract = Contract(extract={"instructions": "Copy", "execution": "direct", "output": "files"})
    decision = ReviewDecision(
        action="revise", contract=contract, plan="p.finalize(mapped, into=target)"
    )

    async def ask(*args, **kwargs):
        args[4](decision)
        if failure != "record":
            error = asyncio.CancelledError() if failure == "cancel" else OSError("Review failed")
            raise error
        return decision

    runtime = SimpleNamespace(
        contract=contract,
        model=SimpleNamespace(ask=ask),
        files=SimpleNamespace(put=AsyncMock()),
        resources=SimpleNamespace(name="read_skill_resource"),
        failures=[],
        metrics=Counter(),
    )
    if failure == "record":
        runtime.files.put.side_effect = OSError("Record failed")
    args = (runtime, Node("mapped", "map", "sources", task="extract"), [], "", set(), 0, "", {})
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await review.run(*args)
    else:
        assert await review.run(*args) is None
    assert runtime.contract is contract


def response(name, arguments):
    """Return one scripted reader or final-result tool call."""
    return LLMResponse(None, [ToolCallRequest("call", name, arguments, 0)])


@pytest.mark.parametrize("fatal", [False, True])
async def test_incremental_entry_preserves_provider_failure_semantics(fatal):
    """Scoped readers retain pipeline fallback for limits and propagate provider failures."""
    error = ModelCallError("provider failure") if fatal else ModelCallLimitError("deadline")
    runtime = SimpleNamespace(
        output_instructions="",
        tool_context=None,
        model=SimpleNamespace(ask=AsyncMock(side_effect=error)),
        direct_output_format="files",
    )
    if fatal:
        with pytest.raises(ModelCallError, match="provider failure"):
            await entry.run(runtime, [], {})
    else:
        assert await entry.run(runtime, [], {}) is None
    assert runtime.model.ask.await_args.kwargs["readers"] == {}
    assert runtime.direct_output_format is None


@pytest.mark.parametrize("case", ["repeat", "progress", "error"])
async def test_direct_reads_stop_repetition_but_allow_progress(case):
    offsets = [n // 2 for n in range(102)] if case == "progress" else list(range(4))
    files = SimpleNamespace(put=AsyncMock())
    responses = [response("read", {"offset": n}) for n in offsets]
    responses.append(response("emit", {"action": "continue"}))
    provider = SimpleNamespace(thinking=True, chat=AsyncMock(side_effect=responses))
    model = JsonModel(provider, "test", 0, files, CompileLimits(), {}, Counter())
    reader = SimpleNamespace(
        name="read", description="", parameters={}, execute=AsyncMock(return_value="x")
    )
    if case == "progress":
        reader.execute.side_effect = [f"chunk{n}" for n in offsets]
    if case == "error":
        reader.execute.side_effect = [
            error(f"request_id={i}")
            for i, error in enumerate((OSError, ValueError, TimeoutError, PermissionError))
        ]
    result = model.direct("map", "", {}, ReviewDecision, None, "test", readers={"read": reader})
    if case == "progress":
        assert (await result).action == "continue"
    else:
        with pytest.raises(ModelCallLimitError, match="made no progress|failed repeatedly"):
            await result
    assert provider.chat.await_count == (len(offsets) + 1 if case == "progress" else 4)
    assert provider.chat.await_args.kwargs["thinking"] is False


@pytest.mark.parametrize("case", ["absent", "empty", "null", "plain", "rescan", "delete", "mixed"])
async def test_deleted_sources_keep_valid_input_lineage(case):
    """Deletion updates retain checked IDs; absent changes and rescans invent no deletions."""
    from vikingbot.compile import file_ops
    from vikingbot.compile.ops.common import payload
    from vikingbot.compile.pipeline import Pipeline
    from vikingbot.compile.results import FileDraft, FileResponse, Group
    from vikingbot.compile.sources import split_source

    changes = None if case in ("absent", "plain") else {}
    if case in ("rescan", "delete", "mixed"):
        changes = {
            "scopes": [
                {
                    "uri": "viking://resources/docs/a.md",
                    "change_type": "rescan" if case == "rescan" else "deleted",
                    "recursive": case == "rescan",
                    "change_types": ["deleted"],
                }
            ]
        }
    elif case == "null":
        changes = {"scopes": None}
    stored = {"inputs": {}}
    runtime = Pipeline.__new__(Pipeline)
    runtime.files = SimpleNamespace(
        get=AsyncMock(side_effect=stored.get), put=AsyncMock(side_effect=stored.__setitem__)
    )
    runtime.changes, runtime.limits = changes, CompileLimits()
    runtime.records, runtime.evidence, runtime.status, runtime.metrics = {}, {}, {}, Counter()
    runtime.target = "viking://resources/wiki"

    async def batches():
        if case in ("plain", "mixed"):
            stored["inputs"]["viking://resources/docs/b.md"] = "pending"
            yield split_source("viking://resources/docs/b.md", "Unrelated fact", 100)

    records = await runtime.seed(batches())
    deleted = [r for r in records if runtime.evidence[r.record_id].get("change_type") == "deleted"]
    assert len(deleted) == int(case in ("delete", "mixed"))
    if deleted:
        record = deleted[0]
        assert (await payload(runtime, record))["payload"]["change_type"] == "deleted"
        assert record.source_refs == [record.record_id]
        result = FileResponse(
            files=[FileDraft(path="a.md", content="Remaining facts", inputs=[record.record_id])]
        )
        file_ops.validate_files(runtime, result, Group("test", records), records, {})
        result.files[0].inputs = ["viking://resources/docs/a.md"]
        with pytest.raises(file_ops.InputReferenceError):
            file_ops.validate_files(runtime, result, Group("test", records), records, {})
    assert len(records) == len(deleted) + int(case in ("plain", "mixed"))
    await runtime.write_coverage()
    assert set(stored["coverage"]["inputs"]) == set(stored["inputs"])


@pytest.mark.parametrize(
    "mode,source_files",
    [("full", ["viking://resources/docs/a.md"]), ("full", []), (None, []), ("incremental", [])],
)
async def test_execution_mode_and_snapshot(tmp_path, monkeypatch, mode, source_files):
    from vikingbot.compile import service as module

    root = "viking://agent/skills/wiki"
    snapshot = {"uri": root, "snapshot_id": "a" * 40}
    changes = {"scopes": [], "skill_snapshot": snapshot}
    if mode:
        changes["execution"] = {"mode": mode}
    request = SanitizedCompileRequest(
        from_=["viking://resources/docs"],
        to="viking://resources/wiki",
        skill=root,
        instruction="Change title",
        instruction_provided=True,
        changes=changes,
    )
    config = Config(storage_workspace=str(tmp_path))
    service = BotCompileService(agent_loop=SimpleNamespace(config=config))
    for name in ("_set_state", "_record_usage", "_cleanup_execution_resources"):
        setattr(service, name, AsyncMock())
    service._list_source_files = AsyncMock(wraps=service._list_source_files)
    service._list_changed_files = AsyncMock(return_value=[])
    service._run_pipeline = AsyncMock(return_value=False)
    service.store.update = AsyncMock()
    client = SimpleNamespace(
        read_raw=AsyncMock(return_value="Frozen Skill"),
        stat=AsyncMock(return_value={"isDir": True}),
        list_resources=AsyncMock(return_value=[{"uri": uri} for uri in source_files]),
    )
    create = AsyncMock(return_value=client)
    monkeypatch.setattr(module.VikingClient, "create", create)
    manager = SimpleNamespace(
        get_workspace_path=lambda key: tmp_path / "sandbox",
        get_sandbox=AsyncMock(return_value=object()),
    )
    monkeypatch.setattr(module, "CompileSandboxManager", lambda *args, **kwargs: manager)
    connection = {}
    await service._execute_task("cmp_test", request, connection)
    assert connection == {}
    assert create.await_args.kwargs["connection"]["skill_snapshot"] == snapshot
    if mode == "incremental":
        service._run_pipeline.assert_not_awaited()
        service._list_source_files.assert_not_awaited()
        client.read_raw.assert_not_awaited()
        service.store.update.assert_awaited_once()
    else:
        service._list_source_files.assert_awaited_once_with(client, request.from_, allow_empty=True)
        args = service._run_pipeline.await_args.kwargs
        assert (args["skill_text"], args["source_files"]) == ("Frozen Skill", source_files)
        assert args["tool_context"].openviking_connection["skill_snapshot"] == snapshot


async def test_skill_snapshot_reads_are_fixed_per_client(tmp_path, monkeypatch):
    from vikingbot.openviking_mount import ov_server as module

    root = "viking://agent/skills/wiki"
    config = Config(storage_workspace=str(tmp_path))
    config.ov_server.server_url = "http://localhost:1933"
    config.ov_server.set_effective_auth_mode("dev")

    async def show(ref, *, path):
        if path in {root, f"{root}/references"}:
            raise InvalidArgumentError(f"path is a directory, not a file: {path}")
        await asyncio.sleep(0)
        return {"bytes": f"{ref[0]}\nFrozen Skill\n".encode()}

    sdk = SimpleNamespace(git_show=AsyncMock(side_effect=show))
    sdk.read_raw = AsyncMock(return_value="Live")
    matches = [{"uri": f"{root}/SKILL.md", "content": "Live"}, {"uri": "viking://resources/a.md"}]
    sdk.grep = AsyncMock(return_value={"matches": matches, "count": 2})
    sdk.search = AsyncMock(return_value={"skills": matches, "total": 2})
    monkeypatch.setattr(module.ov, "AsyncHTTPClient", lambda **kwargs: sdk)
    clients = []
    for character in "ab":
        snapshot = {"uri": root, "snapshot_id": character * 40}
        clients.append(VikingClient(config=config, connection={"skill_snapshot": snapshot}))

    async def read_version(client, character):
        expected = f"{character}\nFrozen Skill\n"
        assert await client.read_raw(f"{root}/SKILL.md") == expected
        assert await client.read_content(f"{root}/SKILL.md", level="read", offset=1) == expected[2:]
        assert await client.download_bytes(f"{root}/references/rules.md") == expected.encode()
        assert (await client.stat(f"{root}/references/rules.md"))["size"] == len(expected.encode())
        assert (await client.stat(f"{root}/references"))["isDir"]
        assert (
            await client.read_content(f"{root}/references/clip.mp3", level="overview") == expected
        )
        sdk.git_show.assert_any_await(character * 40, path=f"{root}/references/.overview.md")
        assert await client.read_raw("viking://resources/other.md") == "Live"

    await asyncio.gather(read_version(clients[0], "a"), read_version(clients[1], "b"))
    with pytest.raises(ValueError, match="openviking_multi_read"):
        await clients[0].grep(root, "Live")
    assert (await clients[0].grep("viking://", "Live"))["matches"] == matches[1:]
    result = await clients[0].search("Live")
    assert result["skills"] == matches[1:] and result["total"] == 1
    sdk.git_show.side_effect = FileNotFoundError("Snapshot missing")
    sdk.read_raw.reset_mock()
    with pytest.raises(FileNotFoundError, match="Snapshot missing"):
        await clients[0].read_raw(f"{root}/SKILL.md")
    sdk.read_raw.assert_not_awaited()
