"""Core regressions for account isolation and advisory Compile execution."""

import asyncio
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from vikingbot.agent.tools.base import ToolContext
from vikingbot.agent.tools.ov_file import VikingDiffPathsTool, VikingMultiReadTool
from vikingbot.compile import entry, review
from vikingbot.compile.models import CompileLimits
from vikingbot.compile.pipeline import Pipeline
from vikingbot.compile.pipeline_io import JsonModel, ModelCallLimitError
from vikingbot.compile.plan import Contract, Node, ReviewDecision
from vikingbot.compile.sandbox import CompileSandboxManager, _CompileSandbox
from vikingbot.compile.service import BotCompileService
from vikingbot.config.schema import Config, SessionKey
from vikingbot.providers.base import LLMResponse, ToolCallRequest


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
    validate = AsyncMock()
    result = model.direct("map", "", {}, ReviewDecision, validate, "test", readers={"read": reader})
    if case == "progress":
        assert (await result).action == "continue"
        validate.assert_awaited_once()
    else:
        with pytest.raises(ModelCallLimitError, match="made no progress|failed repeatedly"):
            await result
    assert provider.chat.await_count == (len(offsets) + 1 if case == "progress" else 4)
    assert provider.chat.await_args.kwargs["thinking"] is False


def test_compile_workspace_root_is_absolute_for_relative_storage(tmp_path, monkeypatch):
    """A relative storage.workspace must not reach sandbox subprocess paths (#6376)."""
    state_root = tmp_path / "state"
    state_root.mkdir()
    monkeypatch.chdir(state_root)
    config = Config(storage_workspace="data")
    config.ov_server.set_effective_auth_mode("dev")
    manager = CompileSandboxManager(
        config,
        config.bot_data_path / "compile_workspaces" / "task-1",
        config.workspace_path,
        connection={},
    )
    expected = (state_root / "data" / "bot" / "compile_workspaces" / "task-1").resolve()
    assert manager.workspace == expected

    # The credentials path is injected into sandbox shell commands, whose working
    # directory is the workspace itself; a relative path would resolve twice.
    workspace = manager.get_workspace_path(
        SessionKey(type="compile", channel_id="task-1", chat_id="task-1")
    )
    sandbox = _CompileSandbox(SimpleNamespace(sandbox_cwd=str(workspace)))
    config_path = Path(sandbox._config_path)
    assert config_path.is_absolute()
    assert config_path.is_relative_to(manager.workspace)


@pytest.mark.parametrize(
    ("publish", "skill_target"), [(True, False), (False, False), (False, True)]
)
async def test_compile_agent_reads_evidence_and_accepts_noop(tmp_path, publish, skill_target):
    root = "viking://resources/source"
    page = {
        "changes": [{"uri": f"{root}/gone.md", "change_type": "deleted"}],
        "has_more": True,
        "next_offset": 10,
    }
    client = SimpleNamespace(
        diff_paths=AsyncMock(return_value=page),
        read_raw=AsyncMock(return_value="Context\nFact\n"),
    )
    request = SimpleNamespace(
        from_=[root],
        to="viking://resources/output",
        skill="viking://user/default/skills/test",
        changes={"base": "old", "head": "new", "diff_dirs": [root], "full_dirs": []},
    )
    runtime = Pipeline(
        client=client,
        sandbox=SimpleNamespace(sandbox_cwd=str(tmp_path)),
        provider=SimpleNamespace(),
        model="test",
        temperature=0,
        limits=CompileLimits(),
        request=request,
        skill="Test skill",
        usage={},
    )
    stored = {}

    async def put(path, value):
        stored[path] = value

    runtime.files = SimpleNamespace(
        get=AsyncMock(side_effect=stored.get), put=AsyncMock(side_effect=put)
    )
    runtime.tool_context = ToolContext()
    runtime.tools = {tool.name: tool for tool in (VikingDiffPathsTool(), VikingMultiReadTool())}
    runtime.skill_target = skill_target
    runtime.old["summary.md"] = None

    async def ask(stage, system, data, schema, validate, *, readers):
        assert data["inputs"] == []
        client.read_raw.assert_not_awaited()
        result = json.loads(
            await readers["openviking_diff_paths"].execute(
                paths=[root], base="old", head="new", offset=9, limit=1
            )
        )
        assert {key: result[key] for key in page} == page
        assert result["inputs"][0]["payload"]["change_type"] == "deleted"
        material = json.loads(
            await readers["openviking_multi_read"].execute(uris=[f"{root}/a.md"], offset=1, limit=1)
        )
        source = material["inputs"][0]
        assert source["payload"]["start_line"] == 2
        files = [{"path": "summary.md", "content": "Fact", "inputs": [source["id"]]}]
        response = schema(files=files if publish else [], no_changes_reason="Already covered.")
        await validate(response)
        return response

    runtime.model.ask = AsyncMock(side_effect=ask)
    assert len(await entry.run(runtime, [], {})) == int(publish)
    assert runtime.noop is not publish
    client.diff_paths.assert_awaited_once_with([root], base="old", head="new", offset=9, limit=1)
    client.read_raw.assert_awaited_once_with(f"{root}/a.md")
