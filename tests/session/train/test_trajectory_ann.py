# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_case_experience_pipeline import observe
from test_case_experience_pipeline import setup_observation as setup_observation
from test_gradient_estimator_component import analysis
from test_success_trajectory_references import ann_result, make_plan, make_run

from openviking.session.memory.dataclass import MemoryFile, StoredLink
from openviking.session.memory.memory_type_registry import get_default_registry
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.train.components.trajectory_lifecycle import archive_trajectory
from openviking.session.train.components.trajectory_observation import (
    TRAJECTORY_ANN_QUERY_TOKENS,
    TRAJECTORY_FORMAT,
    find_successful_trajectory,
)
from openviking.session.train.domain import Trajectory
from openviking.utils.tags import build_search_tags_filter
from openviking.utils.token_estimation import estimate_text_tokens

DIRECTORY = "viking://user/u/memories/trajectories/case_a/archive"
QUERY_URI = DIRECTORY + "/2026/10/19/F_query.md"
SUCCESS_URI = DIRECTORY + "/2026/10/18/S_old.md"


def success_file():
    return MemoryFile(
        uri=SUCCESS_URI,
        memory_type="trajectories",
        content="Reference steps.",
        extra_fields={
            "case_name": "case_a",
            "outcome": "success",
            "trajectory_format": TRAJECTORY_FORMAT,
            "trajectory_name": "old",
            "experience_update": {"state": "applied"},
        },
        links=[
            StoredLink(
                from_uri=SUCCESS_URI,
                to_uri="viking://session/old/history/1",
                link_type="derived_from",
            ).model_dump()
        ],
    )


def current(text):
    return Trajectory(
        name="query",
        uri=QUERY_URI,
        outcome="failure",
        content="Management header not for retrieval",
        metadata={"steps": text},
    )


def fs_with(*uris):
    return SimpleNamespace(
        find=AsyncMock(
            return_value=SimpleNamespace(
                memories=[SimpleNamespace(uri=uri, score=0.9) for uri in uris]
            )
        ),
        read_file=AsyncMock(return_value=MemoryFileUtils.write(success_file())),
        ls=AsyncMock(side_effect=AssertionError("ANN must not fall back to a directory scan")),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["long ASCII facts ", "很长的中文轨迹事实。", "😀中文andEnglish "])
async def test_query_is_bounded_and_keeps_head_tail_not_management_header(unit):
    fs = fs_with(SUCCESS_URI)
    trajectory = current("BEGIN QUERY\n" + unit * 20000 + "\nEND QUERY")
    ctx = object()
    result = await find_successful_trajectory(
        fs,
        ctx,
        trajectory=trajectory,
        directory=DIRECTORY,
        case_name="case_a",
        exclude_uris={QUERY_URI},
    )
    assert result.uri == SUCCESS_URI
    call = fs.find.await_args.kwargs
    query = call["query"]
    assert query.startswith("BEGIN QUERY") and query.endswith("END QUERY")
    assert estimate_text_tokens(query) <= TRAJECTORY_ANN_QUERY_TOKENS
    assert "Management header" not in query
    assert call["ctx"] is ctx and call["target_uri"] == DIRECTORY and call["level"] == [2]
    assert call["filter"]["conds"] == [
        build_search_tags_filter(["trajectory_outcome=success"]),
        {"op": "must_not", "field": "uri", "conds": [QUERY_URI]},
    ]
    assert call["limit"] == 8
    fs.ls.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_steps_do_not_degenerate_into_filter_only_lookup():
    fs = fs_with(SUCCESS_URI)
    assert (
        await find_successful_trajectory(
            fs,
            None,
            trajectory=current(" \n"),
            directory=DIRECTORY,
            case_name="case_a",
            exclude_uris=set(),
        )
        is None
    )
    fs.find.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uri",
    [
        "viking://user/other/memories/trajectories/case_a/S_old.md",
        "viking://user/u/memories/trajectories/case_b/S_old.md",
        DIRECTORY + "_other/S_old.md",
        DIRECTORY + "/../case_b/S_old.md",
        DIRECTORY + "/%2e%2e/case_b/S_old.md",
        DIRECTORY + "/.hidden/S_old.md",
        DIRECTORY + "/.overview.md",
        QUERY_URI,
    ],
)
async def test_untrusted_ann_hit_cannot_escape_scope_or_read_current_batch(uri):
    fs = fs_with(uri)
    result = await find_successful_trajectory(
        fs,
        None,
        trajectory=current("Current facts"),
        directory=DIRECTORY,
        case_name="case_a",
        exclude_uris={QUERY_URI},
    )
    assert result is None
    fs.read_file.assert_not_awaited()
    fs.ls.assert_not_awaited()


@pytest.mark.asyncio
async def test_candidate_reads_are_bounded_and_missing_hits_never_trigger_scan():
    fs = fs_with(*[DIRECTORY + f"/gone{i}.md" for i in range(20)], SUCCESS_URI)
    fs.read_file.side_effect = FileNotFoundError("gone")
    result = await find_successful_trajectory(
        fs,
        None,
        trajectory=current("Current facts"),
        directory=DIRECTORY,
        case_name="case_a",
        exclude_uris=set(),
    )
    assert result is None and fs.read_file.await_count == 8
    fs.ls.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["failure", "wrong_case", "pending", "no_source"])
async def test_file_state_is_verified_instead_of_trusting_success_index_tag(defect):
    memory = success_file()
    if defect == "failure":
        memory.extra_fields["outcome"] = "failure"
    if defect == "wrong_case":
        memory.extra_fields["case_name"] = "case_b"
    if defect == "pending":
        memory.extra_fields["experience_update"]["state"] = "pending"
    if defect == "no_source":
        memory.links = []
    fs = fs_with(SUCCESS_URI)
    fs.read_file.return_value = MemoryFileUtils.write(memory)
    result = await find_successful_trajectory(
        fs,
        None,
        trajectory=current("Current facts"),
        directory=DIRECTORY,
        case_name="case_a",
        exclude_uris=set(),
    )
    assert result is None
    fs.ls.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiple_queries_are_independent_sequential_and_deduplicated(setup_observation):
    setup = setup_observation
    fs = setup[0]
    history, _ = await make_run(setup, task="different-task", session="history", passed=True)
    first, a = await make_run(setup, task="a", session="a")
    second, b = await make_run(setup, task="b", session="b")
    first.trajectories[0].metadata["steps"] = "Query for trajectory A"
    second.trajectories[0].metadata["steps"] = "Query for trajectory B"
    active = 0
    queries = []

    async def find(**kwargs):
        nonlocal active
        import asyncio

        active += 1
        assert active == 1
        queries.append(kwargs["query"])
        await asyncio.sleep(0)
        active -= 1
        return ann_result(history)

    fs.find.side_effect = find
    fs.ls = AsyncMock(side_effect=AssertionError("No scan"))
    _plan, selected = await make_plan(setup, [*a, *b], setup[-1])
    assert queries == ["Query for trajectory A", "Query for trajectory B"]
    assert [t.uri for t in selected.trajectories] == [
        first.trajectories[0].uri,
        second.trajectories[0].uri,
        history.trajectories[0].uri,
    ]
    fs.ls.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "passed,outcome", [(True, "success"), (False, "failure"), (None, "unknown")]
)
async def test_outcome_tag_is_enqueued_with_first_trajectory_vector(
    setup_observation, passed, outcome
):
    fs, _, _, analyzer, context, policies = setup_observation
    database = SimpleNamespace(enqueue_embedding_msg=AsyncMock(return_value=True))
    analyzer.vikingdb = database
    run = analysis(passed=passed)
    await observe(analyzer, run, policies, context)
    database.enqueue_embedding_msg.assert_not_awaited()
    memory = run.metadata["trajectory_memory"]
    memory.extra_fields["experience_update"]["state"] = "applied"
    fs.files[memory.uri] = MemoryFileUtils.write(memory)
    archived = await archive_trajectory(
        memory,
        viking_fs=fs,
        ctx=context.request_context,
        registry=get_default_registry(),
        vikingdb=database,
    )
    database.enqueue_embedding_msg.assert_awaited_once()
    message = database.enqueue_embedding_msg.await_args.args[0]
    assert message.context_data["uri"] == archived.uri
    assert message.context_data["search_tags"] == ["trajectory_outcome=" + outcome]
    assert message.context_data["_upsert_options"]["search_tag_mode"] == "append"
    assert run.trajectories[0].outcome == outcome
