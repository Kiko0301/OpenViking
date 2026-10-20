from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.session.memory.dataclass import MemoryFile, StoredLink
from openviking.session.memory.memory_type_registry import get_default_registry
from openviking.session.memory.memory_updater import write_stored_links
from openviking.session.memory.merge_op.link_merge import is_persisted_link
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.train.components.trajectory_observation import trajectory_source_session_uri

ROOT = "viking://user/u/memories"
CASE = f"{ROOT}/cases/case_a.md"
EXP = f"{ROOT}/experiences/case_a.md"
TRAJ = f"{ROOT}/trajectories/case_a/run.md"
SESSION = "viking://user/u/sessions/s/history/archive_001"
OTHER = f"{ROOT}/events/one.md"


def link(source, target, kind="related_to"):
    return StoredLink(from_uri=source, to_uri=target, link_type=kind).model_dump()


@pytest.mark.parametrize(
    "source,target,kind,allowed",
    [
        (CASE, EXP, "related_to", False),
        (EXP, CASE, "related_to", False),
        (TRAJ, CASE, "related_to", False),
        (CASE, TRAJ, "related_to", False),
        (OTHER, CASE, "related_to", False),
        (CASE, OTHER, "related_to", False),
        (EXP, TRAJ, "derived_from", False),
        (TRAJ, EXP, "related_to", False),
        (OTHER, TRAJ, "related_to", False),
        (TRAJ, OTHER, "derived_from", False),
        (TRAJ, SESSION, "derived_from", True),
        (TRAJ, SESSION, "related_to", False),
        (EXP, SESSION, "derived_from", False),
        (EXP, OTHER, "derived_from", True),
        (OTHER, EXP, "related_to", True),
    ],
)
def test_relation_policy(source, target, kind, allowed):
    assert is_persisted_link(link(source, target, kind)) is allowed


@pytest.mark.parametrize(
    "uri,memory_type", [(CASE, "cases"), (TRAJ, "trajectories"), (EXP, "experiences")]
)
def test_serialization_cleans_legacy_relationships_without_mutating_input(uri, memory_type):
    fields = {
        "memory_type": memory_type,
        "links": [
            link(uri, CASE),
            link(uri, TRAJ, "derived_from"),
            link(uri, SESSION, "derived_from"),
        ],
        "backlinks": [link(CASE, uri), link(TRAJ, uri), link(OTHER, uri)],
        "case_name": "case_a",
        "source_sessions": [{"source_session_uri": SESSION}],
        "success_trajectory_refs": {"task": "../trajectories/case_a/run.md"},
        "experience_update": {"state": "applied", "experience_uri": EXP, "feedback_recorded": True},
    }
    # A literal old file bypasses the new write policy so this exercises cleanup on rewrite.
    raw = "unchanged body\n\n<!-- MEMORY_FIELDS\n" + json.dumps(fields) + "\n-->"
    previous = MemoryFileUtils.read(raw, uri=uri)
    before = previous.model_dump()
    result = MemoryFileUtils.read(MemoryFileUtils.write(previous), uri=uri)
    assert previous.model_dump() == before
    assert result.content == previous.content
    if memory_type == "trajectories":
        assert len(result.links) == 1
        assert trajectory_source_session_uri(result) == SESSION
        assert result.backlinks == []
        assert result.extra_fields["experience_update"] == {
            "state": "applied",
            "feedback_recorded": True,
        }
    elif memory_type == "cases":
        assert result.links == [] and result.backlinks == []
    else:
        assert result.links == []
        assert [value["from_uri"] for value in result.backlinks] == [OTHER]
        assert result.extra_fields["case_name"] == "case_a"
        assert not {"source_sessions", "success_trajectory_refs"} & result.extra_fields.keys()


def test_case_template_does_not_render_relationship_section():
    case = MemoryFile(
        uri=CASE,
        memory_type="cases",
        extra_fields={
            "case_name": "case_a",
            "domain": "scope",
            "description": "rules",
            "rubric": "[]",
        },
        links=[link(CASE, EXP)],
    )
    raw = MemoryFileUtils.write(
        case, content_template=get_default_registry().get("cases").content_template
    )
    assert "Linked Experiences" not in raw and EXP not in raw
    assert "## Description\nrules" in raw
    assert MemoryFileUtils.read(raw).links == []


@pytest.mark.asyncio
async def test_direct_relation_writer_skips_case_endpoints_entirely():
    fs = SimpleNamespace(read_file=AsyncMock(), write_file=AsyncMock())
    result = await write_stored_links(
        [StoredLink.model_validate(link(CASE, EXP)), StoredLink.model_validate(link(TRAJ, CASE))],
        None,
        fs,
    )
    assert result == []
    fs.read_file.assert_not_awaited()
    fs.write_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_relation_writer_preserves_trajectory_source_without_touching_archive():
    before = MemoryFile(uri=TRAJ, memory_type="trajectories", content="immutable evidence")
    fs = SimpleNamespace(
        read_file=AsyncMock(return_value=MemoryFileUtils.write(before)), write_file=AsyncMock()
    )
    result = await write_stored_links(
        [StoredLink.model_validate(link(TRAJ, SESSION, "derived_from"))],
        None,
        fs,
        lease_ref="lease",
    )
    assert result == [TRAJ]
    fs.read_file.assert_awaited_once_with(TRAJ, ctx=None)
    saved = MemoryFileUtils.read(fs.write_file.await_args.args[1], uri=TRAJ)
    assert saved.content == before.content
    assert trajectory_source_session_uri(saved) == SESSION
    assert saved.backlinks == []
