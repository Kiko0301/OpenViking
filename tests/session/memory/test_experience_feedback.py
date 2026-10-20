# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from test_content_gate import CTX, SOURCE, URI, Store, batch, file, operation, registry
from test_extraction_output_protocol import _bind, _context

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.experience.feedback import (
    action_feedback_for_content,
    injected_actions,
    normalize_action_scores,
    record_experience_feedback_stats,
)
from openviking.session.memory.extraction_output_protocol import create_extraction_output_protocol
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking.session.memory.patch_merge_context_provider import _hide_system_fields
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.utils.config.memory_config import ActionFeedbackConfig, MemoryConfig

SINGLE_SAMPLE = ActionFeedbackConfig(min_samples=1)


def source(*, slot="reminder", instruction="old reminder", uri=URI):
    return {"experience_uri": uri, "slot_name": slot, "instruction": instruction}


def trajectory(name="t1", *, score=1, sources=None, evidence="Observed action-specific evidence."):
    sources = {"A1": source()} if sources is None else sources
    scores = [
        {"id": alias, "evidence": evidence, **({"score": score} if score is not None else {})}
        for alias in sources
    ]
    return MemoryFile(
        uri=f"viking://user/alice/memories/trajectories/case_a/{name}.md",
        memory_type="trajectories",
        extra_fields={"action_scores": scores, "action_score_sources": sources},
    )


@pytest.fixture
def fs(monkeypatch):
    monkeypatch.setattr(MemoryUpdater, "_vectorize_memories", AsyncMock(return_value=0))
    monkeypatch.setattr(MemoryUpdater, "generate_overview", AsyncMock())
    monkeypatch.setattr(MemoryUpdater, "_sync_resource_refs_for_result", AsyncMock())
    return Store({URI: MemoryFileUtils.write(file())})


async def record(fs, value):
    fs.files.setdefault(value.uri, MemoryFileUtils.write(value))
    return await record_experience_feedback_stats(value, registry=registry(), viking_fs=fs, ctx=CTX)


def stats(fs):
    return MemoryFileUtils.read(fs.files[URI], uri=URI).extra_fields["action_feedback"]


def action_stats(fs, slot="reminder"):
    return stats(fs)["actions"][slot]


def test_aliases_are_per_injected_action_not_experience_or_proposed_node():
    a = {"slot_name": "first", "instruction": "check scope"}
    b = {"slot_name": "second", "instruction": "check payment"}
    assert injected_actions(
        {
            URI: {
                "delivered_advice": [a, b, a],
                "actions": [{"slot_name": "planned", "instruction": "not delivered"}],
                "executed_nodes": ["ack_only"],
                "slot_values": {"ack_only": True},
            }
        }
    ) == [
        {"id": "A1", "experience_uri": URI, **a},
        {"id": "A2", "experience_uri": URI, **b},
    ]
    assert injected_actions({URI: {"actions": [a], "executed_nodes": ["first"]}}) == []


@pytest.mark.parametrize(
    "value",
    [
        "not yaml: [",
        None,
        {},
        [],
        [{"id": "A999", "score": 1, "evidence": "unknown"}],
        [{"id": "A1", "score": True, "evidence": "boolean is not a score"}],
        [{"id": "A1", "score": 0, "evidence": "not allowed"}],
        [{"id": "A1", "score": 1}],
        [{"id": "A1", "score": -2, "evidence": "outside scale"}],
        [
            {"id": "A1", "score": 1, "evidence": "first"},
            {"id": "A1", "score": -1, "evidence": "duplicate"},
        ],
    ],
)
def test_bad_or_missing_scores_remain_unscored_without_blocking(value):
    result = normalize_action_scores(value, {"A1": source()})
    assert len(result) == 1 and result[0]["id"] == "A1"
    assert "score" not in result[0]
    assert "null" not in json.dumps(result)


def test_scores_accept_yaml_binary_values_and_unscored_evidence():
    result = normalize_action_scores(
        """
- id: A1
  score: 1
  evidence: |
    Corrected the selected value after the reminder.
- id: A2
  score: -1
  evidence: The action repeated an already completed check.
- id: A3
  evidence: The suggestion was not followed; usefulness is unknown.
""",
        {"A1": source(), "A2": source(slot="other"), "A3": source(slot="third")},
    )
    assert [x.get("score", "unscored") for x in result] == [1, -1, "unscored"]
    assert "null" not in json.dumps(result)


@pytest.mark.asyncio
async def test_statistics_use_normal_writer_and_are_idempotent(fs, monkeypatch):
    calls = []
    original = MemoryUpdater.apply_operations

    async def apply(updater, operations, ctx, *args, **kwargs):
        assert updater._transaction_handle is not None and fs.lock.locked()
        calls.append(operations)
        return await original(updater, operations, ctx, *args, **kwargs)

    monkeypatch.setattr(MemoryUpdater, "apply_operations", apply)
    before = MemoryFileUtils.read(fs.files[URI], uri=URI)
    assert await record(fs, trajectory()) == {"updated_uris": [URI], "skipped_uris": []}
    persisted = fs.files[URI]
    assert await record(fs, trajectory()) == {"updated_uris": [], "skipped_uris": [URI]}
    after = MemoryFileUtils.read(fs.files[URI], uri=URI)
    assert fs.files[URI] == persisted and len(calls) == 1
    assert set(calls[0].upsert_operations[0].memory_fields) == {"action_feedback"}
    assert action_stats(fs)["useful_count"] == 1
    assert stats(fs)["schema_version"] == 2
    assert "observed_trajectories" not in stats(fs)
    assert (
        MemoryFileUtils.read(fs.files[trajectory().uri]).extra_fields["experience_update"][
            "feedback_recorded"
        ]
        is True
    )
    assert after.content == before.content
    assert after.extra_fields["version"] == before.extra_fields["version"] + 1
    assert not fs.lock.locked()


@pytest.mark.asyncio
async def test_concurrent_new_and_duplicate_feedback_does_not_lose_counts(fs):
    scores = (1, -1, None)
    values = [
        trajectory(
            f"t{i}", score=scores[i % 3], sources={"A1": source(instruction=f"reminder {i}")}
        )
        for i in range(18)
    ]
    await asyncio.gather(*(record(fs, value) for value in values * 2))
    assert "observed_trajectories" not in stats(fs)
    assert all(
        action_stats(fs)[key] == 6 for key in ("useful_count", "not_useful_count", "unscored_count")
    )
    assert not fs.lock.locked()


@pytest.mark.asyncio
async def test_two_actions_in_one_experience_get_independent_scores(fs):
    value = trajectory(
        sources={"A1": source(), "A2": source(slot="other", instruction="different action")}
    )
    value.extra_fields["action_scores"][1]["score"] = -1
    await record(fs, value)
    assert action_stats(fs)["useful_count"] == 1
    assert action_stats(fs)["not_useful_count"] == 0
    assert action_stats(fs, "other")["not_useful_count"] == 1
    assert "observed_trajectories" not in stats(fs)


@pytest.mark.asyncio
async def test_changed_instruction_keeps_cumulative_slot_feedback(fs):
    await record(fs, trajectory("old"))
    new_source = SOURCE.replace("old reminder", "new reminder")
    updater = MemoryUpdater(registry=registry())
    updater._viking_fs = fs
    await updater.apply_operations(batch(operation(new_source)), CTX)
    visible = action_feedback_for_content(stats(fs), new_source, config=SINGLE_SAMPLE)
    assert visible["unchanged"][0]["feedback"] == "positive 100.0% (1/1)"
    assert "instruction" not in json.dumps(visible)
    await record(
        fs, trajectory("new", score=-1, sources={"A1": source(instruction="new reminder")})
    )
    # Late feedback for an earlier instruction still belongs to this slot.
    old = trajectory("late_old")
    await record(fs, old)
    await record(fs, old)
    assert set(stats(fs)["actions"]) == {"reminder"}
    assert "instruction" not in action_stats(fs) and "slot_name" not in action_stats(fs)
    visible = action_feedback_for_content(stats(fs), new_source, config=SINGLE_SAMPLE)
    assert set(visible) == {"revise"}
    assert visible["revise"][0]["feedback"] == "negative 33.3% (1/3)"
    assert action_stats(fs)["useful_count"] == 2 and action_stats(fs)["not_useful_count"] == 1
    assert "instruction" not in json.dumps(visible)
    assert "observed_trajectories" not in json.dumps(visible)


@pytest.mark.asyncio
async def test_renamed_slot_starts_independent_counts_and_hides_removed_slot(fs):
    await record(fs, trajectory("old"))
    renamed = SOURCE.replace("reminder = tell", "renamed = tell").replace(
        "condition.then(reminder)", "condition.then(renamed)"
    )
    assert action_feedback_for_content(stats(fs), renamed, config=SINGLE_SAMPLE) == {}
    await record(fs, trajectory("new", score=-1, sources={"A1": source(slot="renamed")}))
    assert set(stats(fs)["actions"]) == {"reminder", "renamed"}
    visible = action_feedback_for_content(stats(fs), renamed, config=SINGLE_SAMPLE)
    assert set(visible) == {"revise"}
    assert visible["revise"][0]["slot_name"] == "renamed"
    assert visible["revise"][0]["feedback"] == "negative 100.0% (1/1)"


@pytest.mark.asyncio
async def test_same_slot_in_different_experiences_is_isolated(fs):
    other_uri = URI.replace("case_a.md", "case_b.md")
    fs.files[other_uri] = MemoryFileUtils.write(file(uri=other_uri))
    value = trajectory(sources={"A1": source(), "A2": source(uri=other_uri)})
    value.extra_fields["action_scores"][1]["score"] = -1
    await record(fs, value)
    await record(fs, value)
    other = MemoryFileUtils.read(fs.files[other_uri], uri=other_uri).extra_fields["action_feedback"]
    assert action_stats(fs)["useful_count"] == 1
    assert action_stats(fs)["not_useful_count"] == 0
    assert other["actions"]["reminder"]["useful_count"] == 0
    assert other["actions"]["reminder"]["not_useful_count"] == 1


@pytest.mark.asyncio
async def test_multiple_injected_texts_in_one_trajectory_accumulate_by_slot(fs):
    actions = injected_actions(
        {URI: {"delivered_advice": [source(), source(instruction="new reminder"), source()]}}
    )
    assert [a["id"] for a in actions] == ["A1", "A2"]
    value = trajectory(
        sources={a["id"]: {k: v for k, v in a.items() if k != "id"} for a in actions}
    )
    value.extra_fields["action_scores"][1]["score"] = -1
    await record(fs, value)
    await record(fs, value)
    assert set(stats(fs)["actions"]) == {"reminder"}
    assert action_stats(fs)["useful_count"] == action_stats(fs)["not_useful_count"] == 1
    assert "observed_trajectories" not in stats(fs)
    assert len(action_feedback_for_content(stats(fs), SOURCE, config=SINGLE_SAMPLE)["revise"]) == 1


@pytest.mark.asyncio
async def test_unscored_feedback_does_not_count_as_not_useful(fs):
    await record(fs, trajectory(score=None, evidence="Not enough evidence."))
    assert action_stats(fs)["unscored_count"] == 1
    assert action_stats(fs)["useful_count"] == action_stats(fs)["not_useful_count"] == 0


@pytest.mark.asyncio
async def test_feedback_and_dag_update_preserve_each_other(fs):
    updater = MemoryUpdater(registry=registry())
    updater._viking_fs = fs
    result, _ = await asyncio.gather(
        updater.apply_operations(batch(operation(SOURCE.replace("old", "new"))), CTX),
        record(fs, trajectory()),
    )
    assert not result.errors
    assert "new reminder" in MemoryFileUtils.read(fs.files[URI], uri=URI).content
    assert action_stats(fs)["useful_count"] == 1


@pytest.mark.asyncio
async def test_write_failure_is_retryable_without_losing_or_duplicating_feedback(fs, monkeypatch):
    original = fs.write_file
    attempts = 0

    async def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("disk unavailable")
        await original(*args, **kwargs)

    monkeypatch.setattr(fs, "write_file", fail_once)
    with pytest.raises(RuntimeError, match="feedback update failed"):
        await record(fs, trajectory())
    assert not fs.lock.locked()
    await record(fs, trajectory())
    assert action_stats(fs)["useful_count"] == 1


@pytest.mark.asyncio
async def test_missing_experience_is_not_recreated_and_unattributed_history_is_not_backfilled(fs):
    fs.files.clear()
    assert (await record(fs, trajectory()))["skipped_uris"] == [URI]
    assert URI not in fs.files
    assert await record(fs, trajectory(sources={})) == {"updated_uris": [], "skipped_uris": []}


@pytest.mark.asyncio
async def test_corrupt_counters_are_not_reset(fs):
    value = file()
    value.extra_fields["action_feedback"] = {"actions": "bad"}
    fs.files[URI] = MemoryFileUtils.write(value)
    before = fs.files[URI]
    with pytest.raises(ValueError, match="action_feedback"):
        await record(fs, trajectory())
    assert fs.files[URI] == before and not fs.lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["old_schema", "empty_slot", "negative_count", "boolean_count"])
async def test_invalid_slot_feedback_is_rejected_without_erasing_counts(fs, invalid):
    await record(fs, trajectory())
    memory = MemoryFileUtils.read(fs.files[URI], uri=URI)
    state = memory.extra_fields["action_feedback"]
    if invalid == "old_schema":
        state["schema_version"] = 1
    elif invalid == "empty_slot":
        state["actions"][""] = state["actions"].pop("reminder")
    else:
        state["actions"]["reminder"]["useful_count"] = -1 if invalid == "negative_count" else True
    fs.files[URI] = MemoryFileUtils.write(memory)
    before = fs.files[URI]
    with pytest.raises(ValueError, match="action_feedback"):
        await record(fs, trajectory("next"))
    with pytest.raises(ValueError, match="action_feedback"):
        action_feedback_for_content(state, SOURCE)
    assert fs.files[URI] == before and not fs.lock.locked()


def test_system_feedback_is_not_model_writable():
    schema = registry().get("experiences")
    assert "action_feedback" not in {field.name for field in schema.fields}
    value = file()
    value.extra_fields.update(
        action_feedback={"private_marker": 123}, action_score_sources={"A1": source()}
    )
    context = _context([schema], files=[value])
    protocol = create_extraction_output_protocol("python")
    rendered = _bind(protocol, context)
    assert "action_feedback" not in rendered and "private_marker" not in rendered
    assert "action_score_sources" not in rendered
    operations, error = protocol.parse(
        "experiences_1.update(action_feedback='tampered', action_score_sources='tampered')\nsdk.commit()",
        context,
    )
    assert error is None and "action_feedback" not in json.dumps(operations.model_dump())
    assert _hide_system_fields({"action_feedback": 123, "content": "keep"}) == {"content": "keep"}


def test_json_output_cannot_write_system_feedback_metadata():
    context = _context([registry().get("experiences")])
    operations, error = create_extraction_output_protocol("json").parse(
        json.dumps(
            {
                "experiences": [
                    {
                        "page_id": 1,
                        "experience_name": "case_a",
                        "content": SOURCE,
                        "action_feedback": {"useful_count": 999},
                        "action_score_sources": {"A1": source()},
                    }
                ],
                "delete_ids": [],
            }
        ),
        context,
    )
    assert error is None
    result = operations.model_dump()["experiences"][0]
    assert "action_feedback" not in result and "action_score_sources" not in result


@pytest.mark.asyncio
async def test_invalid_saved_sources_cannot_write(fs):
    for sources in (
        {"A1": source(uri=URI.replace("experiences", "records"))},
        [],
        {"A1": source(), "A2": source()},
    ):
        value = trajectory()
        value.extra_fields["action_score_sources"] = sources
        with pytest.raises(ValueError, match="sources"):
            await record(fs, value)
    assert not fs.writes and not fs.lock.locked()


@pytest.mark.asyncio
async def test_partial_write_retry_can_repeat_feedback_without_an_experience_ledger(
    fs, monkeypatch
):
    original = fs.write_file

    async def written_but_ack_lost(*args, **kwargs):
        await original(*args, **kwargs)
        raise OSError("write acknowledgement lost")

    monkeypatch.setattr(fs, "write_file", written_but_ack_lost)
    with pytest.raises(RuntimeError, match="feedback update failed"):
        await record(fs, trajectory())
    persisted = fs.files[URI]
    monkeypatch.setattr(fs, "write_file", original)
    assert (await record(fs, trajectory()))["updated_uris"] == [URI]
    assert fs.files[URI] != persisted and action_stats(fs)["useful_count"] == 2
    assert "observed_trajectories" not in stats(fs)
    # Once the Trajectory marker is saved, normal repeated calls are skipped.
    assert (await record(fs, trajectory()))["skipped_uris"] == [URI]
    assert action_stats(fs)["useful_count"] == 2


@pytest.mark.asyncio
async def test_cancelled_write_releases_lock_and_can_retry(fs, monkeypatch):
    original = fs.write_file
    monkeypatch.setattr(fs, "write_file", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await record(fs, trajectory())
    assert not fs.lock.locked()
    monkeypatch.setattr(fs, "write_file", original)
    await record(fs, trajectory())
    assert action_stats(fs)["useful_count"] == 1


@pytest.mark.asyncio
async def test_policy_snapshot_does_not_overwrite_newer_feedback(fs):
    from openviking.session.train.components.policy_updater import MemoryFilePolicyUpdater
    from openviking.session.train.domain import Policy, PolicyPlanItem, PolicySet, PolicyUpdatePlan

    await record(fs, trajectory("first"))
    before = MemoryFileUtils.read(fs.files[URI], uri=URI)
    snapshot = PolicySet(
        root_uri=URI.rsplit("/", 1)[0],
        policies=[
            Policy(
                name="case_a",
                uri=URI,
                status="production",
                content=before.content,
                version=before.extra_fields["version"],
                metadata=dict(before.extra_fields),
            )
        ],
    )
    await record(fs, trajectory("second"))
    plan = PolicyUpdatePlan(
        items=[
            PolicyPlanItem(
                kind="upsert",
                memory_type="experiences",
                target_name="case_a",
                target_uri=URI,
                before_content=before.content,
                after_content=before.content.replace("old reminder", "new reminder"),
            )
        ]
    )
    result = await MemoryFilePolicyUpdater(viking_fs=fs).apply(plan, snapshot, CTX)
    assert not result.errors
    assert action_stats(fs)["useful_count"] == 2
    assert "observed_trajectories" not in stats(fs)


def test_nested_score_yaml_is_unscored_without_blocking():
    import sys

    depth = sys.getrecursionlimit() + 10
    result = normalize_action_scores("[" * depth + "0" + "]" * depth, {"A1": source()})
    assert result == [{"id": "A1", "evidence": ""}]


@pytest.mark.asyncio
async def test_feedback_view_matches_ask_call_and_tell_actions(fs):
    program = (
        'dag = workflow("review")\n'
        'question = ask("Which record?")\n'
        'lookup = call("get_record", "Use the confirmed identifier")\n'
        'hint = tell("Review the result")\n'
        "question.then(lookup)\nlookup.then(hint)"
    )
    fs.files[URI] = MemoryFileUtils.write(file(source=program))
    value = trajectory(
        sources={
            "A1": source(slot="question", instruction="Ask the user: Which record?"),
            "A2": source(
                slot="lookup", instruction="Call get_record: Use the confirmed identifier"
            ),
            "A3": source(slot="hint", instruction="Review the result"),
        }
    )
    await record(fs, value)
    visible = action_feedback_for_content(stats(fs), program, config=SINGLE_SAMPLE)
    assert [item["slot_name"] for item in visible["unchanged"]] == ["hint", "lookup", "question"]
    assert all(item["feedback"] == "positive 100.0% (1/1)" for item in visible["unchanged"])
    assert "observed_trajectories" not in stats(fs)


def feedback_state(**slots):
    return {
        "schema_version": 2,
        "actions": {
            slot: {
                "useful_count": positive,
                "not_useful_count": negative,
                "unscored_count": unscored,
                "examples": {
                    label: {"trajectory_uri": "internal_uri", "evidence": f"{label} evidence"}
                    for label in ("useful", "not_useful", "unscored")
                },
            }
            for slot, (positive, negative, unscored) in slots.items()
        },
    }


@pytest.mark.parametrize(
    "positive,negative,unscored,group",
    [
        (0, 0, 100, None),
        (9, 0, 100, None),
        (0, 9, 100, None),
        (9, 1, 100, "unchanged"),
        (89, 11, 0, None),
        (7, 3, 100, "revise"),
        (71, 29, 0, None),
        (0, 10, 0, "revise"),
    ],
)
def test_feedback_groups_use_scored_sample_and_ratio_boundaries(
    positive, negative, unscored, group
):
    state = feedback_state(reminder=(positive, negative, unscored))
    before = deepcopy(state)
    groups = action_feedback_for_content(state, SOURCE)
    assert state == before
    if group is None:
        assert groups == {}
    else:
        total = positive + negative
        numerator, polarity = (
            (positive, "positive") if group == "unchanged" else (negative, "negative")
        )
        assert set(groups) == {group}
        assert (
            groups[group][0]["feedback"]
            == f"{polarity} {numerator / total:.1%} ({numerator}/{total})"
        )
        assert groups[group][0]["slot_name"] == "reminder"
        label = "useful" if group == "unchanged" else "not_useful"
        assert groups[group][0]["evidence"] == f"{label} evidence"
    for hidden in (
        "instruction",
        "internal_uri",
        "internal_trajectory",
        "observed_trajectories",
        "unscored",
    ):
        assert hidden not in json.dumps(groups)


def test_feedback_groups_order_by_ratio_and_cap_each_group():
    state = feedback_state(
        keep_low=(90, 10, 0),
        keep_high=(10, 0, 0),
        keep_mid=(99, 1, 0),
        keep_next=(95, 5, 0),
        revise_low=(70, 30, 0),
        revise_high=(0, 10, 0),
        revise_mid=(1, 9, 0),
        revise_next=(2, 8, 0),
        removed=(0, 100, 0),
        condition=(0, 100, 0),
    )
    content = 'dag = workflow("review")\ncondition = check("scope")\n' + "\n".join(
        f'{slot} = tell("instruction")\ncondition.then({slot})'
        for slot in state["actions"]
        if slot not in {"removed", "condition"}
    )
    groups = action_feedback_for_content(state, content)
    assert [item["slot_name"] for item in groups["unchanged"]] == [
        "keep_high",
        "keep_mid",
        "keep_next",
    ]
    assert [item["slot_name"] for item in groups["revise"]] == [
        "revise_high",
        "revise_mid",
        "revise_next",
    ]
    groups = action_feedback_for_content(
        state, content, config=ActionFeedbackConfig(max_per_group=1)
    )
    assert [item["slot_name"] for item in groups["unchanged"]] == ["keep_high"]
    assert [item["slot_name"] for item in groups["revise"]] == ["revise_high"]


def test_feedback_groups_use_custom_thresholds_and_revision_wins_overlap():
    state = feedback_state(reminder=(6, 4, 20))
    config = ActionFeedbackConfig(unchanged_min_ratio=0.6, revise_min_ratio=0.5)
    assert set(action_feedback_for_content(state, SOURCE, config=config)) == {"unchanged"}
    config.revise_min_ratio = 0.4
    assert set(action_feedback_for_content(state, SOURCE, config=config)) == {"revise"}


def test_feedback_evidence_is_one_short_existing_example_or_empty():
    state = feedback_state(reminder=(9, 1, 0))
    state["actions"]["reminder"]["examples"]["useful"]["evidence"] = (
        "First\n" + "detail " * 100 + " last."
    )
    evidence = action_feedback_for_content(state, SOURCE)["unchanged"][0]["evidence"]
    assert len(evidence) <= 160 and "\n" not in evidence
    assert evidence.startswith("First ") and evidence.endswith(" last.")
    del state["actions"]["reminder"]["examples"]["useful"]
    assert action_feedback_for_content(state, SOURCE)["unchanged"][0]["evidence"] == ""


@pytest.mark.parametrize("state", [None, {"schema_version": 2, "actions": {}}])
def test_absent_feedback_omits_groups_without_parsing_dag(state):
    assert action_feedback_for_content(state, "not a DAG") == {}


@pytest.mark.parametrize(
    "values",
    [
        {"min_samples": 0},
        {"min_samples": True},
        {"min_samples": 1.5},
        {"max_per_group": 0},
        {"max_per_group": True},
        {"unchanged_min_ratio": 0},
        {"unchanged_min_ratio": 1.01},
        {"revise_min_ratio": -0.1},
        {"revise_min_ratio": float("nan")},
        {"revise_min_ratio": float("inf")},
    ],
)
def test_feedback_display_config_rejects_invalid_limits(values):
    with pytest.raises(ValueError):
        MemoryConfig.from_dict({"action_feedback": values})


def test_feedback_display_config_round_trip():
    values = {
        "min_samples": 15,
        "unchanged_min_ratio": 0.95,
        "revise_min_ratio": 0.4,
        "max_per_group": 2,
    }
    config = MemoryConfig.from_dict({"action_feedback": values})
    assert config.action_feedback.model_dump() == values
    assert config.to_dict()["action_feedback"] == values


@pytest.mark.asyncio
async def test_normal_experience_update_drops_old_ledgers_without_resetting_feedback(fs):
    await record(fs, trajectory())
    previous = MemoryFileUtils.read(fs.files[URI], uri=URI)
    previous.extra_fields["source_sessions"] = [{"source_session_uri": "old-session"}]
    previous.extra_fields["action_feedback"]["observed_trajectories"] = ["old-trajectory"]
    previous.extra_fields["success_trajectory_refs"] = {"task": "../trajectories/case_a/old.md"}
    business_link = {
        "from_uri": URI,
        "to_uri": URI.replace("experiences", "records"),
        "link_type": "related_to",
    }
    previous.links = [
        business_link,
        {"from_uri": URI, "to_uri": trajectory().uri, "link_type": "derived_from"},
        {"from_uri": URI, "to_uri": "viking://session/old/archives/1", "link_type": "derived_from"},
    ]
    # Seed the old on-disk shape directly; the normal serializer now strips these fields.
    old_fields = previous.to_metadata()
    old_content = old_fields.pop("content")
    fs.files[URI] = old_content + "\n\n<!-- MEMORY_FIELDS\n" + json.dumps(old_fields) + "\n-->"
    trajectory_before = fs.files[trajectory().uri]
    op = operation(SOURCE.replace("old reminder", "new reminder"))
    op.memory_fields["source_sessions"] = [{"source_session_uri": "incoming-session"}]
    updater = MemoryUpdater(registry=registry())
    updater._viking_fs = fs

    result = await updater.apply_operations(batch(op), CTX)

    assert not result.errors
    stored = MemoryFileUtils.read(fs.files[URI], uri=URI)
    assert "source_sessions" not in stored.extra_fields
    assert "success_trajectory_refs" not in stored.extra_fields
    assert [(link["to_uri"], link["link_type"]) for link in stored.links] == [
        (business_link["to_uri"], "related_to")
    ]
    assert fs.files[trajectory().uri] == trajectory_before
    assert "observed_trajectories" not in stored.extra_fields["action_feedback"]
    assert action_stats(fs)["useful_count"] == 1
    assert "new reminder" in stored.content


@pytest.mark.asyncio
async def test_lost_trajectory_marker_ack_skips_feedback_on_retry(fs, monkeypatch):
    original = fs.write_file

    async def lose_marker_ack(uri, *args, **kwargs):
        await original(uri, *args, **kwargs)
        if uri == trajectory().uri:
            raise OSError("marker acknowledgement lost")

    monkeypatch.setattr(fs, "write_file", lose_marker_ack)
    with pytest.raises(OSError, match="marker acknowledgement lost"):
        await record(fs, trajectory())
    assert not fs.lock.locked()
    monkeypatch.setattr(fs, "write_file", original)

    assert (await record(fs, trajectory()))["skipped_uris"] == [URI]
    assert action_stats(fs)["useful_count"] == 1


@pytest.mark.asyncio
async def test_experience_write_drops_incoming_history_index_and_only_source_links(fs):
    op = operation(SOURCE)
    op.memory_fields["success_trajectory_refs"] = {"task": "../trajectories/case_a/t.md"}
    op.memory_fields["links"] = [
        {"from_uri": URI, "to_uri": trajectory().uri, "link_type": "derived_from"},
        {"from_uri": URI, "to_uri": "viking://session/s/archives/1", "link_type": "derived_from"},
        {
            "from_uri": URI,
            "to_uri": URI.replace("experiences", "records"),
            "link_type": "derived_from",
        },
    ]
    updater = MemoryUpdater(registry=registry())
    updater._viking_fs = fs
    result = await updater.apply_operations(batch(op), CTX)
    assert not result.errors
    stored = MemoryFileUtils.read(fs.files[URI], uri=URI)
    assert "success_trajectory_refs" not in stored.extra_fields
    assert [(link["to_uri"], link["link_type"]) for link in stored.links] == [
        (URI.replace("experiences", "records"), "derived_from")
    ]
