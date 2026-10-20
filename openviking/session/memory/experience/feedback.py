# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Action-level observations and system-owned Experience feedback."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

from openviking.session.archive_store import is_storage_not_found
from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation, ResolvedOperations
from openviking.session.memory.experience.dag import Check, Dag, GeneralNode, clip_evidence_summary
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.utils.config.memory_config import ActionFeedbackConfig

_COUNTS = ("useful_count", "not_useful_count", "unscored_count")


def injected_actions(execution: dict[str, Any]) -> list[dict[str, str]]:
    """Assign one alias to each distinct action actually inserted into the agent context."""
    actions = []
    seen = set()
    for uri, snapshot in sorted(execution.items()):
        for item in snapshot.get("delivered_advice") or []:
            if not isinstance(item, dict):
                continue
            slot, text = item.get("slot_name"), item.get("instruction")
            if (
                not isinstance(slot, str)
                or not slot
                or not isinstance(text, str)
                or not text.strip()
            ):
                continue
            text = text.strip()
            identity = (uri, slot, text)
            if identity in seen:
                continue
            seen.add(identity)
            actions.append(
                {
                    "id": f"A{len(actions) + 1}",
                    "experience_uri": uri,
                    "slot_name": slot,
                    "instruction": text,
                }
            )
    return actions


def normalize_action_scores(value: Any, sources: dict[str, dict[str, str]]) -> list[dict]:
    """Keep best-effort binary assessments; never turn missing evidence into a negative."""
    if isinstance(value, str):
        try:
            value = yaml.safe_load(value)
        except (yaml.YAMLError, RecursionError):
            value = []
    results = {alias: {"id": alias, "evidence": ""} for alias in sources}
    seen = set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        alias = item.get("id")
        if not isinstance(alias, str) or alias not in sources:
            continue
        if alias in seen:
            results[alias] = {"id": alias, "evidence": ""}
            continue
        seen.add(alias)
        evidence = item.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            continue
        result = {"id": alias, "evidence": evidence.strip()}
        score = item.get("score")
        if type(score) in (int, float) and score in (-1, 1):
            result["score"] = int(score)
        results[alias] = result
    return list(results.values())


def _feedback_state(value: Any) -> dict[str, Any]:
    if value is None:
        return {"schema_version": 2, "actions": {}}
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 2
        or not isinstance(value.get("actions"), dict)
    ):
        raise ValueError(
            "Invalid system-owned action_feedback; expected schema_version=2 (slot_name keys)"
        )
    for key, item in value["actions"].items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(item, dict)
            or any(type(item.get(name)) is not int or item[name] < 0 for name in _COUNTS)
            or sum(item[name] for name in _COUNTS) == 0
            or not isinstance(item.get("examples"), dict)
        ):
            raise ValueError("Invalid system-owned action_feedback counters or identity")
        for label, example in item["examples"].items():
            if (
                label not in {"useful", "not_useful", "unscored"}
                or not isinstance(example, dict)
                or not isinstance(example.get("trajectory_uri"), str)
                or not isinstance(example.get("evidence"), str)
            ):
                raise ValueError("Invalid system-owned action_feedback evidence")
    return {"schema_version": 2, "actions": deepcopy(value["actions"])}


def action_feedback_for_content(
    value: Any, content: str, *, config: ActionFeedbackConfig | None = None
) -> dict[str, list[dict[str, str]]]:
    """Group current slots by historical feedback ratios without changing cumulative counts."""
    if value is None:
        return {}
    from openviking.session.memory.experience.compiler import compile_dag

    config = config if config is not None else ActionFeedbackConfig()
    state = _feedback_state(value)
    if not state["actions"]:
        return {}
    dag = Dag.model_validate_json(compile_dag(content))
    groups: dict[str, list[tuple[float, dict[str, str]]]] = {"unchanged": [], "revise": []}
    for node in dag.nodes.values():
        if not isinstance(node, GeneralNode) or isinstance(node.slot_provider, Check):
            continue
        item = state["actions"].get(node.slot_name)
        if item is None:
            continue
        total = item["useful_count"] + item["not_useful_count"]
        if total < config.min_samples:
            continue
        positive = item["useful_count"] / total
        negative = item["not_useful_count"] / total
        # Never issue contradictory instructions if configurable thresholds overlap.
        if negative >= config.revise_min_ratio:
            group, label, polarity, ratio = "revise", "not_useful", "negative", negative
        elif positive >= config.unchanged_min_ratio:
            group, label, polarity, ratio = "unchanged", "useful", "positive", positive
        else:
            continue
        evidence = item["examples"].get(label, {}).get("evidence", "")
        groups[group].append(
            (
                ratio,
                {
                    "slot_name": node.slot_name,
                    "feedback": f"{polarity} {ratio:.1%} ({item[label + '_count']}/{total})",
                    "evidence": clip_evidence_summary(
                        "action_feedback", " ".join(evidence.split()), 160
                    ).replace("\n", " "),
                },
            )
        )
    return {
        group: [
            item
            for _, item in sorted(items, key=lambda pair: (-pair[0], pair[1]["slot_name"]))[
                : config.max_per_group
            ]
        ]
        for group, items in groups.items()
        if items
    }


async def record_experience_feedback_stats(
    trajectory: MemoryFile,
    *,
    registry: Any,
    viking_fs: Any,
    ctx: Any,
) -> dict[str, list[str]]:
    """Accumulate feedback and mark the Trajectory processed under a shared write lock.

    Cached/concurrent calls skip completed Trajectories without an unbounded Experience
    ledger. Counters and the completion marker are separate writes: a retry after a
    partial write can count feedback again. A deleted Experience is never recreated.
    """
    from openviking.session.memory.memory_updater import MemoryUpdater
    from openviking.session.memory.streaming_memory_updater import _acquire_stable_operation_lease

    result: dict[str, list[str]] = {"updated_uris": [], "skipped_uris": []}
    sources = trajectory.extra_fields.get("action_score_sources", {})
    if not isinstance(sources, dict):
        raise ValueError("Invalid persisted action feedback sources")
    if not sources:
        return result
    if not trajectory.uri or MemoryUpdater.memory_type_from_uri(trajectory.uri) != "trajectories":
        raise ValueError("Invalid action feedback Trajectory")
    grouped: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    scores = {
        item["id"]: item
        for item in normalize_action_scores(trajectory.extra_fields.get("action_scores"), sources)
    }
    for alias, source in sources.items():
        if (
            not isinstance(alias, str)
            or not isinstance(source, dict)
            or any(
                not isinstance(source.get(k), str) or not source[k]
                for k in ("experience_uri", "slot_name", "instruction")
            )
            or MemoryUpdater.memory_type_from_uri(source["experience_uri"]) != "experiences"
        ):
            raise ValueError("Invalid persisted action feedback sources")
        # Distinct injected texts are scored separately, then accumulated under the same slot.
        key = (source["slot_name"], source["instruction"])
        actions = grouped.setdefault(source["experience_uri"], {})
        if key in actions:
            raise ValueError("Duplicate action feedback sources")
        actions[key] = {**source, **scores[alias]}
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(memory_type="experiences", uris=[uri], memory_fields={})
            for uri in sorted(grouped)
        ]
        + [ResolvedOperation(memory_type="trajectories", uris=[trajectory.uri], memory_fields={})],
        delete_file_contents=[],
        errors=[],
    )
    lease = await _acquire_stable_operation_lease(operations, viking_fs, ctx)
    if lease is None:
        raise RuntimeError("Experience feedback requires a write-set pathlock")
    try:
        latest = MemoryFileUtils.read(
            await viking_fs.read_file(trajectory.uri, ctx=ctx), uri=trajectory.uri
        )
        update_state = latest.extra_fields.get("experience_update") or {}
        if not isinstance(update_state, dict):
            raise ValueError("Invalid Trajectory experience_update metadata")
        if update_state.get("feedback_recorded") is True:
            result["skipped_uris"] = sorted(grouped)
            return result
        for operation in operations.upsert_operations[:-1]:
            uri = operation.uris[0]
            actions = grouped[uri]
            try:
                raw = await viking_fs.read_file(uri, ctx=ctx)
            except Exception as exc:
                if not is_storage_not_found(exc):
                    raise
                result["skipped_uris"].append(uri)
                continue
            previous = MemoryFileUtils.read(raw, uri=uri)
            state = _feedback_state(previous.extra_fields.get("action_feedback"))
            for action in actions.values():
                item = state["actions"].setdefault(
                    action["slot_name"], {**dict.fromkeys(_COUNTS, 0), "examples": {}}
                )
                label = {1: "useful", -1: "not_useful"}.get(action.get("score"), "unscored")
                item[f"{label}_count"] += 1
                if action["evidence"]:
                    item["examples"][label] = {
                        "trajectory_uri": trajectory.uri,
                        "evidence": clip_evidence_summary(
                            "action_feedback", action["evidence"], 1000
                        ),
                    }
            operation.memory_fields = {"action_feedback": state}
            operation.old_memory_file_content = previous
            updater = MemoryUpdater(registry=registry, transaction_handle=lease)
            updater._viking_fs = viking_fs
            updated = await updater.apply_operations(
                ResolvedOperations(
                    upsert_operations=[operation], delete_file_contents=[], errors=[]
                ),
                ctx,
            )
            if updated.errors:
                raise RuntimeError(f"Experience feedback update failed: {updated.errors}")
            result["updated_uris"].append(uri)
        latest.extra_fields["experience_update"] = {**update_state, "feedback_recorded": True}
        await viking_fs.write_file(
            latest.uri, MemoryFileUtils.write(latest), ctx=ctx, lease_ref=lease
        )
    finally:
        await viking_fs._async_agfs.pathlock_release(lease)
    return result
