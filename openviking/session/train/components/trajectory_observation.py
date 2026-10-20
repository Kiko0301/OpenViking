"""Shared Trajectory evidence, provenance and pending Experience-learning signals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from openviking.core.namespace import is_session_uri
from openviking.session.memory.dataclass import (
    MemoryFile,
    StoredLink,
)
from openviking.session.memory.experience.dag import clip_evidence_summary, tool_evidence_summary
from openviking.session.train.domain import RolloutAnalysis, RubricEvaluation, Trajectory
from openviking.telemetry import tracer
from openviking.utils.path_safety import normalize_storage_target_uri
from openviking.utils.tags import build_search_tags_filter
from openviking.utils.token_estimation import truncate_text_to_token_budget
from openviking_cli.exceptions import InvalidArgumentError

MAX_EVIDENCE_CHARS = 64_000
EXPERIENCE_EXECUTION_EVIDENCE_ID = "experience_execution"
TRAJECTORY_FORMAT = "ordered_markdown_v3"
TRAJECTORY_OUTCOME_TAG = "trajectory_outcome"
TRAJECTORY_ANN_QUERY_TOKENS = 2048
_TRAJECTORY_ANN_CANDIDATES = 8


def execution_paths(execution: dict) -> dict:
    return {
        snapshot.get("experience_name") or uri.rsplit("/", 1)[-1].removesuffix(".md"): {
            key: snapshot.get(key, {} if key == "slot_values" else [])
            for key in ("slot_values", "executed_nodes", "current_nodes")
        }
        for uri, snapshot in execution.items()
    }


def observation_evidence(messages, evaluation, execution: dict | None = None) -> list[dict]:
    evidence = []
    for index, message in enumerate(messages):
        content = str(message.content or "")
        if (
            content
            and not content.startswith(
                ("system:", "# OpenViking", "task_success:", "[Rollout Evaluation]")
            )
            and message.role != "system"
        ):
            evidence.append(
                {
                    "id": f"message:{index}",
                    "kind": (
                        "experience_reminder"
                        if content.startswith("[Experience Reminder]")
                        else f"{message.role}_message"
                    ),
                    "summary": clip_evidence_summary("text", content, 16_384),
                }
            )
        for part_index, part in enumerate(message.parts):
            if getattr(part, "tool_name", None):
                evidence.append(
                    {
                        "id": f"message:{index}:tool:{part_index}",
                        "kind": "tool_result",
                        "summary": tool_evidence_summary(
                            part.tool_name,
                            part.tool_output or "",
                            tool_input=part.tool_input,
                            tool_status=part.tool_status,
                        ),
                    }
                )
    # Bound complete records, never cut the serialized JSON. Preserve both early and late facts.
    selected = []
    used = 0
    for item in evidence:
        size = len(item["summary"])
        if used + size <= MAX_EVIDENCE_CHARS // 2:
            selected.append(item)
            used += size
        else:
            break
    selected_ids = {item["id"] for item in selected}
    tail = []
    for item in reversed(evidence):
        if item["id"] in selected_ids:
            continue
        size = len(item["summary"])
        if used + size > MAX_EVIDENCE_CHARS:
            continue
        tail.append(item)
        used += size
    selected.extend(reversed(tail))
    if len(selected) != len(evidence):
        selected.append(
            {
                "id": "evidence:omitted",
                "kind": "truncation",
                "summary": "Some conversation records were omitted by the evidence budget.",
            }
        )
    if execution is not None:
        selected.append(
            {
                "id": EXPERIENCE_EXECUTION_EVIDENCE_ID,
                "kind": "experience_execution",
                "summary": clip_evidence_summary(
                    "experience_execution",
                    json.dumps(execution, ensure_ascii=False),
                    16_384,
                ),
            }
        )
    if evaluation is not None:
        # Preserve the supplied evaluator report as evidence. Criterion definitions and
        # judgments belong to the Case/Trajectory LLM, not a whitelist of boolean keys.
        selected.append(
            {
                "id": "evaluation:external",
                "kind": "external_evaluation",
                "summary": json.dumps(asdict(evaluation), ensure_ascii=False),
            }
        )
    return selected


def trajectory_evaluation_summary(evaluation: RubricEvaluation | None) -> dict | None:
    """Keep external results and diagnostics out of the observed business dialogue."""
    if evaluation is None:
        return None
    summary: dict[str, Any] = {"passed": evaluation.passed, "score": evaluation.score}
    details = {}
    result = evaluation.metadata.get("evaluation_result")
    if isinstance(result, dict):
        # Keep diagnostic facts, not another copy of actions/criteria/rewards or LLM rationales.
        for key in ("termination_reason", "info", "error", "type", "message"):
            value = result.get(key)
            if key == "info" and isinstance(value, dict):
                value = {name: item for name, item in value.items() if item is not None}
            if value is not None and value != {}:
                details[key] = value
    reason = evaluation.metadata.get("termination_reason")
    if reason is not None:
        details["termination_reason"] = reason
    if details:
        summary["details"] = details
    return summary


def trajectory_source_session_uri(memory: MemoryFile) -> str:
    """Resolve the immutable source Session from the Trajectory's provenance link."""
    sources = []
    for value in memory.links or []:
        link = value if isinstance(value, StoredLink) else StoredLink.model_validate(value)
        if link.link_type == "derived_from" and link.to_uri and is_session_uri(link.to_uri):
            sources.append(link.to_uri)
    sources = list(dict.fromkeys(sources))
    if len(sources) != 1:
        raise ValueError("Trajectory must have exactly one derived_from Session link")
    return sources[0]


async def list_trajectory_uris(viking_fs: Any, ctx: Any, directory: str) -> list[str]:
    """List one flat inbox; never recurse into archive or date directories."""
    from openviking.session.archive_store import is_storage_not_found

    directory = directory.rstrip("/")
    result = set()
    offset = 0
    while True:
        try:
            entries = await viking_fs.ls(
                directory,
                output="original",
                node_limit=1000,
                offset=offset,
                sort_by="name",
                sort_order="asc",
                ctx=ctx,
            )
        except Exception as exc:
            if is_storage_not_found(exc):
                break
            raise
        for entry in entries:
            uri = str(entry.get("uri") or "").rstrip("/")
            name = uri.rsplit("/", 1)[-1]
            if (
                uri.rsplit("/", 1)[0] == directory
                and not name.startswith(".")
                and name.endswith(".md")
                and not (entry.get("isDir") or entry.get("is_dir"))
            ):
                result.add(uri)
        if len(entries) < 1000:
            break
        offset += len(entries)
    return sorted(result)


@dataclass(slots=True)
class ObservationGradient:
    """A persisted observation awaiting synthesis, not an Experience patch."""

    analysis: RolloutAnalysis
    target_name: str
    target_uri: str
    after_file: MemoryFile
    before_file: MemoryFile | None = None
    base_version: int | None = None
    rationale: str = "Compare this factual Trajectory with other executions of the same Case."
    confidence: float = 1.0
    links: list[StoredLink] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _processed_success(memory: MemoryFile, case_name: str) -> Trajectory | None:
    """Validate factual outcome/provenance after retrieval, never trust index tags alone."""
    metadata = memory.extra_fields
    if (
        memory.memory_type != "trajectories"
        or metadata.get("case_name") != case_name
        or metadata.get("outcome") != "success"
        or metadata.get("trajectory_format") != TRAJECTORY_FORMAT
        or not metadata.get("trajectory_name")
    ):
        return None
    state = metadata.get("experience_update")
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except ValueError:
            return None
    if not isinstance(state, dict) or state.get("state") not in {"applied", "no_op"}:
        return None
    try:
        source = trajectory_source_session_uri(memory)
    except ValueError:
        return None
    return Trajectory(
        name=metadata["trajectory_name"],
        uri=memory.uri,
        outcome="success",
        content=memory.content,
        metadata={**metadata, "source_session_uri": source},
    )


@tracer("train.trajectory.success_ann", ignore_args=True, ignore_result=True)
async def find_successful_trajectory(
    viking_fs: Any,
    ctx: Any,
    *,
    trajectory: Trajectory,
    directory: str,
    case_name: str,
    exclude_uris: set[str],
) -> Trajectory | None:
    """Return at most one ranked, validated success; never scan the filesystem as fallback."""
    from openviking.session.archive_store import is_storage_not_found
    from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils

    # Stored steps omit result headers, lifecycle metadata and policy boilerplate.
    # The domain object's content is the query for trajectories without extracted steps.
    text = trajectory.metadata.get("steps", trajectory.content)
    if not isinstance(text, str) or not text.strip():
        return None
    query = truncate_text_to_token_budget(text, TRAJECTORY_ANN_QUERY_TOKENS)
    tracer.info(
        f"Trajectory ANN query: uri={trajectory.uri} chars={len(query)} truncated={query != text}"
    )
    conditions = [build_search_tags_filter([f"{TRAJECTORY_OUTCOME_TAG}=success"])]
    if exclude_uris:
        conditions.append({"op": "must_not", "field": "uri", "conds": sorted(exclude_uris)})
    try:
        result = await viking_fs.find(
            query=query,
            target_uri=directory,
            limit=_TRAJECTORY_ANN_CANDIDATES,
            filter={"op": "and", "conds": conditions},
            ctx=ctx,
            level=[2],
        )
    except Exception as exc:
        if is_storage_not_found(exc):
            return None
        raise
    prefix = directory.rstrip("/") + "/"
    seen = set()
    for match in result.memories[:_TRAJECTORY_ANN_CANDIDATES]:
        uri = match.uri
        try:
            canonical = normalize_storage_target_uri(uri)
        except (ValueError, InvalidArgumentError):
            continue
        if canonical != uri or not uri.startswith(prefix) or not uri.endswith(".md"):
            continue
        if any(part.startswith(".") for part in uri[len(prefix) :].split("/")):
            continue
        if uri in exclude_uris or uri in seen:
            continue
        seen.add(uri)
        try:
            raw = await viking_fs.read_file(uri, ctx=ctx)
        except Exception as exc:
            if is_storage_not_found(exc):
                continue
            raise
        if not raw:
            continue
        memory = MemoryFileUtils.read(raw, uri=uri)
        selected = _processed_success(memory, case_name)
        if selected is not None:
            tracer.info(
                f"Trajectory ANN match: query_uri={trajectory.uri} success_uri={uri} score={match.score}"
            )
            return selected
    return None
