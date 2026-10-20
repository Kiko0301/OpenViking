# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Jointly analyze an archive and persist one new Trajectory per associated Case."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from openviking.config.vlm import VLMResolver
from openviking.message import Message
from openviking.server.identity import RequestContext
from openviking.session.memory import ExtractLoop
from openviking.session.memory.dataclass import (
    MemoryFile,
    MemoryTypeSchema,
    ResolvedOperation,
    ResolvedOperations,
    StoredLink,
)
from openviking.session.memory.experience.feedback import (
    injected_actions,
    record_experience_feedback_stats,
)
from openviking.session.memory.experience.trajectory_context import (
    AgentTrajectoryContextProvider,
    TrajectoryTarget,
    trajectory_source_steps,
)
from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.memory_updater import ExtractContext, MemoryUpdater
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.memory.utils.uri import generate_uri
from openviking.session.skill.session_skill_context_provider import (
    SESSION_SKILL_MEMORY_TYPE,
    SessionSkillContextProvider,
    load_skill_extract_registry,
)
from openviking.session.train.components.trajectory_observation import (
    TRAJECTORY_FORMAT,
    TRAJECTORY_OUTCOME_TAG,
    execution_paths,
    observation_evidence,
    trajectory_evaluation_summary,
    trajectory_source_session_uri,
)
from openviking.session.train.domain import Rollout, RolloutAnalysis, Trajectory
from openviking.session.train.gradients import PatchSemanticGradient
from openviking.session.train.interfaces import RolloutEvaluator
from openviking.storage.viking_fs import get_viking_fs
from openviking.telemetry import tracer
from openviking.utils.tags import merge_search_tags


@dataclass(slots=True)
class SessionAnalyzerContext:
    request_context: RequestContext
    strict_extract_errors: bool = False
    evaluator_context: Any = None
    include_session_skills: bool = False
    source_archive_uri: str = ""
    memory_registry: MemoryTypeRegistry | None = None


@dataclass(slots=True)
class SessionRolloutAnalyzer:
    """Return evaluated Session evidence with a persisted Trajectory.

    Analysis may call the model and write the observation. Experience synthesis
    remains a separate, cross-Session batch operation.
    """

    viking_fs: Any = None
    vikingdb: Any = None
    vlm: Any = None
    evaluator: RolloutEvaluator | None = None
    vlm_resolver: VLMResolver | None = None

    @tracer("train.rollout_analyzer.session.analyze", ignore_result=True, ignore_args=True)
    async def analyze(self, rollout: Rollout, context: SessionAnalyzerContext) -> RolloutAnalysis:
        """Single-rollout interface used by the generic policy trainer."""
        return (await self.analyze_many([rollout], context))[0]

    @tracer("train.rollout_analyzer.session.analyze_many", ignore_result=True, ignore_args=True)
    async def analyze_many(
        self, rollouts: list[Rollout], context: SessionAnalyzerContext
    ) -> list[RolloutAnalysis]:
        """Jointly extract this archive's Cases in one model conversation."""
        if context is None or context.request_context is None:
            raise ValueError("SessionAnalyzerContext.request_context is required")
        if not rollouts:
            return []
        source_uri = str(
            rollouts[0].metadata.get("source_session_uri") or context.source_archive_uri
        )
        if not source_uri:
            raise ValueError("Trajectory requires a source Session archive")
        if len({r.case.name for r in rollouts}) != len(rollouts):
            raise ValueError("Joint Trajectory extraction requires unique Case names")
        if any(
            str(r.metadata.get("source_session_uri") or context.source_archive_uri) != source_uri
            or r.messages != rollouts[0].messages
            for r in rollouts
        ):
            raise ValueError("Joint Trajectory extraction requires the same archive and messages")
        gradients = []
        if context.include_session_skills:
            result = await self.extract_session_skills(
                messages=rollouts[0].messages,
                ctx=context.request_context,
                source_archive_uri=source_uri,
                strict_extract_errors=context.strict_extract_errors,
            )
            gradients = result["skill_gradients"]
        analyses = []
        for rollout in rollouts:
            evaluation = rollout.evaluation
            if evaluation is None and self.evaluator is not None:
                evaluation = await self.evaluator.evaluate(rollout, context.evaluator_context)
            analyses.append(
                RolloutAnalysis(
                    evaluation=evaluation,
                    rollout=rollout,
                    gradients=gradients if not analyses else [],
                    metadata={
                        "case_uri": rollout.metadata.get("case_uri", ""),
                        "task_spec": rollout.metadata.get("task_spec"),
                        "source_session_uri": source_uri,
                        "policy_snapshot_id": rollout.policy_snapshot_id,
                        "experience_execution": _experience_execution_from_rollout(rollout),
                    },
                )
            )
        await self._extract_trajectories(analyses, context)
        return analyses

    @tracer("train.trajectory.observe", ignore_args=True, ignore_result=True)
    async def _extract_trajectories(
        self, analyses: list[RolloutAnalysis], context: SessionAnalyzerContext
    ) -> None:
        """Create one new observation per Case, without scanning/reusing historical sources."""
        registry = context.memory_registry
        if registry is None:
            from openviking.session.memory.memory_type_registry import get_default_registry

            registry = get_default_registry()
        schema = registry.get("trajectories")
        if schema is None or not schema.enabled:
            raise ValueError("Trajectory schema is required for Experience learning")
        exp_schema = registry.get("experiences")
        if exp_schema is None or not exp_schema.enabled:
            raise ValueError("Experience schema is required for deferred learning")
        fs = self.viking_fs or get_viking_fs()
        if fs is None:
            raise RuntimeError("VikingFS is required for Trajectory extraction")
        ctx = context.request_context
        targets = []
        naming_contexts = {}
        created_at = datetime.now(timezone.utc)
        for analysis in analyses:
            rollout = analysis.rollout
            if rollout is None or not analysis.metadata.get("source_session_uri"):
                raise ValueError("Trajectory requires a Case rollout and source Session archive")
            outcome = (
                "unknown"
                if analysis.evaluation is None
                else "success"
                if analysis.evaluation.passed
                else "failure"
            )
            naming = ExtractContext(rollout.messages, split_long_text_messages=False)
            naming._session_timestamp = created_at.strftime("%Y%m%d%H%M%S")
            fields = {
                "trajectory_name": naming.get_random_id(),
                "case_name": rollout.case.name,
                "outcome": outcome,
            }
            uri = await self._allocate_trajectory_uri(fs, schema, fields, ctx, naming)
            naming_contexts[rollout.case.name] = naming
            targets.append(
                TrajectoryTarget(
                    case=asdict(rollout.case),
                    trajectory_name=fields["trajectory_name"],
                    outcome=outcome,
                    target_uri=uri,
                    task_spec=analysis.metadata.get("task_spec"),
                    external_evaluation=asdict(analysis.evaluation)
                    if analysis.evaluation
                    else None,
                    injected_actions=injected_actions(
                        analysis.metadata.get("experience_execution") or {}
                    ),
                )
            )
        model = self.vlm
        if model is None:
            if self.vlm_resolver is None:
                raise RuntimeError("Trajectory extraction requires a VLM or VLM resolver")
            model = await self.vlm_resolver.get_vlm(ctx.account_id)
        provider = AgentTrajectoryContextProvider(
            messages=analyses[0].rollout.messages,
            targets=targets,
            source_steps=trajectory_source_steps(analyses[0].rollout.messages),
            memory_registry=registry,
            ctx=ctx,
            viking_fs=fs,
            vlm_config=model,
        )

        async def allocate_targets(operations: ResolvedOperations) -> list[dict[str, Any]]:
            # Another creator may claim a name while the model runs. Rebind before
            # ExtractLoop's create guards; the file lease still performs the final check.
            for operation in operations.upsert_operations:
                target = provider.targets[operation.memory_fields["case_name"]]
                target.target_uri = await self._allocate_trajectory_uri(
                    fs, schema, operation.memory_fields, ctx, naming_contexts[target.case_name]
                )
                target.trajectory_name = operation.memory_fields["trajectory_name"]
                operation.uris = [target.target_uri]
            return []

        provider.operation_validator = allocate_targets
        await provider.prepare_extraction_messages()
        extract_context = provider.get_extract_context()
        extract_context._random_id = targets[0].trajectory_name
        extract_context._session_timestamp = naming_contexts[
            targets[0].case_name
        ].get_session_timestamp()
        isolation = MemoryIsolationHandler(
            ctx, extract_context, allowed_memory_types={"trajectories"}
        )
        isolation.prepare_messages()
        provider._isolation_handler = isolation
        operations, _ = await ExtractLoop(
            vlm=model,
            viking_fs=fs,
            ctx=ctx,
            context_provider=provider,
            isolation_handler=isolation,
            thinking=True,
        ).run()
        errors = await provider.validate_resolved_operations(operations)
        if errors:
            raise ValueError(f"Invalid joint Trajectory operations: {errors}")
        by_case = {op.memory_fields["case_name"]: op for op in operations.upsert_operations}
        # Validate the entire model result before the first write; the destinations
        # and provenance are server-owned, not generated by the model.
        for analysis, target in zip(analyses, targets, strict=True):
            operation = by_case[target.case_name]
            paths = execution_paths(analysis.metadata.get("experience_execution") or {})
            operation.memory_fields.update(
                {
                    "task_spec": analysis.metadata.get("task_spec"),
                    "trajectory_format": TRAJECTORY_FORMAT,
                    "evaluation": trajectory_evaluation_summary(analysis.evaluation),
                    "experience_execution": paths,
                    "created_at": created_at.isoformat(),
                    "experience_update": {
                        "state": "pending",
                        "attempts": 0,
                        "batch_id": None,
                        "lease_until": None,
                        "processed_at": None,
                        "error": None,
                    },
                }
            )
            uri, raw = await self._write_trajectory(
                operation,
                analysis.metadata["source_session_uri"],
                fs,
                schema,
                registry,
                ctx,
                naming_contexts[target.case_name],
            )
            memory = MemoryFileUtils.read(raw, uri=uri)
            linked_source = trajectory_source_session_uri(memory)
            if (
                memory.extra_fields.get("case_name") != target.case_name
                or linked_source != analysis.metadata["source_session_uri"]
            ):
                raise ValueError("Trajectory identity mismatch")
            if memory.extra_fields.get("task_spec") != analysis.metadata.get("task_spec"):
                raise ValueError("Trajectory task identity mismatch")
            metadata = dict(memory.extra_fields)
            metadata["source_session_uri"] = linked_source
            metadata["gate_evidence"] = observation_evidence(
                analysis.rollout.messages,
                analysis.evaluation,
                metadata.get("experience_execution") or {},
            )
            metadata["feedback"] = (
                list(analysis.evaluation.feedback) if analysis.evaluation is not None else []
            )
            analysis.trajectories = [
                Trajectory(
                    name=memory.extra_fields["trajectory_name"],
                    uri=uri,
                    content=memory.content,
                    outcome=memory.extra_fields["outcome"],
                    metadata=metadata,
                )
            ]
            analysis.metadata["trajectory_memory"] = memory
            analysis.metadata["experience_feedback_stats"] = await record_experience_feedback_stats(
                memory, registry=registry, viking_fs=fs, ctx=ctx
            )

    async def _write_trajectory(
        self,
        operation: ResolvedOperation,
        source: str,
        fs: Any,
        schema: MemoryTypeSchema,
        registry: MemoryTypeRegistry,
        ctx: RequestContext,
        naming_context: ExtractContext,
    ) -> tuple[str, str]:
        """Lock only the actual write set; a filename collision never becomes an overwrite."""
        fields = operation.memory_fields
        for _ in range(5):
            uri = generate_uri(
                schema, fields, user_space=ctx.user.user_id, extract_context=naming_context
            )
            write_uris = [uri]
            if schema.overview_template:
                write_uris.append(uri.rsplit("/", 1)[0] + "/.overview.md")
            lock = await fs._async_agfs.pathlock_acquire_exact_batch(
                sorted(fs._uri_to_path(value, ctx=ctx) for value in write_uris), timeout_secs=30
            )
            if lock is None:
                raise RuntimeError("Trajectory creation requires a file write lock")
            try:
                from openviking.session.train.components.trajectory_lifecycle import (
                    trajectory_archive_uri,
                )

                if await fs.exists(uri, ctx=ctx) or await fs.exists(
                    trajectory_archive_uri(uri), ctx=ctx
                ):
                    fields["trajectory_name"] = naming_context.get_random_id(refresh=True)
                    continue
                operation.uris = [uri]
                operation.create_only = True
                operation.old_memory_file_content = None
                operation.search_tags = merge_search_tags(
                    operation.search_tags, [f"{TRAJECTORY_OUTCOME_TAG}={fields['outcome']}"]
                )
                fields["archive_uri"] = trajectory_archive_uri(uri)
                fields["experience_update"]["search_tags"] = list(operation.search_tags)
                operations = ResolvedOperations(
                    upsert_operations=[operation],
                    delete_file_contents=[],
                    errors=[],
                    resolved_links=[
                        StoredLink(
                            from_uri=uri, to_uri=source, link_type="derived_from", weight=1.0
                        )
                    ],
                )
                # Inbox is a durable work queue, not a retrieval corpus. Enqueue
                # embedding at the final URI during archival, before removing inbox.
                updater = MemoryUpdater(registry=registry, transaction_handle=lock)
                updater._viking_fs = fs
                result = await updater.apply_operations(operations, ctx)
                if result.errors:
                    raise ValueError(
                        "; ".join(f"{target}: {error}" for target, error in result.errors)
                    )
                return uri, await fs.read_file(uri, ctx=ctx)
            finally:
                await fs._async_agfs.pathlock_release(lock)
        raise RuntimeError("Could not allocate an unused Trajectory filename")

    @staticmethod
    async def _allocate_trajectory_uri(fs, schema, fields, ctx, naming_context) -> str:
        from openviking.session.train.components.trajectory_lifecycle import trajectory_archive_uri

        # Point lookups only. The definitive collision check is under the file lock.
        for _ in range(5):
            uri = generate_uri(
                schema, fields, user_space=ctx.user.user_id, extract_context=naming_context
            )
            if not await fs.exists(uri, ctx=ctx) and not await fs.exists(
                trajectory_archive_uri(uri), ctx=ctx
            ):
                return uri
            fields["trajectory_name"] = naming_context.get_random_id(refresh=True)
        raise RuntimeError("Could not allocate an unused Trajectory filename")

    async def extract_session_skills(
        self,
        *,
        messages: list[Message],
        ctx: RequestContext,
        source_archive_uri: str = "",
        strict_extract_errors: bool = False,
    ) -> dict[str, Any]:
        vlm_config = None
        if self.vlm is None:
            if self.vlm_resolver is None:
                raise RuntimeError("Session skill extraction requires a VLM resolver")
            vlm_config = await self.vlm_resolver.get_vlm(ctx.account_id)
        provider = SessionSkillContextProvider(messages=messages, vlm_config=vlm_config)
        provider._registry = load_skill_extract_registry()
        provider.include_tool_parts_in_conversation = True
        provider._ctx = ctx
        viking_fs = self.viking_fs or get_viking_fs()
        provider._viking_fs = viking_fs
        isolation = MemoryIsolationHandler(
            ctx, provider.get_extract_context(), allowed_memory_types={SESSION_SKILL_MEMORY_TYPE}
        )
        isolation.prepare_messages()
        provider._isolation_handler = isolation
        orchestrator = ExtractLoop(
            vlm=self.vlm or vlm_config,
            viking_fs=viking_fs,
            ctx=ctx,
            context_provider=provider,
            isolation_handler=isolation,
            thinking=True,
        )
        operations, _ = await orchestrator.run()
        if operations is None:
            return {"skill_gradients": []}
        if operations.errors:
            raise ValueError(f"Session skill extraction failed: {operations.errors}")
        return {"skill_gradients": _skill_operations_to_gradients(operations)}


def _experience_execution_from_rollout(rollout: Rollout) -> dict[str, dict[str, Any]]:
    """Build the final per-Experience DAG snapshots recorded with a Session."""
    metadata = getattr(rollout, "metadata", {})
    precomputed = metadata.get("experience_execution")
    if isinstance(precomputed, dict):
        return {
            str(uri): dict(snapshot)
            for uri, snapshot in precomputed.items()
            if uri and isinstance(snapshot, dict)
        }
    return experience_execution_from_runtime(metadata.get("dag_runtime"))


def experience_execution_from_runtime(runtime: Any) -> dict[str, dict[str, Any]]:
    """Collapse runtime events into one factual final snapshot per Experience."""
    events = runtime.get("events") if isinstance(runtime, dict) else None
    if not isinstance(events, list):
        return {}

    snapshot_fields = (
        "state",
        "revision",
        "slot_values",
        "slot_evidence",
        "executed_nodes",
        "current_nodes",
        "waiting_for_context",
        "actions",
        "completed_nodes",
        "error",
    )
    snapshots: dict[str, dict[str, Any]] = {}
    node_slots_by_experience: dict[str, dict[int, str]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        experience_uri = str(event.get("experience_uri") or "").strip()
        if not experience_uri:
            continue
        snapshot = snapshots.setdefault(experience_uri, {})
        node_slots = node_slots_by_experience.setdefault(experience_uri, {})
        for node_id, slot_name in dict(event.get("node_slots") or {}).items():
            try:
                node_slots[int(node_id)] = str(slot_name)
            except (TypeError, ValueError):
                continue
        for collection_name in ("actions", "completed_nodes"):
            for node in event.get(collection_name) or []:
                if not isinstance(node, dict) or not node.get("slot_name"):
                    continue
                try:
                    node_slots[int(node.get("node_id"))] = str(node["slot_name"])
                except (TypeError, ValueError):
                    continue
        snapshot["experience_name"] = (
            experience_uri.rstrip("/").rsplit("/", 1)[-1].removesuffix(".md")
        )
        for field_name in snapshot_fields:
            if field_name in event:
                snapshot[field_name] = event[field_name]
        if event.get("delivered_advice"):
            # Delivery is a separate fact; asking or suggesting a tool never proves completion.
            snapshot.setdefault("delivered_advice", []).extend(event["delivered_advice"])
    for experience_uri, snapshot in snapshots.items():
        node_slots = node_slots_by_experience.get(experience_uri, {})
        for field_name in ("executed_nodes", "current_nodes", "waiting_for_context"):
            node_ids = snapshot.get(field_name)
            if isinstance(node_ids, list):
                snapshot[field_name] = [node_slots.get(node_id, node_id) for node_id in node_ids]
    return snapshots


def _skill_operations_to_gradients(
    operations: ResolvedOperations,
    *,
    viking_fs: Any = None,
    ctx: Any = None,
) -> list[PatchSemanticGradient]:
    """Convert skill ResolvedOperations to PatchSemanticGradient instances.

    The resulting gradients carry the full proposed skill content in their
    ``after_file`` so the patch-merge optimizer can reconcile multiple
    proposals against the current policy set.
    """
    gradients: list[PatchSemanticGradient] = []
    for op in operations.upsert_operations or []:
        if op.memory_type != SESSION_SKILL_MEMORY_TYPE:
            continue
        fields = dict(op.memory_fields or {})
        skill_name = str(fields.get("skill_name") or _fallback_skill_name(op))
        target_uri = (op.uris or [None])[0]
        after_content = str(fields.get("content") or "")
        if not after_content.strip():
            continue

        old_file = op.old_memory_file_content
        after_file = MemoryFile(
            uri=target_uri,
            content=after_content,
            memory_type="skills",
            extra_fields={
                **dict(getattr(old_file, "extra_fields", {}) or {}),
                **{k: v for k, v in fields.items() if k != "content"},
                "memory_type": "skills",
                "skill_name": skill_name,
            },
        )
        links: list[StoredLink] = []

        gradients.append(
            PatchSemanticGradient(
                before_file=old_file,
                after_file=after_file,
                base_version=_base_version_from_old_file(old_file),
                rationale=(
                    "Session skill patch extracted from Session by SessionSkillContextProvider."
                ),
                links=links,
                confidence=0.7,
                metadata={
                    "source": "session_skill_extract",
                    "memory_fields": fields,
                    "skill_name": skill_name,
                    "uris": list(op.uris or []),
                },
            )
        )
    return gradients


def _fallback_skill_name(op: Any) -> str:
    uris = getattr(op, "uris", None) or []
    if uris:
        uri = str(uris[0])
        # path/to/skills/my_skill/SKILL.md → my_skill
        parts = uri.rstrip("/").split("/")
        if len(parts) >= 2 and parts[-1] == "SKILL.md":
            return parts[-2]
        return parts[-1].removesuffix(".md")
    return "unknown_skill"


def _base_version_from_old_file(old_file: Any) -> int | None:
    if old_file is None:
        return None
    fields = getattr(old_file, "extra_fields", {}) or {}
    try:
        v = int(fields.get("version"))
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None
