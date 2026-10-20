"""Synthesize one Experience per Case from batched factual Trajectories."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

from openviking.session.train.components.gradient_estimator import (
    ExperienceGradientContext,
    ExperienceGradientEstimator,
)
from openviking.session.train.components.policy_optimizer import case_experience_plan_items
from openviking.session.train.components.trajectory_observation import (
    ObservationGradient,
    find_successful_trajectory,
)
from openviking.session.train.domain import PolicyUpdatePlan
from openviking.session.train.utils import (
    EXPERIENCE_SOURCE_CONTEXTS_KEY,
)
from openviking.telemetry import tracer


@dataclass(slots=True)
class TrajectoryExperienceOptimizer:
    viking_fs: Any
    estimator: ExperienceGradientEstimator
    memory_registry: Any = None

    @tracer("train.experience.synthesize_trajectories", ignore_args=True, ignore_result=True)
    async def plan(self, gradients, policy_set, context):
        groups = defaultdict(list)
        for signal in gradients:
            if not isinstance(signal, ObservationGradient):
                raise ValueError(
                    "Trajectory optimizer requires observation signals, not DAG patches"
                )
            groups[signal.target_uri].append(signal)
        items, gates, comparisons = [], [], []
        for target, group in groups.items():
            first = group[0]
            if any(g.target_name != first.target_name for g in group):
                raise ValueError("Observation batch contains conflicting Case identities")
            current = {t.uri: t for g in group for t in g.analysis.trajectories}
            history = await self._history(
                first,
                context.request_context,
                current=list(current.values()),
            )
            trajectories = [*current.values(), *history]
            analysis = replace(
                first.analysis,
                trajectories=trajectories,
                metadata={
                    **first.analysis.metadata,
                    "current_trajectory_uris": list(current),
                },
            )
            gradient_context = ExperienceGradientContext(
                context.request_context,
                [],
                memory_registry=self.memory_registry,
                metadata={
                    EXPERIENCE_SOURCE_CONTEXTS_KEY: [self._source_context(t) for t in trajectories]
                },
            )
            proposals = await self.estimator.estimate(analysis, policy_set, gradient_context)
            gates.extend(proposal.metadata["experience_proposal_gate"] for proposal in proposals)
            case_items = []
            # The estimator produced a complete DAG against the observed target version.
            # The deterministic plan retains its base version for commit-time validation.
            for proposal in proposals:
                from openviking.session.memory.dataclass import (
                    ResolvedOperation,
                    ResolvedOperations,
                )

                operations = ResolvedOperations(
                    upsert_operations=[
                        ResolvedOperation(
                            memory_type="experiences",
                            uris=[target],
                            memory_fields=proposal.metadata["memory_fields"],
                        )
                    ],
                    delete_file_contents=[],
                    errors=[],
                )
                case_items.extend(case_experience_plan_items(operations, [proposal], policy_set))
            items.extend(case_items)
            comparisons.append(
                {
                    "case_name": first.target_name,
                    "current_trajectory_count": len(current),
                    "trajectory_uris": [t.uri for t in trajectories],
                    "outcomes": [t.outcome for t in trajectories],
                }
            )
        return PolicyUpdatePlan(
            items=items,
            read_policy_uris=sorted(groups),
            metadata={
                "optimizer": "trajectory_comparison",
                "comparisons": comparisons,
                "proposal_gates": gates,
            },
        )

    async def _history(self, signal, ctx, *, current):
        """Find one semantic success per non-success trajectory, then deduplicate by URI."""
        queries = [trajectory for trajectory in current if trajectory.outcome != "success"]
        if not queries:
            return []
        from openviking.core.namespace import owner_space_for_uri
        from openviking.session.memory.memory_type_registry import get_default_registry
        from openviking.session.memory.utils.uri import generate_uri

        registry = self.memory_registry or get_default_registry()
        schema = registry.get("trajectories")
        if schema is None or not schema.enabled:
            raise ValueError("Trajectory schema is required for history lookup")
        directory = (
            generate_uri(
                schema.model_copy(update={"filename_template": ""}),
                {"case_name": signal.target_name},
                user_space=owner_space_for_uri(signal.target_uri),
            ).rstrip("/")
            + "/archive"
        )
        selected = {}
        exclude_uris = {
            trajectory.metadata.get("archive_uri") or trajectory.uri for trajectory in current
        }
        # The existing learning batch bounds the query count. Do not launch another
        # unbounded layer of concurrent embedding/retrieval calls inside each Case.
        for trajectory in queries:
            success = await find_successful_trajectory(
                self.viking_fs,
                ctx,
                trajectory=trajectory,
                directory=directory,
                case_name=signal.target_name,
                exclude_uris=exclude_uris,
            )
            if success is not None:
                selected.setdefault(success.uri, success)
        return list(selected.values())

    @staticmethod
    def _source_context(trajectory):
        metadata = trajectory.metadata
        evaluation = metadata.get("evaluation") or {}
        return {
            "source_session_uri": metadata["source_session_uri"],
            "case_name": metadata["case_name"],
            "case_uri": "",
            "passed": evaluation.get("passed"),
            "score": evaluation.get("score"),
            "feedback": metadata.get("feedback", []),
            "experience_execution": metadata["experience_execution"],
            "evidence": [
                {k: v for k, v in item.items() if k != "passed"}
                for item in metadata.get("gate_evidence", [])
            ],
        }
