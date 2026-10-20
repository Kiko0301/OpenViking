# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Externally triggered Phase 2 learning with server-reported Case progress."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from openviking.session.train.components.progress import ProgressPrinter
from openviking.session.train.components.session_commit import wait_for_task
from openviking.session.train.context import PipelineContext
from openviking.session.train.domain import (
    ExperienceSet,
    PolicyApplyResult,
    PolicyUpdatePlan,
    RolloutTrainingResult,
)
from openviking.session.train.utils import average_score, summarize_session_commits
from openviking.telemetry import tracer
from openviking_cli.client.http import AsyncHTTPClient


@dataclass(slots=True)
class RemoteExperienceLearner:
    client: AsyncHTTPClient
    poll_interval_seconds: float = 2.0
    timeout_seconds: float | None = None
    show_progress: bool = False
    event_recorder: Any | None = None

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0 or None")

    async def learn(
        self,
        batches: list[RolloutTrainingResult],
        policy_set: ExperienceSet,
        context: PipelineContext | Any = None,
    ) -> RolloutTrainingResult:
        """Use Phase 1 outputs to trigger their Case inboxes, never resubmit Sessions."""
        analyses = [analysis for batch in batches for analysis in batch.analyses]
        commits = [
            item
            for batch in batches
            for item in batch.apply_result.metadata.get("commit_results", [])
        ]
        errors = [error for batch in batches for error in batch.apply_result.errors]
        trajectory_uris = list(
            dict.fromkeys(
                uri
                for item in commits
                if not item.get("error")
                for uri in item.get("trajectory_uris", [])
                if uri
            )
        )
        execution_metadata = dict(getattr(context, "execution_metadata", {}) or {})
        learning: dict[str, Any] = {
            "case_count": 0,
            "trajectory_count": 0,
            "applied_count": 0,
            "no_op_count": 0,
            "failed_count": 0,
            "written_experience_uris": [],
            "results": [],
        }
        task_id: str | None = None
        printer: ProgressPrinter | None = None
        last_progress: dict[str, int] | None = None

        async def on_update(task: dict[str, Any]) -> None:
            nonlocal printer, last_progress
            progress = task.get("meta", {}).get("progress")
            if progress is None or progress == last_progress:
                return
            last_progress = dict(progress)
            if printer is None:
                printer = ProgressPrinter(
                    total=progress["total"],
                    label="EXPERIENCE_LEARN",
                    enabled=self.show_progress,
                    description=f"Learning {progress['total']} Cases",
                )
            printer.update_counts(
                completed=progress["completed"],
                failed=0,
                running=progress["running"],
            )
            await self._record_learning_event(
                "experience_learn_progress",
                execution_metadata=execution_metadata,
                task_id=task_id,
                **progress,
            )

        # URIs notify the server which canonical Cases have work. It consumes both
        # old and new pending records in those inboxes, not only these URI entries.
        # No notifications must still not trigger a global scan of unrelated Cases.
        if trajectory_uris:
            print(
                "[EXPERIENCE_LEARN_START] Consuming Case inboxes triggered by "
                f"{len(trajectory_uris)} Trajectories",
                flush=True,
            )
            await self._record_learning_event(
                "experience_learn_start",
                execution_metadata=execution_metadata,
                trigger_trajectory_count=len(trajectory_uris),
                selection_mode="case_inbox",
            )
            try:
                submission = await self.client.learn_experiences(
                    trajectory_uris=trajectory_uris,
                    # Use the API's Case limit, not rollout names: one Session can
                    # produce Trajectories for multiple canonical Cases.
                    max_cases=1000,
                    wait=False,
                )
                task_id = submission["task_id"]
                await self._record_learning_event(
                    "experience_learn_submitted",
                    execution_metadata=execution_metadata,
                    task_id=task_id,
                )
                task = await wait_for_task(
                    self.client,
                    task_id,
                    timeout_seconds=self.timeout_seconds,
                    poll_interval_seconds=self.poll_interval_seconds,
                    timeout_error="experience learning task timeout",
                    on_update=on_update,
                )
                if task.get("status") != "completed":
                    raise RuntimeError(f"{task_id}: {task.get('error') or task.get('status')}")
                if not isinstance(task.get("result"), dict):
                    raise ValueError(f"{task_id}: completed learning task has no result")
                learning = task["result"]
                # Failed Cases are skipped in lenient training. Their server-owned
                # state/error remains available for inspection and retry_failed.
                for item in learning.get("results", []) or []:
                    if item.get("state") == "failed":
                        tracer.error(
                            f"Experience learning skipped Case {item.get('case_name')}: "
                            f"{item.get('error') or 'unknown error'}",
                            console=False,
                        )
            except (Exception, asyncio.CancelledError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                tracer.error(
                    "Experience learning interrupted; stopping before evaluation",
                    e=exc,
                    console=False,
                )
                # An interrupted wait is not evidence that every Case failed remotely.
                learning = {
                    **learning,
                    "error": error,
                    "progress": last_progress,
                    "incomplete": True,
                    # Trigger count is not work count: inbox can contain older work.
                    "trajectory_count": None,
                }
                if last_progress is not None:
                    learning.update(
                        case_count=last_progress["total"],
                        applied_count=last_progress["applied"],
                        no_op_count=last_progress["no_op"],
                        failed_count=last_progress["failed"],
                    )
                # An incomplete result would let the pipeline evaluate while server writes continue.
                raise
            finally:
                if printer is not None:
                    printer.finish()
                learning["task_id"] = task_id
                learning["trigger_trajectory_count"] = len(trajectory_uris)
                learning["selection_mode"] = "case_inbox"
                await self._record_learning_event(
                    "experience_learn_done",
                    execution_metadata=execution_metadata,
                    task_id=task_id,
                    **{
                        key: learning.get(key)
                        for key in (
                            "case_count",
                            "trajectory_count",
                            "trigger_trajectory_count",
                            "selection_mode",
                            "applied_count",
                            "no_op_count",
                            "failed_count",
                            "written_experience_uris",
                            "error",
                            "incomplete",
                            "progress",
                        )
                    },
                    case_results=[
                        {
                            "case_name": item.get("case_name"),
                            "state": item.get("state"),
                            "trajectory_count": len(item.get("trajectory_uris", [])),
                            "experience_uri": item.get("experience_uri"),
                            "error": item.get("error"),
                        }
                        for item in learning.get("results", [])
                    ],
                )
                print(
                    "[EXPERIENCE_LEARN] "
                    f"cases={learning.get('case_count', 0)} "
                    f"trajectories={learning.get('trajectory_count', 0)} "
                    f"applied={learning.get('applied_count', 0)} "
                    f"no_op={learning.get('no_op_count', 0)} "
                    f"skipped={learning.get('failed_count', 0)}"
                    + (" incomplete=1" if learning.get("incomplete") else ""),
                    flush=True,
                )
        metadata = {
            **(batches[-1].apply_result.metadata if batches else {}),
            "committed_rollout_count": sum(
                item.get("task_status") != "skipped" for item in commits
            ),
            "skipped_rollout_count": sum(item.get("task_status") == "skipped" for item in commits),
            "commit_results": commits,
            "commit_summary": summarize_session_commits(commits),
            "trajectory_uris": trajectory_uris,
            "experience_learning": learning,
        }
        return RolloutTrainingResult(
            analyses=analyses,
            gradients=[],
            plan=PolicyUpdatePlan(metadata={"trainer": "experience_learner"}),
            apply_result=PolicyApplyResult(
                updated_policy_set=policy_set,
                written_uris=list(learning.get("written_experience_uris", []) or []),
                errors=errors,
                metadata=metadata,
            ),
            metadata={
                **(batches[-1].metadata if batches else {}),
                **metadata,
                "source": "experience_learner",
                "analysis_count": len(analyses),
                "rollout_count": len(analyses),
                "gradient_count": 0,
                "score": average_score(analyses),
            },
        )

    async def _record_learning_event(
        self,
        event: str,
        *,
        execution_metadata: dict[str, Any],
        **fields: Any,
    ) -> None:
        if self.event_recorder is None:
            return
        record = getattr(self.event_recorder, "record", None)
        if record is None:
            return
        result = record(
            event,
            stage="experience_learn",
            epoch=execution_metadata.get("epoch"),
            training=execution_metadata.get("training"),
            rollout_stage=execution_metadata.get("rollout_stage")
            or execution_metadata.get("stage"),
            **fields,
        )
        if asyncio.iscoroutine(result):
            await result
