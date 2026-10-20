# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Small shared helpers for the train pipeline."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from openviking.session.train.domain import Rollout, RolloutAnalysis

EXPERIENCE_SOURCE_CONTEXTS_KEY = "_experience_source_contexts"


def average_score(analyses: list[RolloutAnalysis | None]) -> float | None:
    """Return the mean evaluation score across analyses, ignoring None entries."""
    scores = [
        float(analysis.evaluation.score)
        for analysis in analyses
        if analysis is not None and analysis.evaluation is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def summarize_session_commits(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count successful commits with/without output, not unique Trajectory files."""
    completed = [
        item for item in results if item.get("task_status") != "skipped" and not item.get("error")
    ]
    with_traj = sum(any(item.get("trajectory_uris") or []) for item in completed)
    return {
        "completed": len(completed),
        "with_traj": with_traj,
        "without_traj": len(completed) - with_traj,
        "errors": sum(bool(item.get("error")) for item in results),
    }


def validate_rollouts_have_cases(rollouts: list[Rollout]) -> None:
    """Raise ``ValueError`` if any rollout is missing its ``case``."""
    missing = [
        idx for idx, rollout in enumerate(rollouts) if getattr(rollout, "case", None) is None
    ]
    if missing:
        raise ValueError(
            f"rollout training requires Rollout.case for all rollouts; missing indices={missing}"
        )


def safe_int(value: Any) -> int | None:
    """Parse ``value`` as a positive integer, returning ``None`` on failure/zero."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def first_uri(uris: list[str]) -> str | None:
    """Return the first URI from a list, or ``None`` if empty."""
    return uris[0] if uris else None


async def call_hook(hook: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke a sync or async hook, preserving its result and exceptions."""
    result = hook(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def compact_source_sessions(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep attribution/outcome fields without copying raw evidence into plans."""
    stored_keys = {"source_session_uri", "case_name", "case_uri", "passed", "score"}
    return [
        {key: value for key, value in context.items() if key in stored_keys}
        for context in contexts
        if context.get("source_session_uri")
    ]
