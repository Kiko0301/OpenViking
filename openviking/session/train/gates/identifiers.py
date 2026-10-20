# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Keep session-specific entity identifiers out of reusable Tell reminders."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, ClassVar

from openviking.session.memory.experience.compiler import compile_dag
from openviking.session.memory.experience.dag import Dag, GeneralNode, TellAgent
from openviking.session.train.domain import PolicySet, PolicyUpdatePlan, RolloutAnalysis
from openviking.session.train.gates.errors import ExperienceGateRejected
from openviking.session.train.interfaces import SemanticGradient

_IDENTIFIER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{5,64}(?![A-Za-z0-9_-])")


@dataclass(slots=True)
class ExperienceIdentifierGate:
    # Temporarily disabled: the token heuristic rejects policy terms such as "24-hour".
    ENABLED: ClassVar[bool] = False

    analysis: RolloutAnalysis

    async def validate(
        self,
        plan: PolicyUpdatePlan,
        gradients: list[SemanticGradient],
        policy_set: PolicySet,
        context: Any,
    ) -> PolicyUpdatePlan:
        if not self.ENABLED:
            return plan
        issues = []
        for item in plan.items:
            if item.kind != "upsert" or item.memory_type != "experiences":
                continue
            identifiers = _session_identifier_leaks(item.after_content or "", self.analysis)
            if identifiers:
                issues.append(
                    {
                        "gate": type(self).__name__,
                        "uri": item.target_uri,
                        "identifiers": identifiers,
                        "error": "Tell reminders contain session-specific entity IDs. Replace these "
                        "IDs with descriptions derived from the current request; preserve the "
                        "general rule and re-emit the corrected operations.",
                    }
                )
        if issues:
            raise ExperienceGateRejected(issues)
        return plan


def _session_identifier_leaks(
    candidate_source: str,
    analysis: RolloutAnalysis,
) -> list[str]:
    """Find entity-like identifiers copied from Session evidence into tell reminders."""
    compiled = Dag.model_validate_json(compile_dag(candidate_source))
    reminder_text = "\n".join(
        node.slot_provider.instruction
        for node in compiled.nodes.values()
        if isinstance(node, GeneralNode) and isinstance(node.slot_provider, TellAgent)
    ).casefold()
    if not reminder_text:
        return []
    rollout = analysis.rollout
    source_payload = {
        "case_input": rollout.case.input if rollout is not None else {},
        "trajectories": [
            {
                "content": trajectory.content,
                "task_spec": trajectory.metadata.get("task_spec"),
                "evaluation_criteria": trajectory.metadata.get("evaluation_criteria", []),
            }
            for trajectory in analysis.trajectories
        ],
    }
    source_text = json.dumps(source_payload, ensure_ascii=False, default=str)
    identifiers = {
        match.group(0)
        for match in _IDENTIFIER_TOKEN_RE.finditer(source_text)
        if any(character.isalpha() for character in match.group(0))
        and any(character.isdigit() for character in match.group(0))
    }
    return sorted(
        (identifier for identifier in identifiers if identifier.casefold() in reminder_text),
        key=str.casefold,
    )
