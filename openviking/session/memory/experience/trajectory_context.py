# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Agent Trajectory Context Provider - Phase 1 of agent-scope extraction.

Extracts execution trajectories from the conversation and can optionally
co-extract reusable executable skills in the same ReAct pass.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Dict, List

from openviking.server.identity import RequestContext
from openviking.session.memory.dataclass import ResolvedOperations
from openviking.session.memory.experience.dag import clip_evidence_summary, tool_evidence_summary
from openviking.session.memory.experience.feedback import normalize_action_scores
from openviking.session.memory.experience.rubric import (
    snapshot_rubric_results,
    validate_external_rubric_results,
)
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.session_extract_context_provider import SessionExtractContextProvider
from openviking.session.skill.session_skill_context_provider import (
    SESSION_SKILL_MEMORY_TYPE,
    SessionSkillContextProvider,
    resolve_skill_extract_templates_dir,
)
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

TRAJECTORY_MEMORY_TYPE = "trajectories"
SYSTEM_STEP_MAX_CHARS = 32_768


@dataclass(slots=True)
class TrajectoryTarget:
    """Server-owned destination and Case-specific context for a joint extraction."""

    case: dict[str, Any]
    trajectory_name: str = ""
    outcome: str = "unknown"
    target_uri: str = ""
    task_spec: dict[str, Any] | None = None
    external_evaluation: dict[str, Any] | None = None
    injected_actions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def case_name(self) -> str:
        return self.case["name"]

    @property
    def rubric(self) -> list[dict[str, str]]:
        return [
            {"id": item["name"], "description": item["description"]}
            for item in self.case.get("rubric", {}).get("criteria", [])
        ]

    @property
    def action_score_sources(self) -> dict[str, dict[str, Any]]:
        return {
            item["id"]: {key: item[key] for key in ("experience_uri", "slot_name", "instruction")}
            for item in self.injected_actions
        }


class AgentTrajectoryContextProvider(SessionExtractContextProvider):
    """Phase 1 provider: extract trajectories and optional session skills."""

    include_tool_parts_in_conversation = True
    split_long_text_messages_for_extraction = False

    _SHARED_SKILL_STATE = {
        "messages",
        "latest_archive_overview",
        "_output_language",
        "_extract_context",
        "_isolation_handler",
        "_read_file_contents",
        "_allowed_memory_read_uris",
        "_ctx",
        "_viking_fs",
        "_transaction_handle",
    }

    def __init__(
        self,
        *args,
        include_trajectories: bool = True,
        include_session_skills: bool = False,
        targets: list[TrajectoryTarget] | None = None,
        source_steps: list[dict[str, str]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._include_trajectories = include_trajectories
        self._include_session_skills = include_session_skills
        targets = list(targets or [])
        if any(not target.case_name for target in targets):
            raise ValueError("Trajectory targets require non-empty Case names")
        self.targets = {target.case_name: target for target in targets}
        if len(self.targets) != len(targets):
            raise ValueError("Trajectory targets must have unique Case names")
        self.source_steps = source_steps
        self.operation_validator: (
            Callable[[ResolvedOperations], Awaitable[list[dict[str, Any]]]] | None
        ) = None
        self._skill_provider = SessionSkillContextProvider(*args, **kwargs)
        self._sync_skill_provider_state()

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name not in self._SHARED_SKILL_STATE:
            return
        skill_provider = self.__dict__.get("_skill_provider")
        if skill_provider is not None:
            setattr(skill_provider, name, value)

    def _sync_skill_provider_state(self, *, include_extract_context: bool = False) -> None:
        skill_provider = getattr(self, "_skill_provider", None)
        if skill_provider is None:
            return
        if include_extract_context and self._extract_context is None:
            self.get_extract_context()
        for attr in self._SHARED_SKILL_STATE:
            setattr(skill_provider, attr, getattr(self, attr))

    def instruction(self) -> str:
        language_instruction = f"Write new prose in {self.get_output_language()}.\n"
        if self.source_steps is not None:
            schema = self._get_registry().get(TRAJECTORY_MEMORY_TYPE)
            text_fields = {
                field.name: field
                for field in schema.fields
                if field.name in {"context_rules", "steps"}
            }
            if set(text_fields) != {"context_rules", "steps"} or any(
                not str(field.description or "").strip() for field in text_fields.values()
            ):
                raise ValueError(
                    "Trajectory context_rules and steps require extraction instructions"
                )
            field_instructions = "\n\n".join(
                f"{name}:\n{text_fields[name].description.strip()}"
                for name in ("context_rules", "steps")
            )
            return (
                language_instruction + field_instructions + "\n"
                "Jointly extract exactly one Trajectory for EACH supplied Case, and no other memory "
                "operation. Compare the supplied Cases to assign the relevant evidence to each. "
                "Copy each case_name exactly from cases[].case.name; do not omit, repeat or invent "
                "a Case. Use one call per Case in the SAME reply. Use:\n"
                "sdk.create_trajectories(case_name='<supplied Case name>', "
                "context_rules='''<relevant verbatim rules, or empty>''', "
                "steps='''1. `user` factual step\n...''', "
                "evaluation_criteria=r'''<YAML list>''', action_scores=r'''<YAML list>''')\n"
                "then sdk.commit(). The server supplies the trajectory filename, outcome and "
                "lifecycle metadata. Do not generate those fields.\n"
                "Keep context_rules and steps as separate triple-single-quoted string arguments. "
                "Do not add section titles; the content template renders them. "
                "System rules in source_steps are context, not dialogue steps; do not emit "
                "a step such as 'user received the policy'. Genuine user policy quotations "
                "and assistant explanations remain dialogue. "
                "Assess each supplied injected action in action_scores; keep steps factual. "
                "Use 1 for supported user-goal benefit or -1 for supported side effects. "
                "Otherwise omit score and explain why; do not use 0 or null."
            )
        return (
            language_instruction
            + "Compress each supplied user message, assistant message, Experience Reminder and tool "
            "record into one minimal factual step in the same order. Use system rules only as "
            "Session Context, never as numbered execution steps. Preserve exact relevant "
            "values. Do not analyze causes, write a future procedure or invent a corrected path."
        )

    def get_memory_schemas(self, ctx: RequestContext) -> List[Any]:
        """Expose trajectory schema and optionally session skill schema."""
        del ctx
        registry = self._get_registry()
        memory_types: List[str] = []
        if self._include_trajectories:
            memory_types.append(TRAJECTORY_MEMORY_TYPE)
        if self._include_session_skills:
            memory_types.append(SESSION_SKILL_MEMORY_TYPE)

        schemas: List[Any] = []
        for memory_type in memory_types:
            schema = registry.get(memory_type)
            if schema is None or not schema.enabled:
                continue
            schemas.append(schema)
        return schemas

    async def prefetch(self) -> List[Dict[str, Any]]:
        if self.source_steps is not None:
            return [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "cases": [
                                {
                                    "case": {**target.case, "rubric": target.rubric},
                                    "task_spec": target.task_spec,
                                    "external_evaluation": target.external_evaluation,
                                    "injected_actions": target.injected_actions,
                                }
                                for target in self.targets.values()
                            ],
                            "source_steps": self.source_steps,
                        },
                        ensure_ascii=False,
                    ),
                }
            ]
        if not self._include_session_skills:
            if not isinstance(self.messages, list):
                logger.warning(f"Expected List[Message], got {type(self.messages)}")
                return []
            return [self._build_conversation_message()]
        self._sync_skill_provider_state()
        return await self._skill_provider.prefetch()

    async def execute_tool(self, tool_call) -> Any:
        if not self._include_session_skills:
            return await super().execute_tool(tool_call)
        self._sync_skill_provider_state(include_extract_context=True)
        return await self._skill_provider.execute_tool(tool_call)

    def get_tools(self) -> List[str]:
        return self._skill_provider.get_tools() if self._include_session_skills else []

    def customize_tool_schema(self, tool_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not self._include_session_skills:
            return schema
        self._sync_skill_provider_state(include_extract_context=True)
        return self._skill_provider.customize_tool_schema(tool_name, schema)

    def prepare_operation_fields(self, memory_type: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Supply server-owned identity before the normal URI/isolation resolution."""
        if memory_type != TRAJECTORY_MEMORY_TYPE or self.source_steps is None:
            return fields
        target = self.targets.get(fields.get("case_name"))
        # Keep an unknown Case name intact for provider feedback. Supply only the
        # path-template metadata so resolution can reach validation, never writing.
        return {
            **fields,
            "trajectory_name": target.trajectory_name if target else "",
            "outcome": target.outcome if target else "unknown",
        }

    async def validate_resolved_operations(
        self, operations: ResolvedOperations
    ) -> list[dict[str, Any]]:
        if self.source_steps is None:
            return []
        errors = []
        if operations.errors:
            errors.append({"error": "Invalid Trajectory operations", "details": operations.errors})
        if (
            operations.delete_file_contents
            or operations.delete_replacements
            or operations.case_references
        ):
            errors.append(
                {"error": "Trajectory extraction can only create the supplied Cases' trajectories"}
            )
        if any(op.memory_type != TRAJECTORY_MEMORY_TYPE for op in operations.upsert_operations):
            errors.append({"error": "Every operation must create a Trajectory"})
        counts = Counter(op.memory_fields.get("case_name") for op in operations.upsert_operations)
        if set(counts) != set(self.targets) or any(count != 1 for count in counts.values()):
            errors.append(
                {
                    "error": "Create exactly one Trajectory per supplied Case",
                    "missing_cases": sorted(set(self.targets) - set(counts)),
                    "unexpected_cases": sorted(
                        str(name) for name in set(counts) - set(self.targets)
                    ),
                    "duplicate_cases": sorted(
                        str(name) for name, count in counts.items() if count > 1
                    ),
                }
            )
        if errors:
            return errors
        for operation in operations.upsert_operations:
            target = self.targets[operation.memory_fields["case_name"]]
            criteria = snapshot_rubric_results(
                operation.memory_fields.get("evaluation_criteria", []), target.rubric
            )
            errors.extend(
                {"case_name": target.case_name, **error}
                for error in validate_external_rubric_results(criteria, target.external_evaluation)
            )
            scores = normalize_action_scores(
                operation.memory_fields.get("action_scores", []), target.action_score_sources
            )
            operation.memory_fields.update(
                {
                    "trajectory_name": target.trajectory_name,
                    "outcome": target.outcome,
                    "action_scores": scores,
                    "evaluation_criteria": criteria,
                    "action_score_sources": target.action_score_sources,
                }
            )
            operation.uris = [target.target_uri]
            operation.old_memory_file_content = None
            operation.create_only = True
        operations.resolved_links = []
        if errors:
            return errors
        if self.operation_validator is not None:
            return await self.operation_validator(operations)
        return []

    @staticmethod
    def render_operation_repair(errors: list[dict[str, Any]]) -> str:
        return (
            "The joint Trajectory extraction was invalid. Re-emit the complete operations in the "
            "active output format, with exactly one Trajectory for EACH supplied case_name and "
            "no other Case. In Python, use one sdk.create_trajectories(case_name=..., ...) per "
            "Case, then sdk.commit(). Keep context_rules and steps each as a triple-single-quoted "
            "string ('''...'''). The template generates section titles; do not change the "
            "extracted text merely to match numbering or a step count. Errors: "
            + json.dumps(errors, ensure_ascii=False)
        )

    def _get_registry(self) -> MemoryTypeRegistry:
        if self._registry is None:
            registry = MemoryTypeRegistry(load_schemas=self._include_trajectories)
            if self._include_session_skills:
                loaded = registry.load_from_directory(str(resolve_skill_extract_templates_dir()))
                if loaded == 0:
                    raise RuntimeError(
                        "No session skill schemas loaded from skill_extract templates"
                    )
            self._registry = registry
        return self._registry


def trajectory_source_steps(messages: list[Any]) -> list[dict[str, str]]:
    """Preserve message order/roles, with a larger independent budget for system text."""
    records: list[tuple[str, Any]] = []
    for message in messages:
        content = str(message.content or "")
        step_type = (
            "experience_reminder"
            if content.startswith("[Experience Reminder]")
            else str(message.role)
        )
        tool_parts = [part for part in message.parts if getattr(part, "tool_name", None)]
        # A tool-only transport message is not an empty user/assistant utterance.
        # Keep real empty dialogue and the text of mixed text/tool messages.
        if content.strip() or not tool_parts:
            records.append((step_type, content))
        for part in tool_parts:
            records.append((f"tool:{part.tool_name}", part))
    item_budget = min(16_384, max(1, 64_000 // max(1, len(records))))
    source_steps = []
    for step_type, value in records:
        if step_type.startswith("tool:"):
            content = tool_evidence_summary(
                value.tool_name,
                value.tool_output or "",
                tool_input=value.tool_input,
                tool_status=value.tool_status,
                max_chars=item_budget,
            )
        else:
            text = str(value)
            # Tau2 also records system policy in a "system:" text envelope.
            # Keep its original role/text, but do not squeeze rules as record count grows.
            text_budget = (
                SYSTEM_STEP_MAX_CHARS
                if step_type == "system" or text.lstrip().startswith("system:")
                else item_budget
            )
            content = clip_evidence_summary("text", text, text_budget)
        source_steps.append({"type": step_type, "content": content})
    return source_steps
