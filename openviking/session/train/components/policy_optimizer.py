# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Policy optimizer implementations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from openviking.config.vlm import VLMResolver
from openviking.message import Message
from openviking.server.identity import RequestContext
from openviking.session.memory.dataclass import MemoryFile, StoredLink
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.memory_updater import ExtractContext
from openviking.session.memory.patch_merge_context_provider import (
    PatchMergeContextProvider,
    PatchMergePatch,
)
from openviking.session.memory.streaming_memory_updater import render_operation_after_file
from openviking.session.train.domain import (
    Policy,
    PolicyPlanItem,
    PolicySet,
    PolicyUpdatePlan,
)
from openviking.session.train.interfaces import SemanticGradient
from openviking.session.train.utils import first_uri, safe_int
from openviking.telemetry import tracer
from openviking_cli.utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PatchMergePolicyOptimizerContext:
    """Context for PatchMergePolicyOptimizer."""

    request_context: RequestContext
    messages: list[Message] = field(default_factory=list)


@dataclass(slots=True)
class PatchMergePolicyOptimizer:
    """Merge patch gradients with ExtractLoop before producing update plan items."""

    viking_fs: Any = None
    vlm: Any = None
    memory_type: str = "experiences"
    memory_registry: MemoryTypeRegistry | None = None
    vlm_resolver: VLMResolver | None = None

    @tracer(
        "train.policy_optimizer.patch_merge.plan",
        ignore_result=True,
        ignore_args=True,
    )
    async def plan(
        self,
        gradients: list[SemanticGradient],
        policy_set: PolicySet,
        context: PatchMergePolicyOptimizerContext | None = None,
    ) -> PolicyUpdatePlan:
        if context is None:
            raise ValueError("PatchMergePolicyOptimizerContext.request_context is required")

        patch_gradients = list(gradients)
        if not patch_gradients:
            return PolicyUpdatePlan(
                items=[],
                read_policy_uris=[],
                metadata={
                    "optimizer": "patch_merge",
                    "memory_type": self.memory_type,
                    "gradient_count": len(gradients),
                    "patch_gradient_count": 0,
                },
            )

        groups: dict[str, list[SemanticGradient]] = defaultdict(list)
        for gradient in patch_gradients:
            if self.memory_type == "experiences" and not gradient.target_uri:
                raise ValueError("Case Experience proposals require a fixed target URI")
            groups[gradient.target_uri if self.memory_type == "experiences" else "all"].append(
                gradient
            )
        items: list[PolicyPlanItem] = []
        for target, group in groups.items():
            operations = await self._run_merge_extract_loop(
                gradients=group, policy_set=policy_set, context=context
            )
            if self.memory_type == "experiences":
                group_items = case_experience_plan_items(operations, group, policy_set)
            else:
                group_items = _operations_to_plan_items(
                    operations=operations,
                    gradients=group,
                    policy_set=policy_set,
                    memory_type=self.memory_type,
                )
            items.extend(group_items)
            _log_merge_output(
                target=target, operations=operations, plan_items=group_items, console=False
            )

        return PolicyUpdatePlan(
            items=items,
            read_policy_uris=sorted(groups) if self.memory_type == "experiences" else None,
            metadata={
                "optimizer": "patch_merge",
                "memory_type": self.memory_type,
                "gradient_count": len(gradients),
                "patch_gradient_count": len(patch_gradients),
                "gradients": [
                    _gradient_to_dict(idx, gradient) for idx, gradient in enumerate(patch_gradients)
                ],
                "experience_proposal_gates": [
                    dict(gradient.metadata["experience_proposal_gate"])
                    for gradient in patch_gradients
                    if isinstance(gradient.metadata.get("experience_proposal_gate"), dict)
                ],
            },
        )

    @tracer(
        "train.policy_optimizer.patch_merge.extract_loop",
        ignore_result=True,
        ignore_args=True,
    )
    async def _run_merge_extract_loop(
        self,
        *,
        gradients: list[SemanticGradient],
        policy_set: PolicySet,
        context: PatchMergePolicyOptimizerContext,
    ):
        vlm_config = None
        if self.vlm is None:
            if self.vlm_resolver is None:
                raise RuntimeError(
                    "PatchMergePolicyOptimizer requires a VLM resolver for account-owned work"
                )
            vlm_config = await self.vlm_resolver.get_vlm(context.request_context.account_id)
            vlm = vlm_config
        else:
            vlm = self.vlm
        viking_fs = self.viking_fs or policy_set.viking_fs
        if viking_fs is None:
            raise RuntimeError("VikingFS is required for patch-merge policy optimization")

        extract_context = ExtractContext(list(context.messages or []))
        provider_kwargs = {
            "fixed_target": self.memory_type == "experiences",
            "memory_type": self.memory_type,
            "memory_registry": self.memory_registry,
            "required_file_uris": _required_file_uris(gradients, policy_set),
            "patches": [_gradient_to_merge_patch(gradient) for gradient in gradients],
        }
        if vlm_config is None:
            provider = PatchMergeContextProvider(**provider_kwargs)
        else:
            provider = PatchMergeContextProvider(
                **provider_kwargs,
                vlm_config=vlm_config,
            )
        provider._ctx = context.request_context
        provider._viking_fs = viking_fs
        provider._extract_context = extract_context

        isolation_handler = MemoryIsolationHandler(
            context.request_context,
            extract_context,
            allowed_memory_types={self.memory_type},
        )
        isolation_handler.prepare_messages()
        provider._isolation_handler = isolation_handler

        _seed_read_file_contents(provider, gradients, policy_set)
        prefetch_messages = await provider.prefetch()
        provider.prefetch = _constant_prefetch(prefetch_messages)
        _log_merge_input(
            target="all",
            provider=provider,
            gradients=gradients,
            prefetch_messages=prefetch_messages,
            console=False,
        )

        orchestrator = ExtractLoop(
            vlm=vlm,
            viking_fs=viking_fs,
            ctx=context.request_context,
            context_provider=provider,
            isolation_handler=isolation_handler,
            max_iterations=1,
            thinking=self.memory_type == "experiences",
        )
        operations, _ = await orchestrator.run()
        if self.memory_type == "experiences" and operations is not None:
            # Plans store complete candidates, not serialized patches. Materialize against
            # the prefetched source using the same content gate as extraction and writing.
            schema = provider.get_memory_schemas(context.request_context)[0]
            for operation in operations.upsert_operations:
                if operation.memory_type != "experiences":
                    continue
                after_file = await render_operation_after_file(
                    operation, schema=schema, extract_context=extract_context
                )
                operation.memory_fields = {**operation.memory_fields, "content": after_file.content}
        return operations


def case_experience_plan_items(
    operations: Any, gradients: list[SemanticGradient], policy_set: PolicySet
) -> list[PolicyPlanItem]:
    if any(
        not gradient.metadata.get("proposal_source_sessions")
        or not (gradient.metadata.get("experience_proposal_gate") or {}).get("passed")
        for gradient in gradients
    ):
        raise ValueError("PatchMerge received an Experience proposal that was not admitted")
    if operations is None:
        return []
    if operations.errors:
        raise ValueError(f"Case Experience merge failed: {operations.errors}")
    if operations.delete_file_contents or operations.delete_replacements:
        raise ValueError("Case Experience merge cannot delete or replace targets")
    if not operations.upsert_operations:
        return []
    target_uri = gradients[0].target_uri
    name = gradients[0].target_name
    if any(g.target_uri != target_uri or g.target_name != name for g in gradients):
        raise ValueError("Case Experience merge contains different canonical targets")
    if len(operations.upsert_operations) != 1:
        raise ValueError("Case Experience merge must produce at most one upsert")
    op = operations.upsert_operations[0]
    fields = dict(op.memory_fields or {})
    if (
        op.memory_type != "experiences"
        or op.uris != [target_uri]
        or fields.get("experience_name") != name
        or fields.get("supersedes")
    ):
        raise ValueError("Case Experience merge changed the fixed target")
    content = str(fields.get("content") or "")
    if not content.strip():
        raise ValueError("Case Experience merge produced empty content")
    current = _find_policy_by_uri(policy_set, target_uri)
    links: dict[str, StoredLink] = {}
    source_sessions: dict[str, dict[str, Any]] = {}
    for gradient in gradients:
        for link in gradient.links:
            if link.link_type == "derived_from" and link.to_uri:
                links[link.to_uri] = link.model_copy(update={"from_uri": target_uri})
        for source in gradient.metadata.get("proposal_source_sessions") or []:
            source_uri = str(source.get("source_session_uri") or "")
            if source_uri:
                source_sessions[source_uri] = _compact_source_session(source)
    if not links:
        raise ValueError("Case Experience merge has no source Trajectory provenance")
    return [
        PolicyPlanItem(
            kind="upsert",
            memory_type="experiences",
            target_name=name,
            target_uri=target_uri,
            before_content=current.content if current else None,
            after_content=content,
            base_version=current.version if current else None,
            links=list(links.values()),
            metadata={
                "merge_gradient_count": len(gradients),
                "source_sessions": list(source_sessions.values()),
                "patch_metadata": {"case_name": name},
            },
        )
    ]


def _compact_source_session(source: dict[str, Any]) -> dict[str, Any]:
    keys = {"source_session_uri", "case_name", "case_uri", "passed", "score"}
    return {key: value for key, value in source.items() if key in keys}


def _constant_prefetch(messages: list[dict[str, Any]]):
    async def prefetch() -> list[dict[str, Any]]:
        return list(messages)

    return prefetch


def _log_merge_input(
    *,
    target: str,
    provider: PatchMergeContextProvider,
    gradients: list[SemanticGradient],
    prefetch_messages: list[dict[str, Any]],
    console: bool,
) -> None:
    lines = [
        "\n========== PatchMergePolicyOptimizer Input =========",
        f"target: {target}",
        f"memory_type: {provider.memory_type}",
        f"required_file_uris: {provider.required_file_uris}",
        f"gradient_count: {len(gradients)}",
    ]
    for idx, gradient in enumerate(gradients):
        before_file = gradient.before_file
        after_file = gradient.after_file
        lines.extend(
            [
                "",
                f"[Gradient {idx}]",
                f"target_name: {gradient.target_name}",
                f"target_uri: {gradient.target_uri}",
                f"base_version: {gradient.base_version}",
                f"confidence: {gradient.confidence}",
                f"links: {_links_to_dicts(gradient.links)}",
                f"rationale: {gradient.rationale}",
            ]
        )
        if after_file is not None:
            lines.extend(
                [
                    "before_file:",
                    _memory_file_summary(before_file),
                    "after_file:",
                    _memory_file_summary(after_file),
                ]
            )
    lines.extend(["", "[Prefetch Messages]"])
    for idx, message in enumerate(prefetch_messages):
        lines.extend(
            [f"--- message {idx} role={message.get('role')} ---", str(message.get("content"))]
        )
    lines.append("===================================================\n")
    tracer.info("\n".join(lines), console=console)


def _log_merge_output(
    *,
    target: str,
    operations: Any,
    plan_items: list[PolicyPlanItem],
    console: bool,
) -> None:
    lines = [
        "\n========== PatchMergePolicyOptimizer Output =========",
        f"target: {target}",
        "[Resolved Operations]",
        _dump_model_or_value(operations),
        "",
        "[Policy Plan Items]",
    ]
    for idx, item in enumerate(plan_items):
        lines.extend(
            [
                f"--- item {idx} ---",
                f"kind: {item.kind}",
                f"memory_type: {item.memory_type}",
                f"target_name: {item.target_name}",
                f"target_uri: {item.target_uri}",
                f"base_version: {item.base_version}",
                f"confidence: {item.confidence}",
                f"links: {_links_to_dicts(item.links)}",
                "before_content:",
                str(item.before_content),
                "after_content:",
                str(item.after_content),
                f"metadata: {item.metadata}",
            ]
        )
    lines.append("====================================================\n")
    tracer.info("\n".join(lines), console=console)


def _dump_model_or_value(value: Any) -> str:
    dumper = getattr(value, "model_dump_json", None)
    if dumper is not None:
        try:
            return str(dumper(indent=2))
        except TypeError:
            return str(dumper())
    return str(value)


def _memory_file_summary(file: MemoryFile | None) -> str:
    if file is None:
        return "None"
    return _dump_model_or_value(
        {
            "uri": file.uri,
            "memory_type": file.memory_type,
            "content": file.content,
            "links": file.links,
            "backlinks": file.backlinks,
            "extra_fields": file.extra_fields,
        }
    )


def _gradient_to_dict(index: int, gradient: SemanticGradient) -> dict[str, Any]:
    result = {
        "index": index,
        "target_name": gradient.target_name,
        "target_uri": gradient.target_uri,
        "base_version": gradient.base_version,
        "rationale": gradient.rationale,
        "links": _links_to_dicts(gradient.links),
        "confidence": gradient.confidence,
        "metadata": _compact_gradient_metadata(gradient.metadata),
    }
    before_file = gradient.before_file
    after_file = gradient.after_file
    if before_file is not None:
        result["before_file"] = _memory_file_to_dict(before_file)
    if after_file is not None:
        result["after_file"] = _memory_file_to_dict(after_file)
    return result


def _memory_file_to_dict(file: MemoryFile) -> dict[str, Any]:
    return {
        "uri": file.uri,
        "memory_type": file.memory_type,
        "content": file.content,
        "links": list(file.links or []),
        "backlinks": list(file.backlinks or []),
        "extra_fields": dict(file.extra_fields or {}),
    }


def _links_to_dicts(links: list[StoredLink] | None) -> list[dict[str, Any]]:
    return [link.model_dump() for link in links or []]


def _gradient_to_merge_patch(gradient: SemanticGradient) -> PatchMergePatch:
    return PatchMergePatch(
        before_file=gradient.before_file,
        after_file=gradient.after_file,
        metadata={
            "base_version": gradient.base_version,
            "rationale": gradient.rationale,
            "links": _links_to_dicts(gradient.links),
            "confidence": gradient.confidence,
            "gradient_metadata": _compact_gradient_metadata(gradient.metadata),
        },
    )


def _compact_gradient_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in metadata.items() if not key.startswith("_")}
    memory_fields = compact.get("memory_fields")
    if isinstance(memory_fields, dict) and "content" in memory_fields:
        compact["memory_fields"] = {
            key: value for key, value in memory_fields.items() if key != "content"
        }
    return compact


def _required_file_uris(
    gradients: list[SemanticGradient],
    policy_set: PolicySet,
) -> list[str]:
    uris: list[str] = []
    for gradient in gradients:
        uri = gradient.target_uri
        if uri and uri not in uris:
            uris.append(uri)
    return uris


def _seed_read_file_contents(
    provider: PatchMergeContextProvider,
    gradients: list[SemanticGradient],
    policy_set: PolicySet,
) -> None:
    for policy in policy_set.policies:
        if policy.uri in provider.required_file_uris:
            provider.read_file_contents[policy.uri] = _policy_to_memory_file(
                policy, memory_type=provider.memory_type
            )
    for gradient in gradients:
        before_file = gradient.before_file
        target_uri = gradient.target_uri
        if (
            provider.fixed_target
            or before_file is None
            or target_uri in provider.read_file_contents
        ):
            continue
        if target_uri:
            provider.read_file_contents[target_uri] = before_file


def _policy_to_memory_file(policy: Policy, *, memory_type: str = "experiences") -> MemoryFile:
    name_field = _name_field_for_memory_type(memory_type)
    extra_fields = dict(policy.metadata)
    extra_fields["memory_type"] = memory_type
    extra_fields[name_field] = policy.name
    extra_fields.setdefault("version", policy.version)
    extra_fields.setdefault("status", policy.status)
    return MemoryFile(
        uri=policy.uri,
        content=policy.content,
        links=list(policy.links or []),
        backlinks=list(policy.backlinks or []),
        memory_type=memory_type,
        extra_fields=extra_fields,
    )


def _operations_to_plan_items(
    *, operations: Any, gradients: list[SemanticGradient], policy_set: PolicySet, memory_type: str
) -> list[PolicyPlanItem]:
    """Convert generic skill merge output; Case Experiences use fixed-target validation."""
    if operations is None:
        return []
    items = []
    name_field = _name_field_for_memory_type(memory_type)
    for op in operations.upsert_operations:
        if op.memory_type != memory_type:
            continue
        fields = dict(op.memory_fields or {})
        content = str(fields.get("content") or "")
        if not content.strip():
            continue
        uri = first_uri(op.uris)
        current = _find_policy_by_uri(policy_set, uri) if uri else None
        old_file = op.old_memory_file_content
        items.append(
            PolicyPlanItem(
                kind="upsert",
                memory_type=memory_type,
                target_name=str(
                    fields.get(name_field) or _fallback_policy_name(op, memory_type=memory_type)
                ),
                target_uri=uri,
                before_content=current.content
                if current
                else old_file.plain_content()
                if old_file
                else None,
                after_content=content,
                base_version=_base_version_from_old_file_or_policy(old_file, uri, policy_set),
                confidence=max((g.confidence for g in gradients), default=None),
                links=[link for link in operations.resolved_links if link.from_uri == uri],
                metadata={"merge_gradient_count": len(gradients), "merge_memory_fields": fields},
            )
        )
    for old in operations.delete_file_contents:
        if old.uri and all(item.target_uri != old.uri for item in items):
            items.append(
                PolicyPlanItem(
                    kind="delete",
                    memory_type=memory_type,
                    target_name=str(old.extra_fields.get(name_field) or ""),
                    target_uri=old.uri,
                    before_content=old.plain_content(),
                    after_content=None,
                )
            )
    return items


def _name_field_for_memory_type(memory_type: str) -> str:
    """Return the extra_fields key for the policy name in a given memory type."""
    if memory_type == "experiences":
        return "experience_name"
    if memory_type in {"skills", "session_skills"}:
        return "skill_name"
    if memory_type.endswith("s"):
        return f"{memory_type[:-1]}_name"
    return f"{memory_type}_name"


def _fallback_policy_name(op: Any, *, memory_type: str) -> str:
    uri = first_uri(getattr(op, "uris", []) or [])
    if uri:
        # For skills: path/to/skills/my_skill/SKILL.md → my_skill
        if memory_type == "skills" and uri.endswith("/SKILL.md"):
            parts = uri.rstrip("/").split("/")
            if len(parts) >= 2:
                return parts[-2]
        return uri.rstrip("/").split("/")[-1].removesuffix(".md")
    return f"unknown_{memory_type.rstrip('s')}"


def _find_policy_by_uri(policy_set: PolicySet, uri: str) -> Policy | None:
    for policy in policy_set.policies:
        if policy.uri == uri:
            return policy
    return None


def _base_version_from_old_file_or_policy(
    old_file: Any, target_uri: str | None, policy_set: PolicySet
) -> int | None:
    if old_file is not None:
        version = safe_int((getattr(old_file, "extra_fields", {}) or {}).get("version"))
        if version is not None:
            return version
    if target_uri:
        policy = _find_policy_by_uri(policy_set, target_uri)
        return policy.version if policy is not None else None
    return None
