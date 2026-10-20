# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Simplified ReAct orchestrator for memory updates - single LLM call with tool use.

Reference: bot/vikingbot/agent/loop.py AgentLoop structure
"""

import asyncio
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openviking.models.vlm.base import ToolCall, VLMBase
from openviking.server.identity import RequestContext
from openviking.session.memory.constants import DEFAULT_EXTRACT_MAX_ITERATIONS
from openviking.session.memory.dataclass import (
    DeleteId,
    MemoryFile,
    MemoryOperationSkip,
    MemoryOperationSkipCode,
    ResolvedOperation,
    ResolvedOperations,
    StoredLink,
    WikiLink,
)
from openviking.session.memory.edit_repair import EditRepair, EditRepairResult
from openviking.session.memory.extraction_output_protocol import (
    ExtractionOutputContext,
    ExtractionOutputProtocol,
    create_extraction_output_protocol,
)
from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
from openviking.session.memory.merge_op import FieldType, ImmutableOp, MergeOp, PatchOp
from openviking.session.memory.page_id_map import ResponsePageIdAllocator
from openviking.session.memory.schema_model_generator import SchemaModelGenerator
from openviking.session.memory.tools import MEMORY_TOOLS_REGISTRY
from openviking.session.memory.validation.create import (
    MemoryCreateConflictError,
    memory_create_target_key,
    validate_create_operations,
)
from openviking.storage.viking_fs import VikingFS, get_viking_fs
from openviking.telemetry import bind_telemetry_stage, tracer
from openviking_cli.utils import get_logger
from openviking_cli.utils.config import get_openviking_config

logger = get_logger(__name__)

# Memory extraction may rewrite an entire memory file in one response. Without an
# explicit cap, providers fall back to a small server-side default (e.g. Ark's
# 4096) and truncate the program mid-string, making the output unusable. This is
# a functional floor for extraction, tuned for the primary Doubao models.
# Models whose max output tokens is BELOW this value (e.g. gpt-4o-mini at 16384)
# must set `vlm.max_tokens` in ov.conf to their own limit to override this default.
_DEFAULT_EXTRACTION_MAX_OUTPUT_TOKENS = 32768


@dataclass
class _EventRepairOperations:
    """Server-filtered operation set accepted from one Event repair response."""

    events: List[Dict[str, Any]]
    delete_ids: List[Any] = field(default_factory=list)
    links: List[WikiLink] = field(default_factory=list)


_EVENT_RETRYABLE_RESOLUTION_SKIP_CODES = {
    MemoryOperationSkipCode.INVALID_PEER_ID,
    MemoryOperationSkipCode.INVALID_RANGES,
    MemoryOperationSkipCode.AMBIGUOUS_TARGET,
    MemoryOperationSkipCode.NO_WRITABLE_TARGET,
}


_CANNED_REFUSAL_RE = re.compile(
    r"(抱歉|不好意思|很遗憾|sorry).{0,20}"
    r"(无法|不能|未能|不给|没有找到|未找到|can't|cannot|unable|won't|not able)",
    re.IGNORECASE,
)


def _looks_like_canned_refusal(content: str) -> bool:
    """Return whether a non-JSON LLM response looks like a generic refusal."""

    text = " ".join(str(content or "").split())
    if not text:
        return False
    if _CANNED_REFUSAL_RE.search(text):
        return True
    return any(
        phrase in text
        for phrase in (
            "您的问题我无法回答",
            "您的问题我无法识别",
            "我无法回答这个问题",
            "我无法给到相关内容",
            "这个问题未找到相关结果",
            "没有找到相关的结果",
            "I can't answer that",
            "I cannot answer that",
            "I can't help with that",
            "I cannot help with that",
        )
    )


def _preview_text(content: str, *, limit: int = 240) -> str:
    text = " ".join(str(content or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class ExtractLoop:
    """
    Simplified ReAct orchestrator for memory updates.

    Workflow:
    0. Pre-fetch: System performs ls + read .overview.md + search (via strategy)
    1. LLM call with tools: Model decides to either use tools OR output final operations
    2. If tools used: Execute and continue loop
    3. If operations output: Return and finish
    """

    def __init__(
        self,
        vlm: VLMBase,
        viking_fs: Optional[VikingFS] = None,
        model: Optional[str] = None,
        max_iterations: int = DEFAULT_EXTRACT_MAX_ITERATIONS,
        ctx: Optional[RequestContext] = None,
        context_provider: Optional[Any] = None,  # ExtractContextProvider
        isolation_handler: MemoryIsolationHandler = None,
        thinking: bool = False,
        max_output_tokens: Optional[int] = None,
    ):
        """
        Initialize the ExtractLoop.

        Args:
            vlm: VLM instance (from openviking.models.vlm.base)
            viking_fs: VikingFS instance for storage operations
            model: Model name to use
            max_iterations: Initial LLM iteration budget (default: 3). Existing tool-call
                and repair extensions may grant additional iterations.
            ctx: Request context
            context_provider: ExtractContextProvider - 必须提供（由 provider 加载 schema）
            thinking: Whether to explicitly enable model thinking for this extraction loop.
            max_output_tokens: Per-call output cap for the extraction LLM. When None, the
                configured vlm.max_tokens is used, falling back to a value large enough for
                full memory rewrites so providers do not apply a small server-side default
                (e.g. Ark's 4096) and truncate the program mid-string.
        """
        self.vlm = vlm
        self.viking_fs = viking_fs or get_viking_fs()
        self.model = model or self.vlm.model
        self.max_iterations = max_iterations
        self.ctx = ctx
        self.context_provider = context_provider
        self.thinking = bool(thinking)
        self.max_output_tokens = max_output_tokens
        # Resolved in run() via _resolve_effective_max_output_tokens(); default to
        # the extraction floor so a direct _call_llm() before run() stays safe.
        self._effective_max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else _DEFAULT_EXTRACTION_MAX_OUTPUT_TOKENS
        )
        # Use provided isolation_handler or create one in run()
        self._isolation_handler = isolation_handler
        # Track format error retry (max 1 retry)
        self._format_retry_count = 0
        self._last_llm_failure_kind: Optional[str] = None
        self._last_llm_failure_content: str = ""
        self._last_parse_error: Optional[str] = None
        # Preserve the raw assistant text that produced the most recent parsable
        # operations, so refetch-triggered follow-up turns can replay the prior
        # draft instead of asking the model to author it from scratch.
        self._last_operations_content: str = ""

        # Schema 生成器（在 run() 中初始化）
        self.schema_model_generator = None

        # 预计算：避免每次迭代重复计算
        self._tool_schemas: Optional[List[Dict[str, Any]]] = None
        self._operations_model: Optional[Any] = None
        self._output_protocol: Optional[ExtractionOutputProtocol] = None
        self._output_context: Optional[ExtractionOutputContext] = None

        # Transaction handle for file locking
        self._transaction_handle = None
        # Flag to disable tools in next iteration after unknown tool error
        self._disable_tools_for_iteration = False

        self._tool_ctx = None

    async def run(self) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        """
        Run the simplified ReAct loop for memory updates.

        Returns:
            Tuple of (final operations, tools_used list)
        """
        iteration = 0
        max_iterations = self.max_iterations
        tracer.set("memory.extraction.max_iterations", max_iterations)
        tracer.set("memory.extraction.thinking", self.thinking)
        final_operations = None
        raw_links = []
        tools_used: List[Dict[str, Any]] = []
        # Reset format retry counter for each run
        self._format_retry_count = 0
        patch_repair_count = 0
        reported_create_conflicts: set[tuple[str, str]] = set()
        provider_validation_repair_count = 0
        resolution_repair_count = 0
        pending_resolution_repair: Optional[Tuple[ResolvedOperations, List]] = None

        # 从 provider 获取 schemas（内部自动加载 registry）
        schemas = self.context_provider.get_memory_schemas(self.ctx)

        # 初始化 schema 生成器（使用 schemas 而非 registry）
        output_language = self.context_provider.get_output_language()
        self.schema_model_generator = SchemaModelGenerator(
            schemas,
            template_context={"language": output_language},
            # include_decision_reasoning=(
            #     type(self.context_provider).__name__ != "PatchMergeContextProvider"
            # ),
        )
        self.schema_model_generator.generate_all_models()

        # Resolve the provider's tool names before building the output context.
        allowed_tools = self.context_provider.get_tools()

        # Resolve global link support before generating the operations model.
        config = get_openviking_config()
        self._link_enabled = config.memory.link_enabled if config.memory else False

        self._resolve_effective_max_output_tokens()
        edit_repair = EditRepair(
            self.vlm, thinking=self.thinking, max_output_tokens=self._effective_max_output_tokens
        )

        # 获取 ExtractContext（整个流程复用）
        self._extract_context = self.context_provider.get_extract_context()
        if self._extract_context is None:
            raise ValueError("Failed to get ExtractContext from provider")

        # 预计算 operations_model
        role_scope = self._isolation_handler.get_read_scope() if self._isolation_handler else None

        self._operations_model = self.schema_model_generator.create_structured_operations_model(
            role_scope
        )

        output_format = getattr(config.memory, "extraction_output_format", "python")
        self._output_protocol = create_extraction_output_protocol(output_format)
        self._output_context = ExtractionOutputContext(
            operations_model=self._operations_model,
            schemas=tuple(schemas),
            page_id_map=self._extract_context.page_id_map,
            read_file_contents=self.context_provider.read_file_contents,
            link_enabled=self._link_enabled,
            role_scope=role_scope,
            available_tools=tuple(allowed_tools),
            template_context={"language": output_language},
        )
        configure_output = getattr(self.context_provider, "configure_output_context", None)
        if callable(configure_output):
            configure_output(self._output_context)
            self._operations_model = self._output_context.operations_model
        tracer.set("memory.extraction.output_format", output_format)

        # Build initial messages from provider
        messages = []
        reference_rules = self._output_protocol.render_reference_rules(self._output_context)
        provider_instruction = self._output_protocol.normalize_provider_instruction(
            self.context_provider.instruction()
        )
        output_contract = self._output_protocol.render_contract(self._output_context)
        messages.append(
            {
                "role": "system",
                "content": f"""
{provider_instruction}
{reference_rules}
## Read Format Rules
- Memory read functions accept optional `offset` (0-indexed) and `limit` arguments.
- SDK object fields are shown as plain text.
{output_contract}
        """,
            }
        )

        # Pre-fetch context via provider
        self.context_provider.memory_eval_bindings = None
        tool_call_messages = await self.context_provider.prefetch()
        prefetch_messages = self._output_protocol.render_prefetch_messages(
            tool_call_messages, self._output_context
        )
        messages.extend(self.context_provider.format_prefetch_messages(prefetch_messages))

        # Prefetch may discover or narrow runtime read scopes. Build model-facing
        # tool descriptions afterwards so they describe the same capabilities
        # that execution will enforce.
        self._tool_schemas = [
            self.context_provider.customize_tool_schema(tool.name, tool.to_schema())
            for tool in MEMORY_TOOLS_REGISTRY.values()
            if tool.name in allowed_tools
        ]

        for uri in self.context_provider.read_file_contents:
            self._extract_context.page_id_map.get_page_id(uri)

        while iteration < max_iterations:
            iteration += 1
            tracer.info(f"ReAct iteration {iteration}/{max_iterations}")

            # Check if this is the last iteration - force final result
            is_last_iteration = iteration >= max_iterations

            # If last iteration, add a message telling the model to return result directly
            if is_last_iteration:
                messages.append(
                    {
                        "role": "user",
                        "content": self._build_final_operations_instruction(),
                    }
                )

            # Call LLM with tools - model decides: tool calls OR final operations.
            # The VLM backend traces the (human-readable) input and output, so we do
            # not re-dump them here (avoids duplicate, oversized span events).
            tool_calls, operations = await self._call_llm(messages)

            if tool_calls:
                has_unknown_tool = await self._execute_tool_calls(messages, tool_calls, tools_used)
                # If model called an unknown tool, disable tools in next iteration
                if has_unknown_tool:
                    self._disable_tools_for_iteration = True
                    tracer.info("Unknown tool called, will disable tools in next iteration")
                # Allow one extra iteration for refetch
                if iteration >= max_iterations:
                    max_iterations += 1
                    self._disable_tools_for_iteration = True
                    tracer.info(f"Extended max_iterations to {max_iterations} for tool call")
                continue

            # If model returned final operations, check if refetch is needed
            if operations is not None:
                if pending_resolution_repair is not None:
                    operations = self._event_resolution_repair_subset(
                        operations,
                        pending_resolution_repair[0],
                    )
                candidate_operations, candidate_links = await self.resolve_operations(operations)
                if pending_resolution_repair is not None:
                    base_operations, base_links = pending_resolution_repair
                    final_operations = self._merge_event_resolution_repair(
                        base_operations,
                        candidate_operations,
                    )
                    raw_links = base_links
                    pending_resolution_repair = None
                else:
                    final_operations = candidate_operations
                    raw_links = candidate_links
                provider_errors = await self._validate_provider_operations(final_operations)
                if provider_errors:
                    repaired = await self._try_edit_repair(
                        edit_repair, final_operations, provider_errors
                    )
                    final_operations = repaired.operations
                    if repaired.repaired:
                        provider_errors = []
                if provider_errors:
                    if provider_validation_repair_count == 0:
                        provider_validation_repair_count += 1
                        max_iterations += 1
                        self._disable_tools_for_iteration = True
                        if self._last_operations_content:
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": self._last_operations_content,
                                }
                            )
                        feedback = self._build_provider_validation_repair_instruction(
                            provider_errors
                        )
                        if repaired.failure_reason:
                            feedback += "\n\nLocal Edit repair failed: " + repaired.failure_reason
                        messages.append({"role": "user", "content": feedback})
                        tracer.info(
                            "Extended max_iterations to "
                            f"{max_iterations} for provider validation repair"
                        )
                        continue
                    raise ValueError(
                        "Provider operation validation failed after repair: "
                        + json.dumps(provider_errors, ensure_ascii=False)
                    )
                resolution_issues = self._retryable_resolution_issues(final_operations)
                if resolution_issues and resolution_repair_count == 0:
                    resolution_repair_count += 1
                    pending_resolution_repair = (final_operations, raw_links)
                    max_iterations += 1
                    self._disable_tools_for_iteration = True
                    messages.append(
                        {
                            "role": "user",
                            "content": self._build_resolution_repair_instruction(resolution_issues),
                        }
                    )
                    tracer.info(
                        "Extended max_iterations to "
                        f"{max_iterations} for retry memory target resolution",
                        console=True,
                    )
                    continue
                try:
                    await validate_create_operations(final_operations, self.viking_fs, self.ctx)
                except MemoryCreateConflictError as exc:
                    conflict_keys = {
                        (memory_create_target_key(c["uri"], self.viking_fs, self.ctx), c["code"])
                        for c in exc.conflicts
                    }
                    new_conflicts = conflict_keys - reported_create_conflicts
                    known_read_targets = {
                        memory_create_target_key(uri, self.viking_fs, self.ctx)
                        for uri in self.context_provider.read_file_contents
                    }
                    # Internal reads are not model tool calls: the provider exposes
                    # eval only, so execute_tool(name="read") would reject them.
                    refetched = {}
                    for conflict in exc.conflicts:
                        if conflict["code"] != "already_exists":
                            continue
                        uri = conflict["uri"]
                        target_key = memory_create_target_key(uri, self.viking_fs, self.ctx)
                        if target_key in known_read_targets:
                            continue
                        result = await MEMORY_TOOLS_REGISTRY["read"].execute(
                            self.context_provider.create_tool_context(), uri=uri
                        )
                        if not isinstance(result, dict) or "error" in result:
                            raise exc
                        refetched[uri] = result
                        known_read_targets.add(target_key)
                    if refetched or new_conflicts:
                        # One bonus per round of new information, not per file or kind.
                        # An already-read target can still yield new conflict feedback.
                        max_iterations += 1
                        tracer.info(
                            f"Extended max_iterations to {max_iterations} for create repair "
                            f"({len(new_conflicts)} new conflicts, {len(refetched)} new reads)"
                        )
                    elif iteration >= max_iterations:
                        raise
                    if refetched:
                        await self._add_refetch_results_to_messages(messages, refetched)
                    elif self._last_operations_content:
                        messages.append(
                            {"role": "assistant", "content": self._last_operations_content}
                        )
                    messages.append({"role": "user", "content": str(exc)})
                    reported_create_conflicts.update(conflict_keys)
                    self._disable_tools_for_iteration = False
                    continue
                # Check if any write_uris target existing files that weren't read
                refetch_uris = await self._check_unread_existing_files(final_operations)
                if refetch_uris:
                    tracer.info(f"Found unread existing files: {refetch_uris}, refetching...")
                    # Add refetch results to messages and continue loop
                    await self._add_refetch_results_to_messages(messages, refetch_uris)
                    # A valid program started a new generation phase with richer context.
                    # Give that regenerated response its own format-repair opportunity.
                    self._format_retry_count = 0
                    # Allow one extra iteration for refetch
                    if iteration >= max_iterations:
                        max_iterations += 1
                        tracer.info(f"Extended max_iterations to {max_iterations} for refetch")

                    continue
                gate_errors = await self._validate_content_gates(final_operations)
                patch_errors = await self._validate_patch_operations(final_operations)
                if gate_errors or patch_errors:
                    repaired = await self._try_edit_repair(
                        edit_repair, final_operations, [*gate_errors, *patch_errors]
                    )
                    final_operations = repaired.operations
                    if repaired.repaired:
                        gate_errors = []
                        patch_errors = []
                if (gate_errors or patch_errors) and patch_repair_count == 0:
                    patch_repair_count += 1
                    max_iterations += 1
                    self._disable_tools_for_iteration = True
                    if self._last_operations_content:
                        messages.append(
                            {"role": "assistant", "content": self._last_operations_content}
                        )
                    repair_parts = []
                    if patch_errors:
                        repair_parts.append(self._build_patch_repair_instruction(patch_errors))
                    if gate_errors:
                        repair_parts.append(
                            "Memory content validation failed after applying the proposed edits. "
                            "Re-emit corrected operations. Errors: "
                            + json.dumps(gate_errors, ensure_ascii=False)
                        )
                    if repaired.failure_reason:
                        repair_parts.append("Local Edit repair failed: " + repaired.failure_reason)
                    messages.append({"role": "user", "content": "\n\n".join(repair_parts)})
                    tracer.info(
                        f"Extended max_iterations to {max_iterations} for retry patch repair"
                    )
                    continue
                if gate_errors:
                    self._discard_invalid_content_operations(final_operations, gate_errors)
                    tracer.info(f"Skipping invalid memory content after repair: {gate_errors}")
                break
            # If no tool calls either, continue to next iteration (don't break!)
            failure_kind = self._last_llm_failure_kind or "unknown"
            failure_preview = _preview_text(self._last_llm_failure_content)
            parse_error = self._last_parse_error
            # Grant only one extra attempt. Later failures still need fresh feedback
            # while the existing iteration budget allows another call.
            if self._format_retry_count == 0:
                self._format_retry_count += 1
                max_iterations += 1
                retry_reason = "refusal_text" if failure_kind == "refusal_text" else "format_retry"
                tracer.info(f"Extended max_iterations to {max_iterations} for {retry_reason}")

            # A failure is only terminal when no retry attempt remains after it.
            retry_remaining = iteration < max_iterations
            failure_message = (
                "Failed to parse memory operations "
                f"(iteration {iteration}/{max_iterations}) "
                f"failure_kind={failure_kind} error={parse_error} "
                f"response_preview={failure_preview!r}"
            )
            if retry_remaining:
                self._add_format_error_message(messages)
                # Recoverable: another attempt will run. Keep it in the trace only,
                # not the console/log stream.
                tracer.info(failure_message)
            else:
                # ERROR: no attempt remains to repair the invalid response.
                tracer.error(failure_message)

            # Preserve a valid first pass when only its resolution repair failed.
            # Otherwise propagate failure so the archive is not marked completed.
            if not retry_remaining:
                if pending_resolution_repair is not None:
                    final_operations, raw_links = pending_resolution_repair
                    pending_resolution_repair = None
                    tracer.info(
                        "Event resolution repair response could not be parsed; "
                        "keeping the first-pass operations",
                        console=True,
                    )
                    break
                raise RuntimeError(
                    "Final response could not be parsed as operations "
                    f"after {max_iterations} iterations "
                    f"(failure_kind={failure_kind})"
                )

            self._disable_tools_for_iteration = (
                not self._output_protocol.keep_tools_enabled_after_parse_error(
                    self._last_parse_error
                )
            )
            continue

        if final_operations is None:
            if iteration >= max_iterations:
                raise RuntimeError(f"Reached {max_iterations} iterations without completion")
            else:
                raise RuntimeError("ReAct loop completed but no operations generated")

        tracer.info(f"final_operations={final_operations.model_dump_json(indent=4)}")

        # Resolve links after the loop completes using the URIs already bound in resolve_operations().
        await self.finalize_operations(final_operations, raw_links)

        return final_operations, tools_used

    def _resolve_effective_max_output_tokens(self) -> None:
        """Resolve the extraction output cap.

        Priority: explicit per-loop value > resolved VLM max_tokens >
        _DEFAULT_EXTRACTION_MAX_OUTPUT_TOKENS. The default is a functional floor
        that suits the primary Doubao models; a model with a lower max output
        (e.g. gpt-4o-mini) must set vlm.max_tokens in ov.conf to override it.
        """
        if self.max_output_tokens is not None:
            self._effective_max_output_tokens = self.max_output_tokens
            return
        configured = getattr(self.vlm, "max_tokens", None)
        self._effective_max_output_tokens = (
            configured if configured is not None else _DEFAULT_EXTRACTION_MAX_OUTPUT_TOKENS
        )

    def _retryable_resolution_issues(self, operations: ResolvedOperations) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for operation in operations.upsert_operations:
            resolution_skip = operation.resolution_skip
            if (
                operation.uris
                or operation.memory_type != "events"
                or resolution_skip is None
                or resolution_skip.reason_code not in _EVENT_RETRYABLE_RESOLUTION_SKIP_CODES
            ):
                continue
            if resolution_skip.reason_code == MemoryOperationSkipCode.NO_WRITABLE_TARGET and (
                not self.ctx or not self.ctx.user
            ):
                continue
            issue = {
                "memory_type": operation.memory_type,
                "page_id": operation.page_id,
                "reason_code": resolution_skip.reason_code.value,
                "reason": resolution_skip.reason,
                "operation": operation.memory_fields,
            }
            issues.append(issue)
        return issues

    def _build_resolution_repair_instruction(self, issues: List[Dict[str, Any]]) -> str:
        return self._output_protocol.render_resolution_repair(issues)

    @staticmethod
    def _is_event_schema(schema: Any) -> bool:
        return (
            getattr(schema, "memory_type", None) == "events"
            and getattr(schema, "operation_mode", None) == "add_only"
        )

    def _memory_type_for_uri(self, uri: str) -> Optional[str]:
        memory_file = self.context_provider.read_file_contents.get(uri)
        return memory_file.memory_type if memory_file is not None else None

    @staticmethod
    def _is_retryable_event_operation(operation: ResolvedOperation) -> bool:
        return bool(
            operation.memory_type == "events"
            and not operation.uris
            and operation.resolution_skip is not None
            and operation.resolution_skip.reason_code in _EVENT_RETRYABLE_RESOLUTION_SKIP_CODES
        )

    @classmethod
    def _event_resolution_repair_subset(
        cls,
        operations: Any,
        original: ResolvedOperations,
    ) -> _EventRepairOperations:
        """Keep only one range repair for each failed Event and preserve its original fields."""
        failed_events = {
            operation.page_id: operation
            for operation in original.upsert_operations
            if cls._is_retryable_event_operation(operation) and operation.page_id is not None
        }
        candidates: Dict[int, List[Dict[str, Any]]] = {}
        for item in getattr(operations, "events", None) or []:
            item_dict = dict(item)
            page_id = item_dict.get("page_id")
            if page_id in failed_events:
                candidates.setdefault(page_id, []).append(item_dict)

        repaired_items: List[Dict[str, Any]] = []
        for page_id, failed_operation in failed_events.items():
            page_candidates = candidates.get(page_id, [])
            if len(page_candidates) != 1:
                if page_candidates:
                    logger.warning(
                        "Ignoring ambiguous Event repair response for page_id=%s: count=%s",
                        page_id,
                        len(page_candidates),
                    )
                continue

            candidate = page_candidates[0]
            expected_name = failed_operation.memory_fields.get("event_name")
            if candidate.get("event_name") != expected_name or candidate.get("ranges") is None:
                logger.warning(
                    "Ignoring mismatched Event repair response for page_id=%s",
                    page_id,
                )
                continue

            repaired_item = {
                key: value
                for key, value in failed_operation.memory_fields.items()
                if key not in {"memory_type", "user_id"}
            }
            repaired_item["ranges"] = candidate["ranges"]
            repaired_item["page_id"] = page_id
            repaired_items.append(repaired_item)

        return _EventRepairOperations(events=repaired_items)

    @classmethod
    def _merge_event_resolution_repair(
        cls,
        original: ResolvedOperations,
        repair: ResolvedOperations,
    ) -> ResolvedOperations:
        """Replace only failed Events; preserve every other first-pass operation."""
        failed_page_ids = {
            operation.page_id
            for operation in original.upsert_operations
            if cls._is_retryable_event_operation(operation)
        }
        failed_event_names = {
            operation.page_id: operation.memory_fields.get("event_name")
            for operation in original.upsert_operations
            if cls._is_retryable_event_operation(operation)
        }
        repaired_events: Dict[int, ResolvedOperation] = {}
        duplicate_page_ids = set()
        for operation in repair.upsert_operations:
            if (
                operation.memory_type != "events"
                or operation.page_id not in failed_page_ids
                or operation.memory_fields.get("event_name")
                != failed_event_names.get(operation.page_id)
            ):
                continue
            if operation.page_id in repaired_events:
                duplicate_page_ids.add(operation.page_id)
                continue
            repaired_events[operation.page_id] = operation
        for page_id in duplicate_page_ids:
            repaired_events.pop(page_id, None)

        merged_operations: List[ResolvedOperation] = []
        for operation in original.upsert_operations:
            if cls._is_retryable_event_operation(operation):
                repaired_operation = repaired_events.get(operation.page_id)
                merged_operations.append(repaired_operation or operation)
                continue
            merged_operations.append(operation)

        return original.model_copy(
            update={
                "upsert_operations": merged_operations,
            }
        )

    @staticmethod
    def _normalize_page_id_reference(
        raw_page_id: Optional[int],
        page_id_assignments: Dict[int, List[int]],
        page_id_map: Any,
    ) -> Tuple[Optional[int], bool]:
        """Return the normalized response ID and whether the reference is ambiguous."""
        if raw_page_id is None:
            return None, False
        assigned_ids = page_id_assignments.get(raw_page_id, [])
        if not assigned_ids:
            return raw_page_id, False

        unique_ids = list(dict.fromkeys(assigned_ids))
        existing_uri = page_id_map.resolve(raw_page_id) if page_id_map else None
        if len(unique_ids) != 1 or (existing_uri is not None and unique_ids[0] != raw_page_id):
            return None, True
        return unique_ids[0], False

    @classmethod
    def _normalize_operation_links(
        cls,
        raw_links: List[WikiLink],
        page_id_assignments: Dict[int, List[int]],
        page_id_map: Any,
    ) -> List[WikiLink]:
        if not raw_links or not page_id_assignments:
            return raw_links

        normalized_links: List[WikiLink] = []
        for link in raw_links:
            updates: Dict[str, int] = {}
            ambiguous = False
            for field_name in ("f", "t"):
                raw_page_id = getattr(link, field_name, None)
                normalized_page_id, is_ambiguous = cls._normalize_page_id_reference(
                    raw_page_id,
                    page_id_assignments,
                    page_id_map,
                )
                if is_ambiguous:
                    ambiguous = True
                    break
                if normalized_page_id != raw_page_id:
                    updates[field_name] = normalized_page_id
            if ambiguous:
                logger.warning(
                    "Skipping ambiguous memory link after page_id normalization: f=%s, t=%s",
                    link.f,
                    link.t,
                )
                continue
            normalized_links.append(link.model_copy(update=updates) if updates else link)
        return normalized_links

    def _assign_response_page_ids(
        self,
        operations: Any,
        schemas: List[Any],
        page_id_map: Any,
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[int, List[int]]]:
        """Assign unique IDs to new logical operations within one candidate response."""
        entries: List[Tuple[int, int, Optional[int], bool]] = []
        for schema_index, schema in enumerate(schemas):
            value = getattr(operations, schema.memory_type, None)
            if value is None:
                continue
            items = value if isinstance(value, list) else [value]
            is_event = self._is_event_schema(schema)
            for item_index, item in enumerate(items):
                requested_page_id = dict(item).get("page_id")
                entries.append((schema_index, item_index, requested_page_id, is_event))

        assigned_page_ids: Dict[Tuple[int, int], int] = {}
        page_id_assignments: Dict[int, List[int]] = {}
        page_id_allocator = None

        # Preserve non-Event behavior where possible. Events are always new and therefore
        # allocate after every other memory type has claimed its response-local ID.
        for event_phase in (False, True):
            for schema_index, item_index, requested_page_id, is_event in entries:
                if is_event != event_phase:
                    continue
                existing_uri = None
                if requested_page_id is not None and page_id_map is not None:
                    existing_uri = page_id_map.resolve(requested_page_id)
                if not is_event and existing_uri is not None:
                    page_id = requested_page_id
                else:
                    if page_id_allocator is None:
                        allocator_factory = getattr(
                            page_id_map,
                            "new_page_id_allocator",
                            None,
                        )
                        page_id_allocator = (
                            allocator_factory()
                            if callable(allocator_factory)
                            else ResponsePageIdAllocator()
                        )
                    page_id = page_id_allocator.allocate(requested_page_id)

                assigned_page_ids[(schema_index, item_index)] = page_id
                if requested_page_id is not None:
                    page_id_assignments.setdefault(requested_page_id, []).append(page_id)

        return assigned_page_ids, page_id_assignments

    async def resolve_operations(self, operations) -> tuple[ResolvedOperations, List]:
        upsert_operations: List[ResolvedOperation] = []
        delete_file_contents: List[MemoryFile] = []
        errors: List[str] = []
        invalid_reference_page_ids: set[int] = set()

        role_scope = self._isolation_handler.get_read_scope()
        page_id_map = getattr(self._extract_context, "page_id_map", None)
        schemas = list(self.context_provider.get_memory_schemas(self.ctx))
        assigned_page_ids, page_id_assignments = self._assign_response_page_ids(
            operations,
            schemas,
            page_id_map,
        )

        for schema_index, schema in enumerate(schemas):
            memory_type = schema.memory_type
            value = getattr(operations, memory_type, None)
            if value is None:
                continue

            items = value if isinstance(value, list) else [value]

            for item_index, item in enumerate(items):
                item_dict = dict(item)
                # Supply trusted provider fields before isolation and URI rendering.
                prepare_fields = getattr(self.context_provider, "prepare_operation_fields", None)
                if callable(prepare_fields):
                    item_dict = prepare_fields(memory_type, item_dict)
                item_dict["memory_type"] = memory_type
                identity_resolution_skip = None
                classify_identity_fields = getattr(
                    self._isolation_handler,
                    "_classify_identity_fields",
                    None,
                )
                if callable(classify_identity_fields):
                    identity_resolution_skip = classify_identity_fields(
                        item_dict,
                        memory_type_schema=schema,
                    )
                try:
                    self._isolation_handler.fill_identity_fields(
                        item_dict,
                        role_scope=role_scope,
                        memory_type_schema=schema,
                    )
                except TypeError as exc:
                    # Some tests and older custom isolation handlers implement
                    # the pre-schema signature ``fill_identity_fields(item,
                    # role_scope=None)``.  Keep that extension point working
                    # while the real handler can still use schema peer settings.
                    if "memory_type_schema" not in str(exc):
                        raise
                    self._isolation_handler.fill_identity_fields(
                        item_dict,
                        role_scope=role_scope,
                    )
                if not isinstance(identity_resolution_skip, MemoryOperationSkip):
                    identity_resolution_skip = None

                requested_page_id = item_dict.pop("page_id", None)
                page_id = assigned_page_ids[(schema_index, item_index)]
                is_event = self._is_event_schema(schema)
                resolved_op = ResolvedOperation(
                    create_only=bool(
                        self._output_context is not None
                        and requested_page_id in self._output_context.create_page_ids
                    ),
                    old_memory_file_content=None,
                    memory_fields=item_dict,
                    memory_type=memory_type,
                    uris=[],
                    page_id=page_id,
                    resolution_skip=identity_resolution_skip,
                )

                if is_event:
                    # Event page_ids are temporary link anchors. They must never select an
                    # existing memory file; URI collision handling remains unchanged downstream.
                    resolved_op.uris = self._isolation_handler.calculate_memory_uris(
                        memory_type_schema=schema,
                        operation=resolved_op,
                        extract_context=self._extract_context,
                    )
                elif requested_page_id is not None and page_id_map is not None:
                    resolved_uri = page_id_map.resolve(requested_page_id)
                    if resolved_uri:
                        old_content = self.context_provider.read_file_contents.get(resolved_uri)
                        existing_memory_type = self._memory_type_for_uri(resolved_uri)
                        if old_content is None or existing_memory_type != memory_type:
                            invalid_reference_page_ids.add(requested_page_id)
                            if old_content is None:
                                mismatch_reason = (
                                    f"page_id {page_id} points to an unread memory file; "
                                    f"cannot verify memory type {memory_type}"
                                )
                            elif existing_memory_type is None:
                                mismatch_reason = (
                                    f"page_id {page_id} has no memory_type metadata; "
                                    f"cannot verify memory type {memory_type}"
                                )
                            else:
                                mismatch_reason = (
                                    f"page_id {page_id} belongs to memory type "
                                    f"{existing_memory_type}, not {memory_type}"
                                )
                            resolved_op.resolution_skip = MemoryOperationSkip(
                                reason_code=MemoryOperationSkipCode.PAGE_ID_TYPE_MISMATCH,
                                reason=mismatch_reason,
                            )
                        else:
                            resolved_op.uris = [resolved_uri]
                            # Existing page IDs retain their historical precedence over
                            # an invalid peer hint. Keep the hint runtime-only and do
                            # not persist it into memory metadata.
                            resolved_op.resolution_skip = None
                        if old_content is not None and resolved_op.uris:
                            resolved_op.old_memory_file_content = old_content
                            immutable_fields = {
                                field.name
                                for field in schema.fields
                                if field.merge_op == MergeOp.IMMUTABLE
                            }
                            for field_name in immutable_fields:
                                if ImmutableOp.is_set(old_content.extra_fields.get(field_name)):
                                    resolved_op.memory_fields[field_name] = (
                                        old_content.extra_fields[field_name]
                                    )
                    else:
                        resolved_op.uris = self._isolation_handler.calculate_memory_uris(
                            memory_type_schema=schema,
                            operation=resolved_op,
                            extract_context=self._extract_context,
                        )
                else:
                    resolved_op.uris = self._isolation_handler.calculate_memory_uris(
                        memory_type_schema=schema,
                        operation=resolved_op,
                        extract_context=self._extract_context,
                    )

                upsert_operations.append(resolved_op)

        delete_ids = self._normalize_delete_ids(getattr(operations, "delete_ids", []) or [])
        delete_replacements: dict[str, str] = {}
        for delete_id in delete_ids:
            if delete_id.delete_page_id is None or page_id_map is None:
                continue
            delete_uri = page_id_map.resolve(delete_id.delete_page_id)
            if not delete_uri:
                continue
            old_content = self.context_provider.read_file_contents.get(delete_uri)
            if not old_content:
                continue
            replacement_page_id = delete_id.replacement_page_id
            replacement_uri = None
            if replacement_page_id is not None:
                if replacement_page_id in invalid_reference_page_ids:
                    logger.warning(
                        "Skipping delete with type-mismatched replacement page_id: "
                        "delete_page_id=%s, replacement_page_id=%s",
                        delete_id.delete_page_id,
                        delete_id.replacement_page_id,
                    )
                    continue
                replacement_page_id, ambiguous = self._normalize_page_id_reference(
                    replacement_page_id,
                    page_id_assignments,
                    page_id_map,
                )
                if ambiguous:
                    logger.warning(
                        "Skipping delete with ambiguous replacement page_id: "
                        "delete_page_id=%s, replacement_page_id=%s",
                        delete_id.delete_page_id,
                        delete_id.replacement_page_id,
                    )
                    continue
                replacement_uri = page_id_map.resolve(replacement_page_id)
                if not replacement_uri:
                    for op in upsert_operations:
                        if op.page_id == replacement_page_id and op.uris:
                            replacement_uri = op.uris[0]
                            break

            delete_file_contents.append(old_content)
            if replacement_uri and replacement_uri != delete_uri:
                delete_replacements[delete_uri] = replacement_uri

        case_references = []
        case_schema = next((schema for schema in schemas if schema.memory_type == "cases"), None)
        for selected_id in dict.fromkeys(getattr(operations, "selected_case_ids", []) or []):
            uri = page_id_map.resolve(selected_id) if page_id_map is not None else None
            memory = self.context_provider.read_file_contents.get(uri)
            if case_schema is None or memory is None or self._memory_type_for_uri(uri) != "cases":
                raise ValueError(f"Selected Case page_id {selected_id} is not a loaded Case")
            if any(file.uri == uri for file in delete_file_contents):
                raise ValueError(f"Selected Case page_id {selected_id} is deleted by this response")
            fields = {
                field.name: memory.extra_fields[field.name]
                for field in case_schema.fields
                if field.name in memory.extra_fields
            }
            reference = ResolvedOperation(
                memory_type="cases",
                memory_fields=fields,
                uris=[],
                old_memory_file_content=memory,
                page_id=selected_id,
            )
            allowed_uris = self._isolation_handler.calculate_memory_uris(
                memory_type_schema=case_schema,
                operation=reference,
                extract_context=self._extract_context,
            )
            if uri not in allowed_uris:
                raise ValueError(
                    f"Selected Case page_id {selected_id} is outside the allowed Case scope"
                )
            reference.uris = [uri]
            case_references.append(reference)

        raw_links = getattr(operations, "links", None) or []
        if invalid_reference_page_ids:
            raw_links = [
                link
                for link in raw_links
                if link.f not in invalid_reference_page_ids
                and link.t not in invalid_reference_page_ids
            ]
        raw_links = self._normalize_operation_links(
            raw_links,
            page_id_assignments,
            page_id_map,
        )
        resolved = ResolvedOperations(
            case_references=case_references,
            upsert_operations=upsert_operations,
            delete_file_contents=delete_file_contents,
            errors=errors,
            delete_replacements=delete_replacements,
        )

        for op in upsert_operations:
            for uri in op.uris:
                old_content = self.context_provider.read_file_contents.get(uri)
                if old_content and op.old_memory_file_content is None and not op.create_only:
                    op.old_memory_file_content = old_content
                    break

        resolve_metadata = getattr(self.context_provider, "resolve_operation_metadata", None)
        if callable(resolve_metadata):
            resolve_metadata(operations, resolved)
        return resolved, raw_links

    def _normalize_delete_ids(self, raw_delete_ids: List[Any]) -> List[DeleteId]:
        delete_ids: List[DeleteId] = []
        for raw in raw_delete_ids:
            try:
                delete_ids.append(DeleteId.model_validate(raw))
            except Exception as e:
                tracer.info(f"Skipping invalid delete_ids item: {raw}, error={e}")
        return delete_ids

    async def finalize_operations(self, operations: ResolvedOperations, raw_links: List) -> None:
        """Register new page_ids and resolve links after refetch is complete.

        Must be called after resolve_operations() and any refetch rounds,
        so that existing files discovered by refetch get 1-99 page_ids
        instead of 100+ IDs from register_new_page_id.
        """
        if not self._link_enabled:
            return

        upsert_operations = operations.upsert_operations

        # URIs are already bound in resolve_operations() before any refetch rounds.
        # finalize_operations only consumes them to register new page_ids and resolve links.
        page_id_map = self._extract_context.page_id_map

        # Register new page_ids (100+) after URI resolution, using LLM-declared page_id
        for op in upsert_operations:
            if op.page_id is not None and op.page_id >= 100:
                for uri in op.uris:
                    page_id_map.register_new_page_id(uri, op.page_id)

        # Resolve links from WikiLink (page_ids) to StoredLink (URIs)
        resolved_links = self._resolve_links(raw_links, upsert_operations)

        operations.resolved_links = resolved_links

    def _resolve_links(self, raw_links: List, upsert_operations: List = None) -> List[StoredLink]:
        """Resolve WikiLinks with page_ids to StoredLinks with URIs.

        Returns a flat list of StoredLink objects. Each link is stored once.
        Links go into from_uri's "links" field; backlinks go into to_uri's "backlinks" field.
        The routing is handled by memory_updater based on which file each link belongs to.
        """
        if not raw_links:
            return []

        op_page_map = {}
        if upsert_operations:
            for op in upsert_operations:
                if op.page_id is None or not op.uris:
                    continue
                op_page_map.setdefault(op.page_id, [])
                for uri in op.uris:
                    if uri not in op_page_map[op.page_id]:
                        op_page_map[op.page_id].append(uri)

        page_id_map = self._extract_context.page_id_map

        if not page_id_map._id_to_uri and not op_page_map:
            return []
        page_uri_map = {page_id: list(uris) for page_id, uris in op_page_map.items()}
        for link in raw_links:
            for page_id in (link.f, link.t):
                if page_id is None:
                    continue
                uri = page_id_map.resolve(page_id)
                if uri:
                    page_uri_map.setdefault(page_id, [])
                    if uri not in page_uri_map[page_id]:
                        page_uri_map[page_id].insert(0, uri)

        from openviking.session.memory.utils.link_resolver import resolve_wiki_links

        return resolve_wiki_links(raw_links, page_uri_map)

    @tracer("extract_loop.execute_tool_calls")
    async def _execute_tool_calls(self, messages, tool_calls, tools_used) -> bool:
        """
        Execute tool calls in parallel.

        Returns:
            True if any tool call returned "Unknown tool" error, indicating
            the model should not receive tools in the next iteration.
        """

        # Parallel calls see the same already-presented bindings, not each other's results.
        self.context_provider.memory_eval_bindings = self._output_protocol.get_memory_eval_bindings(
            self._output_context
        )

        # Execute all tool calls in parallel
        async def execute_single_tool_call(idx: int, tool_call):
            """Execute a single tool call."""
            result = await self.context_provider.execute_tool(tool_call)
            return idx, tool_call, result

        action_tasks = [
            execute_single_tool_call(idx, tool_call) for idx, tool_call in enumerate(tool_calls)
        ]
        results = await asyncio.gather(*action_tasks)

        has_unknown_tool = False
        request_calls = []
        result_messages = []

        # Keep parallel calls in one assistant turn, followed by all tool replies.
        for _idx, tool_call, result in results:
            # Check for unknown tool error
            error = result.get("error", "") if isinstance(result, dict) else ""
            if isinstance(error, str) and error.startswith(("Unknown tool:", "Tool not allowed:")):
                has_unknown_tool = True
            tools_used.append(
                {
                    "tool_name": tool_call.name,
                    "params": tool_call.arguments,
                    "result": result,
                }
            )

            request, reply = self._output_protocol.render_tool_result_messages(
                self._output_context,
                call_id=tool_call.id,
                tool_name=tool_call.name,
                params=tool_call.arguments,
                result=result,
                source="tool call",
            )
            request_calls.extend(request["tool_calls"])
            result_messages.append(reply)

        if request_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": request_calls})
        messages.extend(result_messages)
        return has_unknown_tool

    async def _call_llm(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[List], Optional[Any]]:
        """
        Call LLM with tools. Returns either tool calls OR final operations.

        Args:
            messages: Message list
            force_final: If True, force model to return final result (not tool calls)

        Returns:
            Tuple of (tool_calls, operations) - one will be None, the other set
        """
        # Call LLM with tools - use tools from strategy
        tools = None
        tool_choice = None
        if not self._disable_tools_for_iteration and self._tool_schemas:
            tools = self._tool_schemas
            tool_choice = "auto"
        # Refresh only at the request boundary so snapshots never accumulate in history.
        request_messages = messages
        if self._output_protocol is not None:
            binding_index = self._output_protocol.render_binding_index()
            if binding_index:
                request_messages = [*messages, {"role": "user", "content": binding_index}]
        with bind_telemetry_stage("memory_extract"):
            response = await self.vlm.get_completion_async(
                messages=request_messages,
                tools=tools,
                tool_choice=tool_choice,
                thinking=self.thinking,
                max_tokens=self._effective_max_output_tokens,
            )
        self._last_llm_failure_kind = None
        self._last_llm_failure_content = ""
        self._last_parse_error = None
        # print(f'response={response}')
        # Log cache hit info
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            cached_tokens = (
                usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                if isinstance(usage.get("prompt_tokens_details"), dict)
                else 0
            )
            try:
                from openviking.metrics.datasources.cache import CacheEventDataSource

                if int(cached_tokens or 0) > 0:
                    CacheEventDataSource.record_hit("L2")
                else:
                    CacheEventDataSource.record_miss("L2")
            except Exception:
                pass

        # Case 0: Handle string response (when tools are not provided) or None
        if response is None:
            content = ""
        elif isinstance(response, str):
            # When tools=None, VLM returns string instead of VLMResponse
            content = response
        # Case 1: LLM returned tool calls
        elif response.has_tool_calls:
            self._trace_model_output(
                {"tool_calls": [{"id": tc.id, "name": tc.name} for tc in response.tool_calls]}
            )
            return (response.tool_calls, None)
        else:
            # Case 2: VLMResponse without tool calls - get content from response
            content = response.content or ""

        self._trace_model_output({"content": content})

        # Parse operations from content
        if content:
            try:
                # print(f'LLM response content: {content}')
                logger.debug(f"[assistant]\n{content}")

                operations, error = self._output_protocol.parse(content, self._output_context)

                if error is not None:
                    failure_kind = (
                        "refusal_text" if _looks_like_canned_refusal(content) else "parse_error"
                    )
                    self._last_llm_failure_kind = failure_kind
                    self._last_llm_failure_content = content
                    self._last_parse_error = error
                    tracer.set("memory.extraction.parse_error", error)
                    # Diagnostic only; the run loop emits the WARNING/ERROR decision.
                    logger.debug(
                        "Failed to parse memory operations "
                        f"failure_kind={failure_kind} error={error} "
                        f"response_preview={_preview_text(content)!r}"
                    )
                    return (None, None)

                self._last_operations_content = content
                return (None, operations)
            except Exception as e:
                self._last_parse_error = str(e)
                tracer.set("memory.extraction.parse_error", self._last_parse_error)
                logger.debug(f"Error parsing operations: {e}")

        # Case 3: No tool calls and no parsable operations
        self._last_llm_failure_kind = "empty_response" if not content else "parse_error"
        self._last_llm_failure_content = content or ""
        if not content:
            empty_error = self._output_protocol.describe_empty_response()
            if empty_error:
                self._last_parse_error = empty_error
                tracer.set("memory.extraction.parse_error", empty_error)
        logger.debug(
            "No tool calls or operations parsed "
            f"failure_kind={self._last_llm_failure_kind} "
            f"response_preview={_preview_text(self._last_llm_failure_content)!r}"
        )
        return (None, None)

    @staticmethod
    def _trace_model_output(payload: Dict[str, Any]) -> None:
        with tracer.start_as_current_span("memory.model.output"):
            tracer.info(
                "output="
                + json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
            )

    async def _check_unread_existing_files(self, operations: ResolvedOperations) -> Dict:
        refetch_uris = {}
        for operation in operations.upsert_operations:
            for uri in operation.uris:
                if uri in self.context_provider.read_file_contents:
                    continue
                try:
                    content = await self.context_provider.execute_tool(
                        ToolCall(id="", name="read", arguments={"uri": uri})
                    )
                    # 读取出错表示文件不存在（error dict 含 "error" key）
                    if isinstance(content, Dict) and "error" in content:
                        continue

                    # execute_tool(MemoryReadTool) 已经返回 parsed dict，直接使用
                    refetch_uris[uri] = content
                except Exception as e:
                    tracer.error("read tool execute fail", e)
        return refetch_uris

    def _add_format_error_message(self, messages: List[Dict[str, Any]]) -> None:
        """Add format error guidance message to prompt."""
        messages.append(
            {
                "role": "user",
                "content": self._output_protocol.render_format_retry(
                    self._last_parse_error, failed_content=self._last_llm_failure_content
                ),
            }
        )

    def _build_final_operations_instruction(self) -> str:
        """Build schema-aware final-iteration instructions for the LLM."""
        return self._output_protocol.render_final_instruction(self._output_context)

    async def _try_edit_repair(
        self, repairer: EditRepair, operations: ResolvedOperations, errors: List[Dict[str, Any]]
    ) -> EditRepairResult:
        result = await repairer.repair(
            operations, schemas=self._output_context.schemas, errors=errors
        )
        if not result.repaired:
            return result
        # Accept the copy only after all gates pass. Otherwise the original SDK
        # draft and operations remain consistent for the existing full-loop retry.
        validation_errors = [
            *await self._validate_content_gates(result.operations),
            *await self._validate_patch_operations(result.operations),
        ]
        if not validation_errors:
            validation_errors = await self._validate_provider_operations(result.operations)
        if validation_errors:
            reason = "candidate_validation_failed: " + json.dumps(
                validation_errors, ensure_ascii=False
            )
            tracer.info("Local Edit repair rejected; retaining original operations: " + reason)
            return EditRepairResult(operations, failure_reason=reason)
        return result

    async def _validate_content_gates(self, operations: ResolvedOperations) -> List[Dict[str, Any]]:
        from openviking.session.memory.streaming_memory_updater import (
            clone_operation_for_uri,
            render_operation_after_file,
        )
        from openviking.session.memory.validation.content import get_field_gates

        schemas = {
            schema.memory_type: schema
            for schema in self.context_provider.get_memory_schemas(self.ctx)
        }
        errors = []
        staged = {}
        for operation in operations.upsert_operations:
            if not get_field_gates(operation.memory_type):
                continue
            for uri in operation.uris:
                try:
                    preview = clone_operation_for_uri(operation, uri)
                    if uri in staged:
                        preview.old_memory_file_content = staged[uri]
                    staged[uri] = await render_operation_after_file(
                        preview,
                        schema=schemas[operation.memory_type],
                        extract_context=self._extract_context,
                    )
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append({"uri": uri, "page_id": operation.page_id, "error": str(exc)})
        return errors

    @staticmethod
    def _discard_invalid_content_operations(
        operations: ResolvedOperations, errors: List[Dict[str, Any]]
    ) -> None:
        invalid_uris = {str(error.get("uri") or "") for error in errors}
        invalid_page_ids = {
            error["page_id"] for error in errors if error.get("page_id") is not None
        }
        operations.upsert_operations = [
            op
            for op in operations.upsert_operations
            if not (set(op.uris) & invalid_uris) and op.page_id not in invalid_page_ids
        ]
        # Never execute destructive follow-ups to an invalid candidate. Valid
        # independent upserts can still proceed after the repair opportunity.
        operations.delete_file_contents = []
        operations.delete_replacements = {}
        operations.resolved_links = [
            link
            for link in operations.resolved_links
            if link.from_uri not in invalid_uris and link.to_uri not in invalid_uris
        ]

    async def _validate_patch_operations(
        self,
        operations: ResolvedOperations,
    ) -> List[Dict[str, Any]]:
        from openviking.session.memory.merge_op.base import StrPatch
        from openviking.session.memory.merge_op.patch_handler import PatchParseError

        errors = []
        patch_op = PatchOp(FieldType.STRING)
        read_files = self.context_provider.read_file_contents or {}
        for operation in operations.upsert_operations:
            previous = operation.old_memory_file_content
            if previous is None:
                continue
            target_uri = operation.uris[0] if operation.uris else previous.uri
            for field_name, patch_value in operation.memory_fields.items():
                if isinstance(patch_value, dict) and "blocks" in patch_value:
                    patch_value = StrPatch.model_validate(patch_value)
                if not isinstance(patch_value, StrPatch):
                    continue
                current = (
                    previous.plain_content()
                    if field_name == "content"
                    else previous.extra_fields.get(field_name)
                )
                try:
                    await patch_op.apply(current, patch_value, path=f"{target_uri}#{field_name}")
                except PatchParseError as exc:
                    index = exc.edit_index
                    block = patch_value.blocks[index] if index is not None else None
                    search = block.search if block is not None else ""
                    found_in = []
                    for uri, memory_file in read_files.items():
                        other = (
                            memory_file.plain_content()
                            if field_name == "content"
                            else memory_file.extra_fields.get(field_name)
                        )
                        if (
                            uri != target_uri
                            and search
                            and isinstance(other, str)
                            and search in other
                        ):
                            found_in.append(uri)
                    errors.append(
                        {
                            "uri": target_uri,
                            "page_id": operation.page_id,
                            "field": field_name,
                            "block_index": index + 1 if index is not None else None,
                            "search": search,
                            "replace": block.replace if block is not None else None,
                            "reason": exc.reason,
                            "match_count": exc.match_count,
                            "error": str(exc),
                            "found_in_other_uris": found_in,
                        }
                    )
        if errors:
            tracer.info(f"String patch validation failed before apply: {errors}")
        return errors

    def _build_patch_repair_instruction(self, patch_errors: List[Dict[str, Any]]) -> str:
        return self._output_protocol.render_patch_repair(patch_errors)

    async def _validate_provider_operations(
        self, operations: ResolvedOperations
    ) -> List[Dict[str, Any]]:
        validate = getattr(self.context_provider, "validate_resolved_operations", None)
        if not callable(validate):
            return []
        result = validate(operations)
        return list(await result if inspect.isawaitable(result) else result)

    def _build_provider_validation_repair_instruction(self, errors: List[Dict[str, Any]]) -> str:
        render = getattr(self.context_provider, "render_operation_repair", None)
        if callable(render):
            return str(render(errors))
        return (
            "The operations failed provider validation. Re-emit the complete operations only. "
            "Errors: " + json.dumps(errors, ensure_ascii=False)
        )

    async def _add_refetch_results_to_messages(
        self,
        messages: List[Dict[str, Any]],
        refetch_uris: Dict[str, Any],
    ) -> None:
        """Add existing file content as read tool results to messages.

        When the ReAct loop refetches existing files after the model already
        produced parsable operations, replay the model's previous draft as an
        assistant message so the follow-up turn can revise it instead of
        starting over. Otherwise the model tends to treat the prior operations
        as "already delivered" and emits an empty set on the next turn.
        """
        prior_operations_content = self._last_operations_content or ""
        if prior_operations_content:
            messages.append(
                {
                    "role": "assistant",
                    "content": prior_operations_content,
                }
            )

        # Keep every prior binding valid across supplemental reads, including
        # names the model may have used in its own eval calls or previous draft.
        used_names = set(
            re.findall(
                r"\brefetched_memory_\d+\b",
                json.dumps(messages, ensure_ascii=False, default=str),
            )
        )
        index = 1
        for uri, parsed in refetch_uris.items():
            while f"refetched_memory_{index}" in used_names:
                index += 1
            variable = f"refetched_memory_{index}"
            used_names.add(variable)
            messages.extend(
                self._output_protocol.render_tool_result_messages(
                    self._output_context,
                    call_id=f"call_memory_{uuid4().hex}",
                    tool_name="eval",
                    params={"code": f"{variable} = sdk.read(uri={uri!r})"},
                    result={variable: parsed},
                    source="automatic read",
                )
            )

        if prior_operations_content:
            reminder = (
                "Note: The files above were automatically read because they exist and "
                "you didn't read them before deciding to write. Re-emit your previous "
                "draft above as ONE complete operations payload, revising only the "
                "entries affected by the newly read content. Do not drop unaffected "
                "entries and do not return an empty payload."
            )
        else:
            reminder = (
                "Note: The files above were automatically read because they exist and "
                "you didn't read them before deciding to write. Please consider the "
                "existing content when making write decisions. You can now output "
                "updated operations."
            )
        messages.append(
            {
                "role": "user",
                "content": reminder,
            }
        )
