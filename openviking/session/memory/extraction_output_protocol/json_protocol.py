# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Original JSON output protocol for memory extraction."""

from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter, ValidationError

from openviking.session.memory.extraction_output_protocol.base import (
    ExtractionOutputContext,
    ExtractionOutputProtocol,
)
from openviking.session.memory.extraction_output_protocol.edit_format import EDIT_TOOL_DESCRIPTION
from openviking.session.memory.tools import add_tool_call_pair_to_messages
from openviking.session.memory.utils import (
    parse_json_with_stability,
    split_content_lines,
    strip_line_numbers,
)


def _strip_numbered_read_copy(content: str) -> str:
    stripped = strip_line_numbers(content)
    if stripped == content:
        return content

    line_pairs = zip(split_content_lines(content), split_content_lines(stripped), strict=True)
    nonempty_pairs = [(original, clean) for original, clean in line_pairs if original]
    if len(nonempty_pairs) >= 2 and all(original != clean for original, clean in nonempty_pairs):
        return stripped
    if nonempty_pairs and all(clean.startswith("- ") for _, clean in nonempty_pairs):
        return stripped
    return content


def _strip_numbered_read_copies(operations: Any, context: ExtractionOutputContext) -> None:
    """Keep read-tool display prefixes out of complete memory content values."""
    for memory_type in getattr(context.operations_model, "_memory_type_fields", ()):
        for operation in getattr(operations, memory_type, ()) or ():
            content = getattr(operation, "content", None)
            if isinstance(content, str):
                operation.content = _strip_numbered_read_copy(content)


class JsonExtractionOutputProtocol(ExtractionOutputProtocol):
    """Preserve the existing JSON prompt and tolerant parsing behavior."""

    name = "json"

    def render_prefetch_messages(
        self, messages: list[dict[str, Any]], context: ExtractionOutputContext
    ) -> list[dict[str, Any]]:
        # Register the named results while preserving the original JSON transport verbatim.
        super().render_prefetch_messages(messages, context)
        return list(messages)

    def render_tool_result_messages(
        self,
        context: ExtractionOutputContext,
        *,
        call_id: str | int,
        tool_name: str,
        params: dict[str, Any],
        result: Any,
        source: str,
    ) -> list[dict[str, Any]]:
        del source
        messages: list[dict[str, Any]] = []
        add_tool_call_pair_to_messages(
            messages,
            call_id=call_id,
            tool_name=tool_name,
            params=params,
            result=result,
        )
        if not (isinstance(result, dict) and isinstance(result.get("error"), str)):
            bindings = (
                result if tool_name == "eval" and isinstance(result, dict) else {tool_name: result}
            )
            context.memory_eval_bindings.update(bindings)
        return messages

    def render_contract(self, context: ExtractionOutputContext) -> str:
        schema = json.dumps(context.operations_model.model_json_schema(), ensure_ascii=False)
        return (
            "## Output Format\n"
            "The final output of the model must strictly follow the JSON Schema format shown below:\n"
            f"```json\n{schema}\n```\n\n## String field edits\n{EDIT_TOOL_DESCRIPTION}"
        )

    def render_reference_rules(self, context: ExtractionOutputContext) -> str:
        rules = """
## Page ID Rules
- Every memory item you create or edit MUST include "page_id".
- For existing items, use the page_id shown in read/search results.
- For new items, assign a unique page_id >= 100.
- When editing an existing item, reuse its existing page_id.
- To delete an existing item, add an entry to `delete_ids` using its page_id.
- `delete_ids` deletes the whole item: use it only if every substantive fact is in scope; otherwise MUST use DELETE blocks for affected lines, preserving the rest and not inferring scope from the file name/topic.
- For canonical merges, set `replacement_page_id` to the surviving page that should inherit the deleted page's existing links/backlinks; for pure deletes, set `replacement_page_id` to null.
"""
        if context.link_enabled:
            rules += """
## Link Rules
- Link fields `f` and `t` must reference these page_id values.
- Only create links when the relationship is meaningful and clear from the conversation. Do NOT force links between unrelated items.
"""
        return rules

    def parse(
        self, content: str, context: ExtractionOutputContext
    ) -> tuple[Any | None, str | None]:
        if "selected_case_ids" in context.operations_model.model_fields:
            # The tolerant parser coerces booleans/strings to integers and drops invalid
            # items. Never let those repairs silently change which Case was selected.
            raw, _ = parse_json_with_stability(content=content)
            if raw is not None and "selected_case_ids" in raw:
                selected = raw["selected_case_ids"]
                if not isinstance(selected, list) or any(
                    type(item) is not int for item in selected
                ):
                    return None, "selected_case_ids must be a list of integer page_ids"
            if raw is not None and "case_sources" in context.operations_model.model_fields:
                # Source patch numbers must not be repaired into different associations.
                if not isinstance(raw, dict):
                    return None, "Case merge output must be an object with case_sources"
                try:
                    TypeAdapter(
                        context.operations_model.model_fields["case_sources"].annotation
                    ).validate_python(raw.get("case_sources", []))
                except ValidationError as exc:
                    return None, str(exc)
        operations, error = parse_json_with_stability(
            content=content,
            model_class=context.operations_model,
            expected_fields=list(context.operations_model.model_fields),
        )
        if operations is not None:
            _strip_numbered_read_copies(operations, context)
        return operations, error

    def render_final_instruction(self, context: ExtractionOutputContext) -> str:
        if "case_sources" in context.operations_model.model_fields:
            return (
                "Return the complete operations JSON now, without tool calls. Include "
                "case_sources entries with case_id and patch_ids for every final Case, including "
                "unchanged existing Cases. Cover every proposed Case patch. Output JSON only."
            )
        fields = ["delete_ids", *context.operations_model.model_fields]
        skeleton = json.dumps(
            {name: [] for name in dict.fromkeys(fields)},
            ensure_ascii=False,
            indent=2,
        )
        return (
            "You have reached the maximum number of tool call iterations. "
            "Do not call any more tools. Return your final result now as ONLY a valid JSON object "
            "matching the required schema. Do not include explanations or markdown. "
            + (
                "Include matched existing Cases in selected_case_ids even when no content changes are needed. "
                if "selected_case_ids" in context.operations_model.model_fields
                else ""
            )
            + "If there are no changes or selections, return this empty-shape JSON with all fields present:\n"
            + skeleton
        )

    def render_format_retry(self, error: str | None = None, *, failed_content: str = "") -> str:
        del error, failed_content
        return (
            "Your previous output could not be parsed as valid JSON. "
            "Please output ONLY a valid JSON object matching the required schema. "
            "Do not include any explanation, markdown formatting, or text outside the JSON."
        )

    def render_patch_repair(self, patch_errors: list[dict[str, Any]]) -> str:
        details = json.dumps(patch_errors, ensure_ascii=False, indent=2)
        return (
            "A SEARCH/REPLACE or DELETE edit could not be applied to its target field. "
            "Follow the Edit instructions in the initial output contract. "
            "Regenerate the complete operations JSON, preserving other intended operations "
            "and source associations. Do not move the edit to another page merely because "
            "its text matches.\n\nFailed patch operations:\n" + details
        )

    def render_resolution_repair(self, issues: list[dict[str, Any]]) -> str:
        details = json.dumps(issues, ensure_ascii=False, indent=2)
        return (
            "Some event operations could not resolve a safe write target. "
            "Return only corrected event operations for the failed items below; leave every "
            "other memory-type field, delete_ids, and links empty. The server has preserved "
            "all successful operations from the previous response. "
            "Reuse exactly the page_id shown for each failed event. "
            "For event ranges, use valid in-bounds message indexes and include the user-role "
            "message that establishes the event so its owner can be resolved. "
            "Do not target a disallowed or ambiguous peer. Output ONLY one JSON object matching "
            "the required schema.\n\n"
            f"Resolution issues:\n{details}"
        )
