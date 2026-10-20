# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Context provider for merging structured memory-file patches via ExtractLoop."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from openviking.server.identity import RequestContext
from openviking.session.memory.dataclass import MemoryFile, MemoryTypeSchema, ResolvedOperations
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.merge_policy import MEMORY_MERGE_POLICY
from openviking.session.memory.session_extract_context_provider import (
    SessionExtractContextProvider,
)
from openviking.session.memory.tools import add_tool_call_pair_to_messages
from openviking.session.memory.utils.language import resolve_output_language_from_text

if TYPE_CHECKING:
    from openviking.config.vlm import VLMHandle

_SYSTEM_HIDDEN_FIELDS = {
    "action_feedback",
    "action_score_sources",
    # Relations are maintained by storage, not reconciled as model-editable fields.
    "links",
    "backlinks",
    "feedback_stats",
    "experience_effect_sources",
    "source_extraction_id",
    "source_extraction_ids",
    "last_update_trace_id",
}
_MAX_EXTRA_CANDIDATE_FILES = 10
_PATCH_METADATA_KEYS = ("confidence",)


class _CaseSources(BaseModel):
    """Response-local patch numbers, never persisted Case fields."""

    model_config = ConfigDict(extra="forbid")
    case_id: StrictInt
    patch_ids: list[StrictInt] = Field(min_length=1)


@dataclass(slots=True)
class PatchMergePatch:
    """A file proposal: absent before means create; absent after means delete."""

    before_file: MemoryFile | None
    after_file: MemoryFile | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_uri(self) -> str | None:
        return (self.after_file.uri if self.after_file is not None else None) or (
            self.before_file.uri if self.before_file is not None else None
        )

    @property
    def memory_type(self) -> str:
        return str(
            (self.after_file.memory_type if self.after_file is not None else "")
            or (self.before_file.memory_type if self.before_file is not None else "")
            or (
                self.after_file.extra_fields.get("memory_type")
                if self.after_file is not None
                else ""
            )
            or (
                self.before_file.extra_fields.get("memory_type")
                if self.before_file is not None
                else ""
            )
        )

    @property
    def target_name(self) -> str:
        file = self.after_file if self.after_file is not None else self.before_file
        fields = file.extra_fields if file is not None else {}
        memory_type = self.memory_type
        type_specific_key = f"{str(memory_type).rstrip('s')}_name"
        name = (
            fields.get(type_specific_key)
            or fields.get("experience_name")  # backward compat
            or fields.get("name")
        )
        if name:
            return str(name)
        uri = self.target_uri
        if uri:
            # For SKILL.md-style paths, use the directory name.
            if uri.endswith("/SKILL.md"):
                parts = uri.rstrip("/").split("/")
                if len(parts) >= 2:
                    return parts[-2]
            return uri.rstrip("/").split("/")[-1].removesuffix(".md")
        return "unknown"


def _patch_language_text(patches: list[PatchMergePatch], schema: MemoryTypeSchema) -> str:
    parts = [
        text
        for patch in patches
        for value in _memory_file_fields(patch.after_file or patch.before_file, schema).values()
        for text in _string_values(value)
    ]
    return "\n".join(part for part in parts if part)


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for entry in value for item in _string_values(entry)]
    if isinstance(value, dict):
        return [item for entry in value.values() for item in _string_values(entry)]
    return []


class PatchMergeContextProvider(SessionExtractContextProvider):
    """Provide original memory files and structured field diffs to ExtractLoop.

    The provider is generic: callers decide which patches to pass in; this class
    only exposes original files as prefetched read results and memory-file field
    diffs as compact merge context.
    """

    def __init__(
        self,
        *,
        memory_type: str,
        patches: list[PatchMergePatch],
        required_file_uris: list[str] | None = None,
        output_language: str | None = None,
        memory_registry: MemoryTypeRegistry | None = None,
        vlm_config: VLMHandle | None = None,
        fixed_target: bool = False,
    ):
        super().__init__(messages=[], vlm_config=vlm_config)
        self.fixed_target = fixed_target
        self.memory_type = memory_type
        self._registry = memory_registry
        self.required_file_uris = list(required_file_uris or [])
        self.patches = list(patches)
        self._case_sources = {
            index: tuple(patch.metadata.get("source_extraction_ids", ()))
            for index, patch in enumerate(self.patches, 1)
            if memory_type == "cases" and patch.after_file is not None
        }
        if not any(self._case_sources.values()):
            self._case_sources = {}
        if output_language:
            self._output_language = output_language
        else:
            text = (
                _patch_language_text(self.patches, self.get_memory_schemas(None)[0])
                if self.patches
                else ""
            )
            self._output_language = resolve_output_language_from_text(text, fallback_language="en")

    def configure_output_context(self, context: Any) -> None:
        if self.fixed_target and self.memory_type == "experiences":
            context.include_link_example = False
        if not self._case_sources:
            return
        patch_ids = set(self._case_sources)
        base_model = context.operations_model

        class CaseMergeOperations(base_model):
            case_sources: list[_CaseSources] = Field(
                default_factory=list,
                description=(
                    "Bind every proposed Case patch number to its final Case page_id, including "
                    "unchanged existing Cases. A patch may contribute to multiple final Cases."
                ),
            )

            @model_validator(mode="after")
            def validate_case_sources(self):
                covered: set[int] = set()
                targets: set[int] = set()
                for binding in self.case_sources:
                    if binding.case_id in targets:
                        raise ValueError("Combine source patch IDs for each final Case")
                    targets.add(binding.case_id)
                    if len(set(binding.patch_ids)) != len(binding.patch_ids):
                        raise ValueError("Duplicate Case source patch ID")
                    covered.update(binding.patch_ids)
                if covered != patch_ids:
                    raise ValueError(
                        "case_sources must associate every proposed Case with its final Case; "
                        f"missing patch IDs: {sorted(patch_ids - covered)}, "
                        f"unknown patch IDs: {sorted(covered - patch_ids)}"
                    )
                writes = {item.page_id for item in self.cases}
                if (writes | set(self.selected_case_ids)) - targets:
                    raise ValueError("Every returned Case must have explicit source patches")
                # Preserve associations when the chosen Case needs no content write.
                self.selected_case_ids = list(
                    dict.fromkeys(
                        [
                            *self.selected_case_ids,
                            *(
                                case_id
                                for case_id in sorted(targets)
                                if case_id not in writes
                                or context.page_id_map.resolve(case_id)
                                in context.read_file_contents
                            ),
                        ]
                    )
                )
                return self

            def is_empty(self):
                return not self.case_sources and super().is_empty()

        context.operations_model = CaseMergeOperations

    def resolve_operation_metadata(self, raw_operations: Any, resolved: ResolvedOperations) -> None:
        if not self._case_sources:
            return
        outputs = [*resolved.upsert_operations, *resolved.case_references]
        for binding in raw_operations.case_sources:
            matches = [op for op in outputs if op.page_id == binding.case_id]
            if not matches or any(
                op.memory_type != "cases" or not op.uris or op.resolution_skip for op in matches
            ):
                raise ValueError(f"Case source target {binding.case_id} is not an available Case")
            sources = sorted(
                {
                    source
                    for patch_id in binding.patch_ids
                    for source in self._case_sources[patch_id]
                }
            )
            for operation in matches:
                operation.memory_fields["source_extraction_ids"] = sources

    def instruction(self) -> str:
        output_language = self._output_language
        if self.fixed_target:
            target = self.patches[0]
            return f"""Merge the Session proposals for ONE canonical Case Experience.
The only target is {target.target_uri}; keep the exact name {target.target_name!r}.
The prefetched file is the latest stored version. Proposals can be based on older
versions: apply their intended changes to the latest file while preserving unrelated
changes and successful paths. Follow the shared Edit instructions in the output contract,
using the latest source, not a stale proposal. The final content after applying
edits must be one complete valid DAG. Return no operations if the proposals add no useful change.
Never rename, delete, supersede, create another Experience, or merge another Case. Output restricted Python memory SDK code only.
Keep the fixed name; write natural-language DAG descriptions in {output_language}."""
        case_sources = (
            "\nFor Case proposals, retain execution associations independently of content writes. "
            "For each final Case, list the input Patch numbers that belong to it in case_sources. "
            "Cover every proposed Case patch, even when its alias is omitted or the existing "
            "canonical Case is unchanged. Never assign an unrelated patch to a Case.\n"
            if self._case_sources
            else ""
        )
        return f"""You are a memory patch merge agent.

You are given original schema business fields and their diffs. Merge them by producing final memory operations that follow the provided JSON schema.
Respect each field's merge rule. The server renders content_template after applying the merged fields;
do not generate or merge a rendered document as an extra field. A schema-defined content field is
still business data and must be merged normally.

Do not call tools. Output JSON only.

All memory content must be written in {output_language}.

Resolve additions, edits, and deletion proposals together into one final plan.
Apply this shared policy before choosing a canonical file or deleting any proposal:

{MEMORY_MERGE_POLICY}

Normalize URI/path variants for directory/filename fields. Treat path segment
fields as stable schema identifiers, not free-form labels. Reuse existing
equivalent directories across singular/plural, synonym, or language/script
variants. For new segments, use singular snake_case for English and one concise
canonical term for Chinese; e.g. book not books, 书籍 not 书/图书. If a loser URI
is a verified duplicate existing file, put it in delete_ids with the fact-complete
canonical replacement; if it is only a duplicate new proposal, omit it.
{case_sources}
"""

    def get_tools(self) -> list[str]:
        return []

    def get_memory_schemas(self, ctx: RequestContext | None) -> list[MemoryTypeSchema]:
        del ctx
        registry = self._get_registry()
        schema = registry.get(self.memory_type)
        if schema is None or not schema.enabled:
            raise ValueError(f"Memory schema not found or disabled: {self.memory_type}")
        return [schema]

    async def prefetch(self) -> list[dict[str, Any]]:
        schema = self.get_memory_schemas(self._ctx)[0]
        has_content_field = any(field.name == "content" for field in schema.fields)
        messages: list[dict[str, Any]] = []
        file_uris = await self._resolve_prefetch_file_uris()
        self._allowed_memory_read_uris.update(file_uris)
        for batch_start in range(0, len(file_uris), 20):
            batch = file_uris[batch_start : batch_start + 20]
            code = "\n".join(
                f"prefetched_memory_{batch_start + index} = sdk.read(uri={uri!r})"
                for index, uri in enumerate(batch, start=1)
            )
            result = await self._eval_memory(code)
            if not has_content_field and isinstance(result, dict):
                # Keep the full read cache; only the model-facing copy omits rendered body text.
                result = {
                    name: {key: value for key, value in record.items() if key != "content"}
                    if isinstance(record, dict)
                    and record.get("memory_type") == self.memory_type
                    and "error" not in record
                    else record
                    for name, record in result.items()
                }
            add_tool_call_pair_to_messages(
                messages=messages,
                call_id=None,
                tool_name="eval",
                params={"code": code},
                result=result,
            )

        messages.append(
            {
                "role": "user",
                "content": _render_field_diff_patches(self.patches, schema),
            }
        )
        return messages

    async def _resolve_prefetch_file_uris(self) -> list[str]:
        """Resolve required files plus semantic-search candidates for this merge."""

        required_uris = _dedupe_uris(self.required_file_uris)
        if self.fixed_target or not any(patch.before_file is None for patch in self.patches):
            return required_uris

        max_extra_candidate_files = min(_MAX_EXTRA_CANDIDATE_FILES, max(5, len(required_uris)))
        search_limit = max_extra_candidate_files * 2
        candidate_uris = await self._search_candidate_file_uris(limit=search_limit)
        extra_uris: list[str] = []
        required_set = set(required_uris)
        for uri in candidate_uris:
            if not uri or uri in required_set or uri in extra_uris:
                continue
            extra_uris.append(uri)
            if len(extra_uris) >= max_extra_candidate_files:
                break
        return [*required_uris, *extra_uris]

    async def _search_candidate_file_uris(self, *, limit: int) -> list[str]:
        schema = self._get_registry().get(self.memory_type)
        if schema is None or not schema.directory:
            return []
        search_dirs = self._render_search_directories(schema)
        if not search_dirs:
            return []
        query = _build_patch_search_query(self.patches)
        if not query:
            return []
        return await self.search_files(query=query, search_uris=search_dirs, limit=limit)

    def _render_search_directories(self, schema: MemoryTypeSchema) -> list[str]:
        if self._isolation_handler:
            return list(dict.fromkeys(self._isolation_handler.render_schema_directories(schema)))

        ctx = self._ctx
        user = getattr(ctx, "user", None)
        user_id = (
            getattr(ctx, "user_id", None)
            or getattr(user, "user_id", None)
            or _infer_user_space_from_uris(self.required_file_uris)
            or _infer_user_space_from_uris([patch.target_uri for patch in self.patches])
        )
        if not user_id:
            return []

        from openviking.session.memory.utils.uri import render_template

        return [render_template(schema.directory, {"user_space": user_id})]


def _render_field_diff_patches(patches: list[PatchMergePatch], schema: MemoryTypeSchema) -> str:
    if not patches:
        return "# Memory File Patches\n\nNo patches provided."
    rendered = [
        _render_one_field_diff_patch(index, patch, schema)
        for index, patch in enumerate(patches, start=1)
    ]
    return "# Memory File Patches\n\n" + "\n\n".join(rendered).rstrip()


def _render_one_field_diff_patch(
    index: int, patch: PatchMergePatch, schema: MemoryTypeSchema
) -> str:
    lines = [f"Patch {index}"]
    if patch.after_file is None:
        lines.append(f"  delete: {patch.target_uri}")
        return "\n".join(lines)
    if schema.memory_type == "cases" and patch.metadata.get("source_extraction_ids"):
        # Even an unchanged proposal needs an identity for its execution association.
        lines.append(f"  proposed_case: {_compact_value(patch.target_name)}")
    if patch.metadata:
        compact_metadata = _compact_patch_metadata(patch.metadata)
        if compact_metadata:
            lines.append(f"  meta: {_compact_value(compact_metadata)}")
    field_diffs = _field_diffs(patch.before_file, patch.after_file, schema)
    if not field_diffs:
        lines.append("  (no changes)")
        return "\n".join(lines)
    for field_name, diff in field_diffs:
        lines.append(f"  {field_name}:")
        # Strip unified diff headers (---, +++) but keep @@ hunk markers and content
        for diff_line in diff.splitlines():
            if diff_line.startswith("---") or diff_line.startswith("+++"):
                continue
            lines.append(f"    {diff_line}")
    return "\n".join(lines)


def _field_diffs(
    before_file: MemoryFile | None, after_file: MemoryFile, schema: MemoryTypeSchema
) -> list[tuple[str, str]]:
    before_fields = _memory_file_fields(before_file, schema) if before_file is not None else {}
    after_fields = _memory_file_fields(after_file, schema)
    diffs: list[tuple[str, str]] = []
    for field_name in sorted(set(before_fields) | set(after_fields)):
        before_value = before_fields.get(field_name)
        after_value = after_fields.get(field_name)
        if before_value == after_value:
            continue
        diff = _value_unified_diff(
            field_name=field_name,
            before_value=before_value,
            after_value=after_value,
        )
        if diff.strip():
            diffs.append((field_name, diff))
    return diffs


def _memory_file_fields(file: MemoryFile | None, schema: MemoryTypeSchema) -> dict[str, Any]:
    """Project declared business fields; omit the body when it is only template output."""
    if file is None:
        return {}
    fields = {
        field.name: file.extra_fields[field.name]
        for field in schema.fields
        if field.name not in _SYSTEM_HIDDEN_FIELDS and field.name in file.extra_fields
    }
    if any(field.name == "content" for field in schema.fields):
        fields["content"] = file.content
    return fields


def _value_unified_diff(*, field_name: str, before_value: Any, after_value: Any) -> str:
    before_lines = _value_lines(before_value)
    after_lines = _value_lines(after_value)
    diff_lines = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"{field_name}.before",
        tofile=f"{field_name}.after",
        n=1,
        lineterm="",
    )
    return "\n".join(diff_lines)


def _value_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.splitlines()
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).splitlines()


def _compact_value(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _hide_system_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _hide_system_fields(item)
            for key, item in value.items()
            if key not in _SYSTEM_HIDDEN_FIELDS
        }
    if isinstance(value, list):
        return [_hide_system_fields(item) for item in value]
    return value


def _compact_patch_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only metadata that helps reconcile patch proposals.

    Full gradient metadata can contain large duplicated fields (links, uris, and
    memory_fields). The patch body already renders the target URI and field
    changes, while source links are merged outside the LLM response. Keep only
    decision signals that help the merge model rank or reconcile proposals.
    """

    cleaned = _hide_system_fields(dict(metadata or {}))
    result = {
        key: cleaned[key]
        for key in _PATCH_METADATA_KEYS
        if key in cleaned and _metadata_value_is_useful(cleaned[key])
    }

    return result


def _metadata_value_is_useful(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if isinstance(value, (list, dict, tuple, set)) and not value:
        return False
    return True


def _dedupe_uris(uris: list[str] | None) -> list[str]:
    return list(dict.fromkeys(uri for uri in (uris or []) if uri))


def _build_patch_search_query(patches: list[PatchMergePatch]) -> str:
    parts: list[str] = []
    for patch in patches:
        if patch.target_name:
            parts.append(str(patch.target_name))
        if patch.target_uri:
            parts.append(str(patch.target_uri).rstrip("/").split("/")[-1].removesuffix(".md"))
        file = patch.after_file if patch.after_file is not None else patch.before_file
        content = str(file.content or "") if file is not None else ""
        if content:
            parts.append(_truncate_query_text(content, 1200))
    return _truncate_query_text("\n\n".join(parts), 5000)


def _truncate_query_text(text: Any, max_chars: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _infer_user_space_from_uris(uris: list[str | None]) -> str | None:
    for uri in uris:
        if not uri:
            continue
        prefix = "viking://user/"
        if not uri.startswith(prefix):
            continue
        rest = uri.removeprefix(prefix)
        user_space = rest.split("/", 1)[0]
        if user_space and user_space != "memories":
            return user_space
    return None
