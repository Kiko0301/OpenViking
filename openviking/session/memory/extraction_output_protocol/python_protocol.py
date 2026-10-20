# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Restricted Python SDK output protocol for memory extraction."""

from __future__ import annotations

import ast
import json
import keyword
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from openviking.core.peer_id import safe_peer_id
from openviking.session.memory.dataclass import MemoryFile, MemoryTypeSchema
from openviking.session.memory.extraction_output_protocol.base import (
    ExtractionOutputContext,
    ExtractionOutputProtocol,
    ExtractionOutputProtocolError,
)
from openviking.session.memory.extraction_output_protocol.edit_format import EDIT_TOOL_DESCRIPTION
from openviking.session.memory.merge_op import (
    DeleteBlock,
    FieldType,
    ImmutableOp,
    MergeOp,
    SearchReplaceBlock,
    StrPatch,
)
from openviking.session.memory.tools import add_tool_call_pair_to_messages
from openviking.session.memory.utils.description_template import render_description_template
from openviking.session.memory.utils.line_numbers import (
    every_line_has_line_numbers,
    strip_line_numbers,
)

_PYTHON_FENCE_RE = re.compile(r"```python[ \t]*\r?\n(?P<code>[\s\S]*?)```", re.IGNORECASE)
_PYTHON_FENCE_START_RE = re.compile(r"```python[ \t]*\r?\n", re.IGNORECASE)
_HIDDEN_MEMORY_FIELDS = {
    "action_feedback",
    "action_score_sources",
    "feedback_stats",
    "experience_effect_sources",
    "trajectory_format",
    "evaluation",
    "experience_execution",
    "experience_update",
    "source_extraction_id",
    "source_extraction_ids",
    "last_update_trace_id",
    "version",
}
_SAFE_BUILTINS = {
    "bool": bool,
    "dict": dict,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "str": str,
    "tuple": tuple,
}
_SAFE_STRING_METHODS = {
    "endswith",
    "join",
    "lower",
    "replace",
    "split",
    "startswith",
    "strip",
    "upper",
}
# Upper bound on the size of a value produced by a DSL expression. Multiplying or
# joining a small literal by a huge count would otherwise allocate a massive
# string/list in the server process before schema validation runs; the check is
# applied to the would-be result size before any allocation.
_MAX_EXPRESSION_SIZE = 1_000_000
_MAX_FORMAT_RETRY_EXCERPT_CHARS = 2000
_REPEATABLE_SEQUENCES = (str, bytes, list, tuple)
_NESTED_STRING_GUIDANCE = (
    "Inside an outer '''...''' SDK string, use "
    '"""...""" for embedded Python/DAG text literals; '
    'these inner """ delimiters need no escaping at the outer layer.\n'
    r"Escape each literal ''' as \'\'\' and each literal backslash as \\."
    " Escape trailing single quotes that would join the closing delimiter. If retaining a "
    "different valid outer delimiter, escape that delimiter instead.\n"
    "Escape only the string contents, not the enclosing opening and closing delimiters. "
    "Escapes are decoded once by the outer Python string, preserving quotes, backslashes and "
    "newlines. Do not rewrite embedded content to enforce this convention.\n"
    "Syntax-only example (string value, not a write operation):\n"
    "```python\n"
    "content_text = '''note = \"\"\"first line\n"
    "second line\"\"\"'''\n"
    "```"
)

_JSON_STRING_GUIDANCE = r"""For fields defined as JSON strings, use a raw triple-single-quoted string (r'''...''').
For these fields only, follow the JSON escaping rules here instead of the ordinary-string
escaping rules above. The field remains a string, not a Python list or dict.
Inside it, write valid JSON with double-quoted keys/strings and true/false/null. Escape once
for JSON: \" for a quote inside a JSON string, \n for a newline, and \\ for a backslash.
Do not escape those backslashes again for Python. If the JSON text contains literal ''',
encode an apostrophe as \u0027 so it cannot close the outer string; \' is not a JSON escape.
Syntax-only example; use the actual keys and IDs required by the target field:
```python
json_text = r'''[
    {"text": "The user said \"yes\".\nNext line.", "ok": true, "missing": null}
]'''
```
Pass json_text directly to the JSON-string field in a listed write method. Do not use
str(list_or_dict) as JSON. This convention does not change field validation or enable imports."""

_CONTRACT_PREAMBLE = (
    "## Output Format: restricted Python memory SDK",
    "Return only Python code, optionally wrapped in one ```python code fence.",
    "The code is interpreted as a restricted DSL and cannot access Python modules, files, or the network.",
    "The final SDK program is write-only. To inspect memories, use the available native read-only tool before returning the program. Inside eval, each SDK read/search call must be assigned to a variable; never put read/search calls in the final program. Only the system may emit sdk.existing() or sdk.error().",
    "Never return memory content as bare Markdown; all changes must be arguments to SDK calls.",
    "Tool replies are Python result code, not JSON. Eval results keep their assigned variable names. "
    "Loaded memories appear as sdk.existing(...) objects; search results may contain these objects "
    "in lists or dictionaries. Use their shown names or indexed references (e.g. matches[0]) directly "
    "in the final program; do not repeat their binding declarations. Failed reads appear as sdk.error(message); these are not editable memory objects.",
    "Existing objects shown by the system may use obj.update(field=value, ...) for "
    "scalar/immutable fields according to their schema merge rules, obj.delete(replacement=None),",
    "and, when links are enabled, obj.link(target, link_type='related_to', weight=0.5, match_text=None, description='').",
    "Use a triple-single-quoted string ('''...''') for every outer SDK natural-language argument "
    "(content, summary, goal, and every search=/replace=/text= snippet), even one-liners. "
    "Use plain strings by default; use f'''...''' only when interpolation is needed, escaping "
    "literal braces as {{ and }}. This is a generation convention, not a restriction on other "
    "valid Python quote styles.",
    _NESTED_STRING_GUIDANCE,
    _JSON_STRING_GUIDANCE,
    "Only keyword arguments are accepted by create, set, and obj.update(); a field's overwrite() takes one positional string. Unknown business fields are ignored.",
    "You may end the program with sdk.commit(); when present it must be the final call. If there are no changes, return only sdk.commit().",
    "Use system-provided existing-object bindings, or assign a fresh alias such as "
    "alias = existing_1 (using a binding actually shown in the tool results). "
    "Aliases refer to the same memory object; "
    "they do not create or copy memories. Never overwrite an existing binding. When a newly "
    "created memory must be referenced by delete(replacement=...) or link(...), assign the "
    "create call to a variable first, for example: canonical = sdk.create_<type>(...); "
    "duplicate_1.delete(replacement=canonical). Never recreate or rebind an existing object "
    "with sdk.existing().",
    "",
    "Available write methods for this extraction:",
)

_FIELD_WRITE_METHODS = (
    "",
    "## Existing string field methods",
    "Use the real field name (e.g. content), NOT the literal word 'field'. "
    "A field attribute is a write handle only; it cannot be read as a string.",
    "### obj.field.edit(search=..., replace=...)",
    EDIT_TOOL_DESCRIPTION,
    "Local replacement example (use an actual existing-object binding instead of obj):\n"
    "```python\n"
    '# Existing content: "Editor: Vim\\nTheme: dark"\n'
    "# Change only the editor; keep the theme.\n"
    "obj.content.edit(search='''Editor: Vim''', replace='''Editor: Emacs''')\n"
    "```",
    "### obj.field.drop(text=...)",
    "Replace the selected text with an empty string. Include its newline to remove a whole line.",
    "### obj.field.overwrite(new_value)",
    "Use overwrite only for complete field rewrites. "
    "Replace the whole field with a complete new string. This is not an append or merge. "
    "All old content omitted from the new string is removed from this field. "
    "Keep all still-valid existing constraints.",
    "edit()/drop() only work on existing string fields that support partial editing; "
    "new memories from create/set must be given complete field values. "
    "edit()/drop() may be chained; do not mix them with overwrite() in one chain.",
)


_LINK_RULES = (
    "",
    "## Link Rules",
    "- Create links between memories when they have a meaningful and clear relationship "
    "(e.g. an event mentioning a person/entity, a preference about a place/activity). "
    "Do NOT force links between unrelated items.",
    "- Use obj.link(target, link_type='related_to', match_text='<word>') as a standalone "
    "statement after both objects exist.",
    "- match_text must be a single exact word that appears in the source object's content. "
    "It is where the link anchor is inserted. Choose the most specific identifying word "
    "(e.g. a person's name, a place, an activity).",
    "- link_type uses lowercase snake_case: related_to (default), belongs_to, "
    "derived_from, caused_by, contradicts, evolved_from.",
    "- For a newly created object, assign the create/set call to a variable first, then "
    "call .link() on it (see the preamble); existing objects use their provided bindings or aliases.",
    "- Links are optional; only create them when the relationship is explicit in the "
    "conversation. Do not link every memory to every other memory.",
)

_LINK_EXAMPLE = (
    "",
    "Example — linking a newly created memory to another new one and to an existing binding "
    "(replace create_<type>/fields with methods and fields from the contract above):",
    "```python",
    "obj_a = sdk.create_<type_a>(...)  # a new memory, assigned to a variable",
    "obj_b = sdk.create_<type_b>(...)  # another new memory",
    'obj_a.link(obj_b, link_type="related_to", match_text="<word_in_obj_a>")',
    'obj_a.link(existing_1, link_type="related_to", match_text="<word_in_obj_a>")',
    "```",
)


@dataclass(slots=True)
class _MemoryObject:
    name: str
    memory_type: str
    page_id: int
    fields: dict[str, Any]
    existing: bool
    deleted: bool = False
    replacement: _MemoryObject | None = None
    changed_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ExistingMemoryReference:
    """A tool-visible reference; the compiler resolves it from its trusted read cache."""

    uri: str


_UNSET = object()


@dataclass(slots=True)
class _FieldHandle:
    """A pending edit chain on one field of a memory object (obj.field.*)."""

    owner: _MemoryObject
    field_name: str
    blocks: list[Any] = field(default_factory=list)
    full_value: Any = _UNSET


class PythonExtractionOutputProtocol(ExtractionOutputProtocol):
    """Restricted Python-shaped memory SDK compiler."""

    name = "python"

    def __init__(self) -> None:
        self._result_bindings: dict[str, Any] = {}
        self._last_error_allows_tool_retry = False

    def get_memory_eval_bindings(self, context: ExtractionOutputContext) -> dict[str, Any]:
        """Expose the same trusted objects as read-result values, without write handles."""
        schemas = {schema.memory_type: schema for schema in context.schemas}

        def snapshot(value: Any) -> Any:
            if isinstance(value, _ExistingMemoryReference):
                memory = context.read_file_contents[value.uri]
                memory_type = memory.memory_type or memory.extra_fields.get("memory_type")
                fields = _visible_memory_fields(memory, schemas[memory_type], context)
                return {**deepcopy(fields), "uri": value.uri, "memory_type": memory_type}
            if isinstance(value, list):
                return [snapshot(item) for item in value]
            if isinstance(value, dict):
                return {key: snapshot(item) for key, item in value.items()}
            return deepcopy(value)

        return {name: snapshot(value) for name, value in self._result_bindings.items()}

    def render_binding_index(self) -> str:
        paths: list[str] = []

        def collect(value: Any, path: str) -> None:
            if isinstance(value, _ExistingMemoryReference):
                paths.append(path)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    collect(item, f"{path}[{index}]")
            elif isinstance(value, dict):
                for key, item in value.items():
                    collect(item, f"{path}[{key!r}]")

        for name, value in self._result_bindings.items():
            collect(value, name)
        return (
            "Existing-memory access ONLY via: "
            + (", ".join(paths) or "(none)")
            + ". Assign aliases before use; read unlisted memories first."
        )

    def render_contract(self, context: ExtractionOutputContext) -> str:
        _validate_alias_uniqueness(context.schemas)
        lines = list(_CONTRACT_PREAMBLE)
        for schema in context.schemas:
            lines.extend(self._render_schema_contract(context, schema))
        lines.extend(_FIELD_WRITE_METHODS)
        if (
            "selected_case_ids" in context.operations_model.model_fields
            and "case_sources" not in context.operations_model.model_fields
        ):
            lines.append(
                "Binding a matched, loaded Case to a local variable selects it without a write. "
                "sdk.commit() is optional; use sdk.commit(cases=[...]) to override implicit "
                "selections, or cases=[] to clear them. Unbound search hits are not selected. "
                "New or changed Cases use the normal write path."
            )
        if "case_sources" in context.operations_model.model_fields:
            lines.append(
                "For this Case merge, end with sdk.commit(case_sources=["
                "(case_object, [patch_number, ...]), ...]). Each tuple associates a final Case "
                "(loaded or newly created) with its input Patch numbers. Include unchanged Cases; "
                "cover every proposed Case patch and never attach unrelated patches. "
                "This also selects existing Cases without writing their content."
            )
        if not context.link_enabled:
            lines.append("Links are disabled; obj.link(...) is unavailable.")
        else:
            lines.extend(_LINK_RULES)
            if context.include_link_example:
                lines.extend(_LINK_EXAMPLE)
        return "\n".join(lines)

    def _render_schema_contract(
        self, context: ExtractionOutputContext, schema: MemoryTypeSchema
    ) -> list[str]:
        fields = _protocol_fields(context, schema)
        type_alias = _identifier_alias(schema.memory_type)
        schema_fields = {field.name: field for field in schema.fields}
        signature_parts = []
        for name, type_name, _description in fields:
            part = f"{_identifier_alias(name)}: {type_name}"
            field_schema = schema_fields.get(name)
            if field_schema is not None and field_schema.init_value is not None:
                part += f" = {field_schema.init_value!r}"
            signature_parts.append(part)
        signature = ", ".join(signature_parts)
        verb = "create" if schema.filename_has_variables() else "set"
        identity_fields = _model_visible_identity_fields(context, schema)
        identity_label = (
            ", ".join(_identifier_alias(name) for name in identity_fields)
            or "target scope (fixed to self)"
        )
        lines = [
            f"- sdk.{verb}_{type_alias}(*, {signature})",
            f"  - Identity fields (primary key): {identity_label}. Calls with identical "
            "identity field values address the same memory object.",
        ]
        if schema.operation_mode == "add_only":
            lines.append(
                f"  - {type_alias} is append-only. Create a new object for this extraction. "
                "Never update, edit, drop, or delete an existing object. If an existing binding "
                "is shown as context, treat it as read-only reference material."
            )
        operation_field = context.operations_model.model_fields.get(schema.memory_type)
        schema_description = getattr(operation_field, "description", None)
        if schema_description:
            # Reuse the already-rendered description from SchemaModelGenerator.
            # This keeps routing/cleanup rules identical to the JSON schema contract
            # (and resolves template variables such as {{ language }}).
            lines.append("  - Memory type rules:\n" + str(schema_description).strip())
        if verb == "set":
            lines.extend(self._render_singleton_rules(context, schema, identity_fields))
        merge_ops = {
            field.name: (
                "editable string: obj.field.edit/drop/overwrite"
                if field.merge_op == MergeOp.PATCH and field.field_type == FieldType.STRING
                else field.merge_op.value
            )
            for field in schema.fields
        }
        for name, _type_name, description in fields:
            if name in merge_ops:
                # Render only YAML field descriptions, using the same context and
                # restricted renderer as JSON. Keep the DSL's own edit instructions
                # instead of copying the JSON model's merge-operation wrappers.
                description = render_description_template(description, context.template_context)
            normalized_description = " ".join(str(description or "").split())
            qualifier = f" [{merge_ops[name]}]" if name in merge_ops else ""
            lines.append(f"  - {_identifier_alias(name)}{qualifier}: {normalized_description}")
        return lines

    @staticmethod
    def _render_singleton_rules(
        context: ExtractionOutputContext,
        schema: MemoryTypeSchema,
        identity_fields: tuple[str, ...],
    ) -> list[str]:
        type_alias = _identifier_alias(schema.memory_type)
        lines = [
            f"  - {type_alias} is a SINGLETON per identity: sdk.create_{type_alias}() "
            f"does not exist, and sdk.set_{type_alias}() must appear AT MOST ONCE for each "
            "identity in the whole program. Never call it twice for the same identity (e.g. the same "
            "peer_id). If several people or sections belong to one identity, put them all as separate "
            "H1 sections inside a SINGLE content argument of one call — different headings or person "
            "names do NOT create separate identities. To change an already-shown existing object, "
            "edit that object instead of calling set again."
        ]
        if "peer_id" in identity_fields and context.role_scope is not None:
            if context.role_scope.user_ids:
                lines.append(
                    f"  - Current self identity: exactly one sdk.set_{type_alias}() call "
                    "without peer_id; combine all content sections into its single content argument."
                )
            for peer_id in context.role_scope.peer_ids:
                lines.append(
                    f"  - Current identity peer_id={peer_id!r}: exactly one "
                    f"sdk.set_{type_alias}(peer_id={peer_id!r}, ...) call; combine every "
                    "person and section for this peer_id into that one content argument."
                )
        return lines

    def render_reference_rules(self, context: ExtractionOutputContext) -> str:
        del context
        return """
## Memory Object Rules
- Update or delete an existing memory only through its system-provided bound object or an alias of that object; never pass or construct a URI.
- Create collection memories with listed sdk.create_<memory_type>(...) methods. create_* is create-only: an existing target or duplicate target in one program fails without overwriting it. Read existing memories through eval and edit their content in place; use content.overwrite(...) only for a complete field replacement.
- Set a single-file memory with its listed sdk.set_<memory_type>(...) method; each target scope has only one such object.
- Existing-object identity, storage paths, and immutable fields are preserved by the system.
- delete() removes the whole object; use obj.content.drop(text=...) (with the real field name) when only some content must go and the rest stays.
- For canonical merges, use duplicate.delete(replacement=canonical); for pure deletes, call delete() without replacement.
- delete(replacement=canonical) discards the duplicate's content entirely and keeps only the canonical. Before deleting a duplicate, first fold every distinct valid fact it holds into the canonical (e.g. canonical.content.edit(...)); merging or compacting must never drop a unique fact that only the duplicate recorded.
"""

    def parse(
        self, content: str, context: ExtractionOutputContext
    ) -> tuple[Any | None, str | None]:
        self._last_error_allows_tool_retry = False
        context.create_page_ids.clear()
        try:
            code = _extract_python_code(content)
            compiler = _PythonProgramCompiler(context=context, protocol=self)
            operations = compiler.compile(code)
            return operations, None
        except ExtractionOutputProtocolError as exc:
            self._last_error_allows_tool_retry = exc.allow_tool_retry
            return None, str(exc)
        except Exception as exc:
            return None, str(exc)

    def render_final_instruction(self, context: ExtractionOutputContext) -> str:
        if "case_sources" in context.operations_model.model_fields:
            return (
                "Return the complete restricted Python memory SDK program now. Do not call tools. "
                "End with sdk.commit(case_sources=[(final_case, [patch_number, ...]), ...]). "
                "Every proposed Case patch must be associated with its final Case, including "
                "unchanged existing Cases. Output only Python code."
            )
        selection = (
            " Select matched existing Cases by assigning them to local variables "
            "or passing cases= to sdk.commit()."
            if "selected_case_ids" in context.operations_model.model_fields
            else ""
        )
        return (
            "You have reached the maximum number of tool call iterations. Do not call any more "
            "tools. Return the complete restricted Python memory SDK program now. Output only "
            "Python code."
            + selection
            + " If there are no changes or Case selections, return only sdk.commit()."
        )

    def render_format_retry(self, error: str | None = None, *, failed_content: str = "") -> str:
        detail = f" Parser error: {error}" if error else ""
        tool_guidance = ""
        binding_guidance = ""
        quote_guidance = ""
        if error and "sdk.existing() is reserved" in error:
            binding_guidance = (
                " Use the existing-object variable names already supplied by the system; do not "
                "emit sdk.existing(). If a new object is the replacement, assign its create call "
                "first, for example `canonical = sdk.create_<type>(...)`, then call "
                "`duplicate_1.delete(replacement=canonical)`."
            )
        if error and _is_string_literal_syntax_error(error):
            quote_guidance = (
                " If the error is about a string literal, the offending line is shown above; a "
                "quote, newline or unescaped nested delimiter may have ended the string early. "
                "Preserve the intended text while fixing the delimiter or escaping. Write every "
                "natural-language argument (content, summary, goal, and every search=/replace=/"
                "text= snippet) as a triple-single-quoted string ('''...''') so embedded quotes and "
                "newlines stay inside the literal."
            )
        if self.keep_tools_enabled_after_parse_error(error):
            tool_guidance = (
                " The final SDK program is write-only. Call the available native read-only tool "
                "inside eval with an assignment; after its result "
                "is returned, update the system-provided object variable. "
                "Do not put read/search calls in the final program or guess a memory URI."
            )
        return (
            "Your previous output was not a valid restricted Python memory SDK program."
            f"{detail}{tool_guidance}{binding_guidance}{quote_guidance} Regenerate the complete "
            "program and output no explanation. "
            "Make only the changes needed to fix the reported error; preserve all other memory "
            "operations, Case identities, merge decisions, and source associations. Do not "
            "re-plan the extraction because formatting failed. Return the complete corrected "
            "program, not just a repair fragment. "
            "Put Markdown memory content inside quoted SDK arguments, using triple-single-quoted strings "
            "('''...''') for any multi-line or Markdown text so newlines and quotes stay inside "
            "the literal; never return it as the program itself. No changes from the invalid program "
            "were applied. A syntax error pointing at prose or punctuation may mean an embedded "
            "quote delimiter closed the surrounding string early; check its escaping rather than "
            "deleting or rewriting valid content to avoid the reported character."
            f"\n\n{_NESTED_STRING_GUIDANCE}"
            f"\n\n{_JSON_STRING_GUIDANCE}"
            f"{_format_retry_source_excerpt(failed_content, error)}"
        )

    def keep_tools_enabled_after_parse_error(self, error: str | None) -> bool:
        # Set by parse() from the structured error flag, so this stays correct
        # even if the write-only read/search rejection wording changes.
        del error
        return self._last_error_allows_tool_retry

    def describe_empty_response(self) -> str | None:
        return "The model returned an empty Python program"

    def render_patch_repair(self, patch_errors: list[dict[str, Any]]) -> str:
        details = json.dumps(patch_errors, ensure_ascii=False, indent=2)
        return (
            "An obj.field.edit()/drop() change could not be applied to its target field. "
            "Follow the Edit instructions in the initial output contract. "
            "Regenerate the complete SDK program, preserving other intended operations and "
            "source associations. Do not move the edit to another object merely because "
            "its text matches.\n\nFailed edits:\n" + details
        )

    def render_resolution_repair(self, issues: list[dict[str, Any]]) -> str:
        details = json.dumps(issues, ensure_ascii=False, indent=2)
        return (
            "Some sdk.create_events(...) calls could not resolve a safe write target. "
            "Re-emit ONLY corrected create_events() calls for the failed items below; do not "
            "emit any other memory type, delete, or link statements. The server has preserved "
            "all successful operations from the previous program. "
            "For event ranges, pass ranges= with valid in-bounds message indexes that include the "
            "user-role message establishing the event so its owner can be resolved (e.g. "
            'ranges="3-5"). Do not target a disallowed or ambiguous peer. Keep the same event_name '
            "shown for each failed item so it maps back to that event. Use triple-single-quoted strings "
            "('''...''') for every natural-language argument. Output only Python code."
            "\n\nResolution issues:\n" + details
        )

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
            messages, call_id=call_id, tool_name=tool_name, params=params, result=result
        )
        # The transport remains a native tool pair. Only Python's tool content
        # changes representation; raw execution results and JSON mode are unchanged.
        if isinstance(result, dict) and isinstance(result.get("error"), str):
            expression, _, metadata = self._render_result_value(context, result, path=tool_name)
            messages[-1]["content"] = "\n".join([expression, *metadata])
            return messages

        results = (
            result if tool_name == "eval" and isinstance(result, dict) else {tool_name: result}
        )
        declarations = []
        for name, value in results.items():
            _require_identifier(name, "tool result variable")
            if name == "sdk" or name.startswith("_"):
                raise ValueError(f"Invalid tool result variable: {name!r}")
            uri = params.get("uri") if tool_name == "read" and isinstance(params, dict) else None
            expression, binding, metadata = self._render_result_value(
                context, value, path=name, uri=uri
            )
            self._result_bindings[name] = binding
            declarations.append(f"{name} = {expression}")
            declarations.extend(metadata)
        messages[-1]["content"] = "\n".join(declarations) or repr(result)
        return messages

    def _render_result_value(
        self,
        context: ExtractionOutputContext,
        value: Any,
        *,
        path: str,
        uri: str | None = None,
    ) -> tuple[str, Any, list[str]]:
        """Render literal containers, replacing only actually-read memories by SDK objects."""
        if isinstance(value, dict):
            if isinstance(value.get("error"), str):
                return (
                    f"sdk.error({value['error']!r})",
                    value,
                    self._render_result_metadata(value, path=path, exclude={"error"}),
                )
            result_uri = value.get("uri") or uri
            page_id = value.get("page_id")
            mapped_uri = (
                context.page_id_map.resolve(page_id)
                if isinstance(page_id, int) and not isinstance(page_id, bool)
                else None
            )
            if result_uri and mapped_uri and result_uri != mapped_uri:
                # Never turn inconsistent identity metadata into an editable reference.
                return repr(value), value, []
            result_uri = result_uri or mapped_uri
            memory_file = (
                context.read_file_contents.get(result_uri) if isinstance(result_uri, str) else None
            )
            schema = next(
                (s for s in context.schemas if s.memory_type == value.get("memory_type")), None
            )
            if (
                memory_file is not None
                and schema is not None
                and (memory_file.memory_type or memory_file.extra_fields.get("memory_type"))
                == schema.memory_type
                and "error" not in value
            ):
                fields = _visible_memory_fields(memory_file, schema, context)
                fields.update({key: value[key] for key in fields.keys() & value.keys()})
                content = fields.get("content")
                if isinstance(content, str) and every_line_has_line_numbers(content):
                    fields["content"] = strip_line_numbers(content)
                args = [f"memory_type={schema.memory_type!r}"]
                args.extend(f"{_identifier_alias(key)}={v!r}" for key, v in fields.items())
                comments = self._render_result_metadata(
                    value, path=path, exclude={*fields, "memory_type", "uri"}
                )
                return (
                    f"sdk.existing({', '.join(args)})",
                    _ExistingMemoryReference(result_uri),
                    comments,
                )
            items, bindings, metadata = [], {}, []
            for key, item in value.items():
                expression, binding, comments = self._render_result_value(
                    context, item, path=f"{path}[{key!r}]"
                )
                items.append(f"{key!r}: {expression}")
                bindings[key] = binding
                metadata.extend(comments)
            return "{" + ", ".join(items) + "}", bindings, metadata
        if isinstance(value, list):
            items, bindings, metadata = [], [], []
            for index, item in enumerate(value):
                expression, binding, comments = self._render_result_value(
                    context, item, path=f"{path}[{index}]"
                )
                items.append(expression)
                bindings.append(binding)
                metadata.extend(comments)
            return "[" + ", ".join(items) + "]", bindings, metadata
        return repr(value), value, []

    @staticmethod
    def _render_result_metadata(
        value: dict[str, Any], *, path: str, exclude: set[str]
    ) -> list[str]:
        metadata = {
            key: item
            for key, item in value.items()
            if key not in exclude
            and key != "page_id"
            and key not in _HIDDEN_MEMORY_FIELDS
            and not key.startswith("_")
        }
        return [f"# Metadata for {path}: {metadata!r}"] if metadata else []

    def normalize_provider_instruction(self, instruction: str) -> str:
        replacements = {
            "Output JSON only. Do not call any tools.": "Do not call any tools. Follow the output contract below.",
            "Do not call tools. Output JSON only.": "Do not call tools. Follow the output contract below.",
            "follow the provided JSON schema": "follow the provided output contract",
            "Output JSON only.": "Follow the output contract below.",
            "output ONLY a JSON object (no extra text before or after)": "follow the output contract below",
            "output only JSON that matches the schema descriptions": "follow the output contract below",
            "Do NOT use `delete_ids`": "Do NOT call `delete()`",
            "put it in delete_ids": "delete it through its bound memory object",
        }
        for old, new in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            instruction = instruction.replace(old, new)
        return instruction


class _PythonProgramCompiler:
    def __init__(
        self, *, context: ExtractionOutputContext, protocol: PythonExtractionOutputProtocol
    ) -> None:
        self.context = context
        self.protocol = protocol
        # Reject alias collisions before building the maps so a folded-name clash
        # cannot silently overwrite a schema/field (also guards parse() paths that
        # do not go through render_contract first).
        _validate_alias_uniqueness(context.schemas)
        self.schemas = {schema.memory_type: schema for schema in context.schemas}
        # DSL surface uses identifier aliases; map them back to real schema names.
        self._type_alias_to_real = {
            _identifier_alias(schema.memory_type): schema.memory_type for schema in context.schemas
        }
        self._field_alias_to_real = {
            schema.memory_type: {
                _identifier_alias(name): name
                for name, _type, _description in _protocol_fields(context, schema)
            }
            for schema in context.schemas
        }
        self.objects: dict[str, _MemoryObject] = {}
        self.values: dict[str, Any] = {}
        self.links: list[dict[str, Any]] = []
        self.selected_cases: list[_MemoryObject] = []
        self._explicit_case_selection = False
        self.case_sources: list[dict[str, Any]] = []
        self._singleton_identities: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
        self._next_new_page_id = self._find_next_new_page_id(100)
        self._committed = False
        self._load_existing_objects()
        self._input_value_names = {
            name for name, value in self.values.items() if not _contains_memory_object(value)
        }

    def compile(self, code: str) -> Any:
        try:
            module = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            # A valid program that merely embeds Markdown in a triple-quoted
            # string parses fine, so only fall back to the bare-Markdown hint
            # when parsing actually fails.
            if _looks_like_bare_markdown(code):
                raise ExtractionOutputProtocolError(
                    "Output is bare Markdown, not a Python SDK program; put memory content inside "
                    "a quoted sdk.create_*/sdk.set_* argument or an existing string field's overwrite() call"
                ) from exc
            raise ExtractionOutputProtocolError(
                f"Line {exc.lineno or 1}: invalid Python syntax: {exc.msg}"
                + _format_syntax_error_source(exc)
            ) from exc
        # sdk.commit() is optional: collect writes and Case selections from the program.
        # When present, commit() is a terminator and must be the final statement.
        for index, statement in enumerate(module.body):
            if self._committed:
                self._error(statement, "sdk.commit() must be the final statement")
            self._execute_statement(statement)
            if self._committed and index != len(module.body) - 1:
                self._error(statement, "sdk.commit() must be the final statement")
        return self._build_operations()

    def _load_existing_objects(self) -> None:
        by_uri: dict[str, _MemoryObject] = {}

        def load(value: Any, path: str) -> Any:
            if isinstance(value, _ExistingMemoryReference):
                if value.uri not in by_uri:
                    memory_file = self.context.read_file_contents[value.uri]
                    memory_type = str(
                        memory_file.memory_type or memory_file.extra_fields.get("memory_type")
                    )
                    page_id = self.context.page_id_map.get_page_id(value.uri)
                    obj = _MemoryObject(
                        name=path,
                        memory_type=memory_type,
                        page_id=page_id,
                        fields=_visible_memory_fields(
                            memory_file, self.schemas[memory_type], self.context
                        ),
                        existing=True,
                    )
                    by_uri[value.uri] = obj
                    # Multiple eval variables may refer to one file. Persist it
                    # once, while every displayed reference sees the same object.
                    self.objects[f"__existing_{page_id}"] = obj
                return by_uri[value.uri]
            if isinstance(value, list):
                return [load(item, f"{path}[{i}]") for i, item in enumerate(value)]
            if isinstance(value, dict):
                return {key: load(item, f"{path}[{key!r}]") for key, item in value.items()}
            return value

        for name, value in self.protocol._result_bindings.items():
            self.values[name] = load(value, name)

    def _execute_statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                self._error(node, "only one simple assignment target is allowed")
            name = node.targets[0].id
            # Evaluate first so an attempted sdk.read()/sdk.search() reports the
            # actionable write-only SDK error even when the model reuses a
            # system-provided binding name on the left-hand side.
            value = self._eval(node.value)
            self._reserve_name(name, node)
            if value is None:
                self._error(node, "side-effect calls cannot be assigned")
            if isinstance(value, _MemoryObject):
                if value.existing or value.name:
                    # An alias refers to the canonical object; registering it
                    # again would duplicate writes/deletes in _build_operations.
                    self.values[name] = value
                else:
                    value.name = name
                    self.objects[name] = value
                if (
                    value.existing
                    and value.memory_type == "cases"
                    and "selected_case_ids" in self.context.operations_model.model_fields
                    and "case_sources" not in self.context.operations_model.model_fields
                ):
                    self.selected_cases.append(value)
            elif isinstance(value, _FieldHandle):
                self._error(
                    node,
                    "field edits (obj.field.overwrite/edit/drop) must be standalone statements",
                )
            else:
                if _contains_memory_object(value):
                    self._error(node, "memory objects cannot be stored in local containers")
                self.values[name] = value
            return
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            result = self._eval_call(node.value, statement=True)
            if isinstance(result, _MemoryObject):
                # An unassigned create/set is valid but intentionally cannot be referenced later.
                self.objects[f"__created_{result.page_id}"] = result
            elif isinstance(result, _FieldHandle):
                self._apply_field_handle(result, node)
            elif result is not None:
                self._error(node, "only SDK and memory-object calls may be standalone statements")
            return
        self._error(node, "only assignments and SDK/object calls are allowed")

    def _reserve_name(self, name: str, node: ast.AST) -> None:
        _require_identifier(name, "variable")
        if name in self._input_value_names and name != "sdk" and name not in _SAFE_BUILTINS:
            # A failed read or plain result is not an existing object. The final
            # program may reuse that name for a newly created memory once.
            self._input_value_names.remove(name)
            self.values.pop(name)
        if name == "sdk" or name in _SAFE_BUILTINS or name in self.objects or name in self.values:
            self._error(node, f"variable {name!r} is reserved or already assigned")

    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.objects:
                return self.objects[node.id]
            if node.id in self.values:
                return self.values[node.id]
            self._error(node, f"unknown name {node.id!r}")
        if isinstance(node, ast.List):
            return [self._eval(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            result = {}
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None:
                    expanded = self._eval(value)
                    if not isinstance(expanded, dict):
                        self._error(node, "dictionary unpacking requires a dict")
                    result.update(expanded)
                else:
                    result[self._eval(key)] = self._eval(value)
            return result
        if isinstance(node, ast.Attribute):
            owner = self._eval(node.value)
            if not isinstance(owner, _MemoryObject):
                self._error(node, "attribute reads are only allowed on memory objects")
            # Field names use identifier aliases on the DSL surface; map back to
            # the real schema field name before resolving the write handle.
            alias_map = self._field_alias_to_real.get(owner.memory_type, {})
            real_field = alias_map.get(node.attr, node.attr)
            if node.attr.startswith("_") or real_field not in owner.fields:
                available = (
                    ", ".join(sorted(_identifier_alias(name) for name in owner.fields)) or "(none)"
                )
                hint = (
                    " Use the real field name (e.g. content), not the literal word 'field'."
                    if node.attr == "field"
                    else ""
                )
                self._error(
                    node,
                    f"memory field {node.attr!r} is unavailable; editable fields on "
                    f"{owner.name or owner.memory_type}: {available}.{hint}",
                )
            # obj.<field> is a write handle, not the raw value: it exposes
            # .overwrite()/.edit()/.drop() and cannot be read as a string.
            return _FieldHandle(owner=owner, field_name=real_field)
        if isinstance(node, ast.Subscript):
            return self._eval(node.value)[self._eval_slice(node.slice)]
        if isinstance(node, ast.JoinedStr):
            return "".join(
                self._eval_formatted_value(part)
                if isinstance(part, ast.FormattedValue)
                else part.value
                for part in node.values
            )
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node)
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return +value
            self._error(node, "unsupported unary operator")
        if isinstance(node, ast.BoolOp):
            value = self._eval(node.values[0])
            for operand in node.values[1:]:
                if isinstance(node.op, ast.And):
                    if not value:
                        return value
                elif value:
                    return value
                value = self._eval(operand)
            return value
        if isinstance(node, ast.Compare):
            return self._eval_compare(node)
        if isinstance(node, ast.IfExp):
            return self._eval(node.body if self._eval(node.test) else node.orelse)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        self._error(node, f"unsupported expression {type(node).__name__}")

    def _eval_call(self, node: ast.Call, *, statement: bool = False) -> Any:
        if any(keyword.arg is None for keyword in node.keywords):
            self._error(node, "**kwargs is not allowed in calls")
        if isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_BUILTINS:
                self._error(
                    node,
                    f"function {node.func.id!r} is not allowed",
                    allow_tool_retry=node.func.id in ("read", "search"),
                )
            args = [self._eval(arg) for arg in node.args]
            kwargs = {item.arg: self._eval(item.value) for item in node.keywords}
            if _contains_memory_object((args, kwargs)):
                self._error(node, "memory objects cannot be converted or placed in containers")
            return _SAFE_BUILTINS[node.func.id](*args, **kwargs)
        if not isinstance(node.func, ast.Attribute):
            self._error(node, "unsupported call target")
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "sdk":
            return self._call_sdk(node, statement=statement)
        owner = self._eval(node.func.value)
        if isinstance(owner, _MemoryObject):
            return self._call_memory(owner, node.func.attr, node, statement=statement)
        if isinstance(owner, _FieldHandle):
            return self._call_field(owner, node.func.attr, node)
        if isinstance(owner, str) and node.func.attr in _SAFE_STRING_METHODS:
            args = [self._eval(arg) for arg in node.args]
            kwargs = {item.arg: self._eval(item.value) for item in node.keywords}
            if node.func.attr == "join" and args and isinstance(args[0], (list, tuple)):
                pieces = [str(piece) for piece in args[0]]
                projected = sum(len(piece.encode("utf-8")) for piece in pieces) + (
                    len(owner.encode("utf-8")) * max(len(pieces) - 1, 0)
                )
                if projected > _MAX_EXPRESSION_SIZE:
                    self._error(
                        node,
                        f"joined string would exceed the {_MAX_EXPRESSION_SIZE:,}-character size limit",
                    )
            elif node.func.attr == "replace" and len(args) >= 2:
                # str.replace can inflate the result well beyond its input when the
                # replacement is longer than the search text and occurs many times.
                # Bound the projected size BEFORE calling replace so a huge result is
                # never allocated (str.replace builds the whole string in C first).
                count = args[2] if len(args) > 2 else kwargs.get("count")
                self._check_replace_size(node, owner, args[0], args[1], count)
            return getattr(owner, node.func.attr)(*args, **kwargs)
        self._error(node, f"method {node.func.attr!r} is not allowed")

    def _check_replace_size(
        self, node: ast.AST, source: str, old: Any, new: Any, count: Any
    ) -> None:
        if not isinstance(old, str) or not isinstance(new, str):
            self._error(node, "replace() expects string arguments")
        old_bytes = len(old.encode("utf-8"))
        new_bytes = len(new.encode("utf-8"))
        source_bytes = len(source.encode("utf-8"))
        if new_bytes <= old_bytes:
            # Replacement is not longer than the match: result cannot grow.
            return
        if old == "":
            # "".replace inserts new between every character (len+1 positions).
            occurrences = len(source) + 1
        else:
            occurrences = source.count(old)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            occurrences = min(occurrences, count)
        elif count is not None:
            self._error(node, "replace() count must be a non-negative integer")
        projected = source_bytes + occurrences * (new_bytes - old_bytes)
        if projected > _MAX_EXPRESSION_SIZE:
            self._error(
                node,
                f"replaced string would exceed the {_MAX_EXPRESSION_SIZE:,}-character size limit",
            )

    def _call_sdk(self, node: ast.Call, *, statement: bool) -> Any:
        method = node.func.attr
        if method.startswith(("create_", "set_")):
            verb, type_alias = method.split("_", 1)
            if node.args:
                self._error(node, f"{verb} methods accept keyword arguments only")
            memory_type = self._type_alias_to_real.get(type_alias, type_alias)
            schema = self.schemas.get(memory_type)
            if schema is None:
                self._error(node, f"memory type {type_alias!r} is not available")
            expected_verb = "create" if schema.filename_has_variables() else "set"
            if verb != expected_verb:
                self._error(
                    node,
                    f"sdk.{method}() is unavailable for {type_alias!r}; "
                    f"use sdk.{expected_verb}_{type_alias}()",
                )
            allowed_fields = {
                name for name, _type, _description in _protocol_fields(self.context, schema)
            }
            kwargs = self._eval_keywords(
                node,
                allowed=allowed_fields,
                ignore_unknown=True,
                alias_map=self._field_alias_to_real.get(memory_type),
            )
            if _contains_memory_object(kwargs):
                self._error(node, "memory objects cannot be used as business field values")
            if any(isinstance(value, _FieldHandle) for value in kwargs.values()):
                action = "creating" if verb == "create" else "setting"
                self._error(
                    node,
                    f"field edits (obj.field.edit/drop) cannot be used when {action} a memory; "
                    "pass the complete field value",
                )
            for field in schema.fields:
                if field.name not in kwargs and field.init_value is not None:
                    kwargs[field.name] = field.init_value
            missing_fields = [field.name for field in schema.fields if field.name not in kwargs]
            if missing_fields:
                self._error(
                    node,
                    f"{verb} requires complete memory fields; missing: "
                    + ", ".join(missing_fields),
                )
            if verb == "set":
                identity = _memory_identity(schema, kwargs)
                identity_key = (memory_type, identity)
                if identity_key in self._singleton_identities:
                    identity_text = _format_identity(identity)
                    peer_id = _normalized_peer_scope(kwargs.get("peer_id"))
                    target = f"peer {peer_id!r}" if peer_id is not None else "self"
                    self._error(
                        node,
                        f"duplicate {type_alias} identity ({identity_text}); "
                        f"sdk.set_{type_alias}() may be called only once for {target}; "
                        "combine all people and content sections for that target into "
                        "one content argument",
                    )
                self._singleton_identities.add(identity_key)
            obj = _MemoryObject(
                name="",
                memory_type=memory_type,
                page_id=self._next_new_page_id,
                fields=dict(kwargs),
                existing=False,
                changed_fields=dict(kwargs),
            )
            self._next_new_page_id = self._find_next_new_page_id(self._next_new_page_id + 1)
            return obj
        if method == "commit":
            kwargs = self._eval_keywords(node)
            allowed = (
                {"cases"}
                if "selected_case_ids" in self.context.operations_model.model_fields
                else set()
            )
            if "case_sources" in self.context.operations_model.model_fields:
                allowed.add("case_sources")
            if node.args or set(kwargs) - allowed or not statement:
                self._error(
                    node,
                    "sdk.commit() accepts only an optional cases= list when Case selection is enabled",
                )
            selected = kwargs.get("cases", [])
            if not isinstance(selected, list) or any(
                not isinstance(obj, _MemoryObject)
                or obj.memory_type != "cases"
                or not obj.existing
                or obj.deleted
                for obj in selected
            ):
                self._error(
                    node, "cases= must contain only loaded, non-deleted existing Case objects"
                )
            if "cases" in kwargs:
                self.selected_cases = selected
                self._explicit_case_selection = True
            bindings = kwargs.get("case_sources", [])
            if not isinstance(bindings, list):
                self._error(node, "case_sources= must be a list of (Case object, patch IDs) tuples")
            for binding in bindings:
                if not isinstance(binding, (list, tuple)) or len(binding) != 2:
                    self._error(node, "Each case_sources entry must be (Case object, patch IDs)")
                obj, patch_ids = binding
                if not isinstance(obj, _MemoryObject) or obj.memory_type != "cases" or obj.deleted:
                    self._error(node, "case_sources targets must be surviving Case objects")
                self.case_sources.append({"case_id": obj.page_id, "patch_ids": patch_ids})
            if self._committed:
                self._error(node, "sdk.commit() may appear only once")
            self._committed = True
            return None
        if method == "existing":
            self._error(node, "sdk.existing() is reserved for system-provided bindings")
        if method == "error":
            self._error(
                node,
                "sdk.error() is reserved for system-provided tool results; "
                "if there are no memory changes, return sdk.commit()",
            )
        self._error(
            node,
            f"unknown SDK method sdk.{method}()",
            allow_tool_retry=(
                method in ("read", "search", "search_by_session", "ls")
                or method.startswith("read_")
            ),
        )

    def _call_memory(
        self, owner: _MemoryObject, method: str, node: ast.Call, *, statement: bool
    ) -> None:
        if not statement:
            self._error(node, f"{method}() must be a standalone statement")
        if not owner.existing and not owner.name:
            self._error(node, "assign a new memory object before calling methods on it")
        if method == "update":
            if node.args:
                self._error(node, "update() accepts keyword arguments only")
            schema = self.schemas[owner.memory_type]
            allowed_fields = {
                name for name, _type, _description in _protocol_fields(self.context, schema)
            }
            kwargs = self._eval_keywords(
                node,
                allowed=allowed_fields,
                ignore_unknown=True,
                alias_map=self._field_alias_to_real.get(owner.memory_type),
            )
            if _contains_memory_object(kwargs):
                self._error(node, "memory objects cannot be used as business field values")
            if any(isinstance(value, _FieldHandle) for value in kwargs.values()):
                self._error(
                    node,
                    "to edit a string field in place use obj.field.edit(...)/obj.field.drop(...); "
                    "obj.update() only takes complete field values",
                )
            self._update(owner, kwargs, node)
            return None
        if method == "delete":
            kwargs = self._eval_keywords(node)
            if node.args or set(kwargs) - {"replacement"}:
                self._error(node, "delete() accepts only replacement=")
            if not owner.existing:
                self._error(node, "a memory created in this program cannot be deleted")
            if self.schemas[owner.memory_type].operation_mode == "add_only":
                # Append-only history is authoritative: an attempted deletion can
                # never take effect. Treat it as a compatibility no-op so one
                # hallucinated cleanup statement does not discard valid creates
                # from the rest of the extraction program.
                return None
            replacement = kwargs.get("replacement")
            if replacement is not None and not isinstance(replacement, _MemoryObject):
                self._error(node, "delete replacement must be a memory object")
            if replacement is not None and not replacement.existing and not replacement.name:
                self._error(node, "assign a new replacement object before referencing it")
            if replacement is not None and replacement.memory_type != owner.memory_type:
                self._error(node, "delete replacement must have the same memory type")
            if replacement is owner:
                self._error(node, "a memory cannot replace itself")
            owner.deleted = True
            owner.replacement = replacement
            return None
        if method == "link":
            kwargs = self._eval_keywords(node)
            if not self.context.link_enabled:
                self._error(node, "links are disabled by configuration")
            if len(node.args) != 1:
                self._error(node, "link() requires exactly one target memory object")
            target = self._eval(node.args[0])
            if not isinstance(target, _MemoryObject):
                self._error(node, "link target must be a memory object")
            if not target.existing and not target.name:
                self._error(node, "assign a new link target before referencing it")
            allowed = {"link_type", "weight", "match_text", "description"}
            if set(kwargs) - allowed:
                self._error(node, "link() received unsupported arguments")
            self.links.append(
                {
                    "f": owner.page_id,
                    "t": target.page_id,
                    "link_type": kwargs.get("link_type", "related_to"),
                    "weight": kwargs.get("weight", 0.5),
                    "match_text": kwargs.get("match_text"),
                    "description": kwargs.get("description", ""),
                }
            )
            return None
        self._error(
            node,
            f"unknown memory method {method}()",
            allow_tool_retry=method in ("read", "search"),
        )

    def _call_field(self, handle: _FieldHandle, method: str, node: ast.Call) -> _FieldHandle:
        if method == "overwrite":
            kwargs = self._eval_keywords(node)
            if kwargs or len(node.args) != 1:
                self._error(
                    node,
                    "field.overwrite() takes exactly one positional argument: the complete new value",
                )
            value = self._eval(node.args[0])
            if _contains_memory_object(value) or isinstance(value, _FieldHandle):
                self._error(node, "field.overwrite() value must be a plain string")
            if handle.blocks or handle.full_value is not _UNSET:
                self._error(node, "field.overwrite() cannot be chained with edit()/drop()")
            handle.full_value = value
            return handle
        if method in ("edit", "drop"):
            if handle.full_value is not _UNSET:
                self._error(node, "field.edit()/drop() cannot be chained after overwrite()")
            kwargs = self._eval_keywords(node)
            if method == "edit":
                if node.args or set(kwargs) != {"search", "replace"}:
                    self._error(
                        node,
                        "field.edit() requires exactly search= and replace= keyword arguments: "
                        "obj.field.edit(search=..., replace=...)",
                    )
                block: Any = SearchReplaceBlock(search=kwargs["search"], replace=kwargs["replace"])
            else:
                if node.args or set(kwargs) != {"text"}:
                    self._error(
                        node, "field.drop() requires exactly text=: obj.field.drop(text=...)"
                    )
                block = DeleteBlock(delete=kwargs["text"])
            if _contains_memory_object(block):
                self._error(node, "memory objects cannot be used as patch text")
            handle.blocks.append(block)
            return handle
        self._error(
            node,
            f"unknown field method {method}(); use obj.field.overwrite(...), "
            "obj.field.edit(search=..., replace=...), or obj.field.drop(text=...)",
        )

    def _apply_field_handle(self, handle: _FieldHandle, node: ast.AST) -> None:
        owner = handle.owner
        name = handle.field_name
        schema = self.schemas[owner.memory_type]
        field_schema = {item.name: item for item in schema.fields}.get(name)
        if handle.full_value is _UNSET and not handle.blocks:
            self._error(node, f"{owner.name}.{name} edit has no overwrite()/edit()/drop() call")
        if (
            field_schema is not None
            and field_schema.merge_op == MergeOp.IMMUTABLE
            and owner.existing
            and ImmutableOp.is_set(owner.fields.get(name))
        ):
            # Preserve established values, but allow filling a blank template field.
            return
        if handle.full_value is not _UNSET:
            owner.fields[name] = handle.full_value
            owner.changed_fields[name] = handle.full_value
            return
        # Local edit chain: only valid for existing PATCH string fields.
        if not owner.existing:
            self._error(
                node,
                "field.edit()/drop() can only patch an existing memory; a memory created in "
                "this program has no prior content — pass the complete value to field.overwrite()",
            )
        if field_schema is None or not (
            field_schema.merge_op == MergeOp.PATCH and field_schema.field_type == FieldType.STRING
        ):
            self._error(node, f"field {name!r} does not support edit()/drop()")
        # Do NOT apply here. Store the edits as a StrPatch so python mode flows through
        # the same resolve_operations -> _validate_patch_operations -> patch-repair path
        # as json mode: a failed snippet is isolated to its own operation and gets the
        # structured patch-repair retry instead of failing the whole program.
        # Accumulate across multiple obj.field.edit()/drop() statements on the same field.
        existing = owner.changed_fields.get(name)
        prior_blocks = existing.blocks if isinstance(existing, StrPatch) else []
        patch = StrPatch(blocks=[*prior_blocks, *handle.blocks])
        owner.fields[name] = patch
        owner.changed_fields[name] = patch

    def _eval_keywords(
        self,
        node: ast.Call,
        *,
        allowed: set[str] | None = None,
        ignore_unknown: bool = False,
        alias_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for item in node.keywords:
            name = item.arg
            if alias_map is not None:
                # Field names use identifier aliases on the DSL surface; map back
                # to the real schema field name before validating/storing.
                name = alias_map.get(name, name)
            if name in values:
                self._error(node, f"keyword argument {name!r} was provided more than once")
            if allowed is not None and name not in allowed and ignore_unknown:
                continue
            values[name] = self._eval(item.value)
        return values

    def _update(self, owner: _MemoryObject, changes: dict[str, Any], node: ast.AST) -> None:
        schema = self.schemas[owner.memory_type]
        fields = {item.name: item for item in schema.fields}
        dynamic_fields = {
            name for name, _type, _description in _protocol_fields(self.context, schema)
        } - fields.keys()
        owner.deleted = False
        owner.replacement = None
        for name, value in changes.items():
            field_schema = fields.get(name)
            if name in dynamic_fields:
                if not owner.existing:
                    owner.fields[name] = value
                    owner.changed_fields[name] = value
                continue
            if field_schema is None:
                continue
            current = owner.fields.get(name)
            if (
                field_schema.merge_op == MergeOp.IMMUTABLE
                and owner.existing
                and ImmutableOp.is_set(current)
            ):
                continue
            if field_schema.merge_op == MergeOp.SUM:
                value = (current or 0) + value
                previous_delta = owner.changed_fields.get(name, 0)
                owner.changed_fields[name] = previous_delta + changes[name]
                owner.fields[name] = value
                continue
            owner.fields[name] = value
            owner.changed_fields[name] = value

    def _build_operations(self) -> Any:
        payload: dict[str, Any] = {name: [] for name in self.context.operations_model.model_fields}
        deleted_ids: set[int] = set()
        replacement_ids: dict[int, int] = {}
        for obj in self.objects.values():
            if obj.deleted:
                deleted_ids.add(obj.page_id)
                if obj.replacement is not None:
                    replacement_ids[obj.page_id] = obj.replacement.page_id
                payload.setdefault("delete_ids", []).append(
                    {
                        "delete_page_id": obj.page_id,
                        "replacement_page_id": (
                            obj.replacement.page_id if obj.replacement is not None else None
                        ),
                    }
                )
                continue
            if not obj.changed_fields:
                continue
            fields = dict(obj.changed_fields)
            if obj.existing:
                schema = self.schemas[obj.memory_type]
                for memory_field in schema.fields:
                    if memory_field.merge_op == MergeOp.IMMUTABLE:
                        if memory_field.name in obj.fields:
                            fields[memory_field.name] = obj.fields[memory_field.name]
                        elif memory_field.init_value is not None:
                            # Hidden server-managed fields are absent from legacy
                            # files and model-visible bindings. Supply their schema
                            # default so an otherwise valid update can be parsed;
                            # the authoritative server value is injected later.
                            fields[memory_field.name] = memory_field.init_value
            payload[obj.memory_type].append({"page_id": obj.page_id, **fields})
        if self.context.link_enabled:
            payload["links"] = []
            for link in self.links:
                remapped = dict(link)
                skip = False
                for endpoint in ("f", "t"):
                    replacement_id = self._resolve_replacement(
                        remapped[endpoint], deleted_ids, replacement_ids
                    )
                    if replacement_id is None:
                        skip = True
                        break
                    remapped[endpoint] = replacement_id
                if not skip and remapped["f"] != remapped["t"]:
                    payload["links"].append(remapped)
        if "selected_case_ids" in payload:
            # Changed Cases use the normal write path; deleted aliases must not reselect them.
            payload["selected_case_ids"] = list(
                dict.fromkeys(
                    obj.page_id
                    for obj in self.selected_cases
                    if self._explicit_case_selection or (not obj.deleted and not obj.changed_fields)
                )
            )
        if "case_sources" in payload:
            payload["case_sources"] = self.case_sources
        try:
            operations = self.context.operations_model.model_validate(payload)
            self.context.create_page_ids = {
                obj.page_id
                for obj in self.objects.values()
                if not obj.existing
                and not obj.deleted
                and obj.changed_fields
                and self.schemas[obj.memory_type].filename_has_variables()
            }
            return operations
        except Exception as exc:
            raise ExtractionOutputProtocolError(
                f"SDK operations failed schema validation: {exc}"
            ) from exc

    @staticmethod
    def _resolve_replacement(
        page_id: int, deleted_ids: set[int], replacement_ids: dict[int, int]
    ) -> int | None:
        visited = set()
        while page_id in deleted_ids:
            if page_id in visited:
                return None
            visited.add(page_id)
            replacement_id = replacement_ids.get(page_id)
            if replacement_id is None:
                return None
            page_id = replacement_id
        return page_id

    def _eval_slice(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Slice):
            return slice(
                self._eval(node.lower) if node.lower else None,
                self._eval(node.upper) if node.upper else None,
                self._eval(node.step) if node.step else None,
            )
        return self._eval(node)

    def _eval_formatted_value(self, node: ast.FormattedValue) -> str:
        value = self._eval(node.value)
        if _contains_memory_object(value):
            self._error(node, "memory objects cannot be formatted")
        if node.conversion == ord("r"):
            value = repr(value)
        elif node.conversion == ord("a"):
            value = ascii(value)
        elif node.conversion == ord("s"):
            value = str(value)
        elif node.conversion != -1:
            self._error(node, "unsupported f-string conversion")
        # A format spec (e.g. {x:>1000001}) can turn a small integer literal into
        # an arbitrarily large padded string with no repeat operator. Memory
        # content never needs printf-style alignment/width, so disallow any
        # non-empty format spec instead of trying to bound width inflation.
        if node.format_spec is not None:
            spec = self._eval(node.format_spec)
            if spec != "":
                self._error(
                    node,
                    "f-string format specs are not allowed; use plain {value} without a "
                    "':' width/format spec",
                )
        return format(value, "")

    def _eval_binop(self, node: ast.BinOp) -> Any:
        if isinstance(node.op, ast.Mult):
            return self._eval_mult(node)
        left, right = self._eval(node.left), self._eval(node.right)
        if isinstance(node.op, ast.Mod) and isinstance(left, (str, bytes)):
            # str/bytes % formatting (e.g. "%1000001s" % "x") can inflate a small
            # literal into a huge padded string via a width field, with no repeat
            # operator. Memory content never needs printf formatting, so disallow it.
            self._error(
                node,
                "string %-formatting is not allowed; build content with plain literals",
            )
        operations = {
            ast.Add: lambda: self._checked_concat(node, left, right),
            ast.Sub: lambda: left - right,
            ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
        }
        operation = operations.get(type(node.op))
        if operation is None:
            self._error(node, "unsupported binary operator")
        return operation()

    def _sequence_repeat_size(self, sequence: Any, count: Any, node: ast.AST) -> int:
        if not isinstance(count, int) or isinstance(count, bool):
            self._error(node, "a repeated literal must be multiplied by an integer count")
        if count < 0:
            self._error(node, "a repeated literal count cannot be negative")
        if isinstance(sequence, str):
            unit = len(sequence.encode("utf-8"))
        elif isinstance(sequence, (bytes, bytearray)):
            unit = len(sequence)
        else:
            unit = len(sequence)
        return unit * count

    def _checked_repeat(self, node: ast.AST, sequence: Any, count: Any) -> Any:
        if not isinstance(sequence, _REPEATABLE_SEQUENCES):
            self._error(node, "multiplication is only supported for repeating a literal")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            self._error(node, "a repeated literal must be multiplied by a non-negative integer")
        if self._sequence_repeat_size(sequence, count, node) > _MAX_EXPRESSION_SIZE:
            self._error(
                node,
                f"repeated literal would exceed the {_MAX_EXPRESSION_SIZE:,}-character size limit",
            )
        return sequence * count

    def _eval_mult(self, node: ast.BinOp) -> Any:
        left_node, right_node = node.left, node.right
        # Repeat a literal sequence by an integer count. Both a bare literal
        # and a literal nested inside another repeatable expression
        # (e.g. ["..."] * n inside "".join(...)) are accepted; evaluate only
        # after the would-be result size is bounded, before any allocation.
        left_seq, left_count = self._repeat_operands(left_node, right_node)
        right_seq, right_count = self._repeat_operands(right_node, left_node)
        if left_seq is not None:
            return self._checked_repeat(node, left_seq, left_count)
        if right_seq is not None:
            return self._checked_repeat(node, right_seq, right_count)
        self._error(
            node,
            "multiplication is only supported to repeat a string/list literal "
            "by a non-negative integer count",
        )

    def _repeat_operands(self, seq_node: ast.AST, count_node: ast.AST) -> tuple[Any, Any]:
        if isinstance(seq_node, ast.Constant) and isinstance(seq_node.value, _REPEATABLE_SEQUENCES):
            return seq_node.value, self._eval(count_node)
        # A 1-element list literal (ast.List with one Constant element) is a
        # common way to build a repeatable argument for str.join.
        if (
            isinstance(seq_node, ast.List)
            and len(seq_node.elts) == 1
            and isinstance(seq_node.elts[0], ast.Constant)
        ):
            return [seq_node.elts[0].value], self._eval(count_node)
        return None, None

    def _checked_concat(self, node: ast.AST, left: Any, right: Any) -> Any:
        if isinstance(left, str) and isinstance(right, str):
            if len(left.encode("utf-8")) + len(right.encode("utf-8")) > _MAX_EXPRESSION_SIZE:
                self._error(
                    node,
                    f"concatenated string exceeds the {_MAX_EXPRESSION_SIZE:,}-character size limit",
                )
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) + len(right) > _MAX_EXPRESSION_SIZE:
                self._error(
                    node, f"concatenated list exceeds the {_MAX_EXPRESSION_SIZE:,}-item size limit"
                )
        return left + right

    def _eval_compare(self, node: ast.Compare) -> bool:
        left = self._eval(node.left)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = self._eval(comparator)
            if isinstance(operator, ast.Eq):
                matches = left == right
            elif isinstance(operator, ast.NotEq):
                matches = left != right
            elif isinstance(operator, ast.Lt):
                matches = left < right
            elif isinstance(operator, ast.LtE):
                matches = left <= right
            elif isinstance(operator, ast.Gt):
                matches = left > right
            elif isinstance(operator, ast.GtE):
                matches = left >= right
            elif isinstance(operator, ast.In):
                matches = left in right
            elif isinstance(operator, ast.NotIn):
                matches = left not in right
            elif isinstance(operator, ast.Is):
                matches = left is right
            elif isinstance(operator, ast.IsNot):
                matches = left is not right
            else:
                self._error(node, "unsupported comparison operator")
            if not matches:
                return False
            left = right
        return True

    def _find_next_new_page_id(self, candidate: int) -> int:
        while self.context.page_id_map.resolve(candidate) is not None:
            candidate += 1
        return candidate

    @staticmethod
    def _error(node: ast.AST, message: str, *, allow_tool_retry: bool = False) -> None:
        raise ExtractionOutputProtocolError(
            f"Line {getattr(node, 'lineno', 1)}: {message}",
            allow_tool_retry=allow_tool_retry,
        )


def _extract_python_code(content: str) -> str:
    stripped = str(content or "").strip()
    matches = list(_PYTHON_FENCE_RE.finditer(stripped))
    if len(matches) == 1:
        match = matches[0]
        surrounding_text = stripped[: match.start()] + stripped[match.end() :]
        if "```" not in surrounding_text:
            return match.group("code").rstrip()
    starts = list(_PYTHON_FENCE_START_RE.finditer(stripped))
    if len(starts) == 1 and stripped.count("```") == 1:
        return stripped[starts[0].end() :].strip()
    if "```" in stripped:
        raise ExtractionOutputProtocolError(
            "Python output may contain only one complete ```python code fence"
        )
    return stripped


def _format_retry_source_excerpt(content: str, error: str | None) -> str:
    """Show the compiler's source line and up to three neighbours on each side."""
    match = re.match(r"Line (\d+):", error or "")
    if not content or match is None:
        return ""
    try:
        # Use exactly the same fence/whitespace normalization as parse(), so its
        # line numbers also work for fenced and incomplete-fence responses.
        code = _extract_python_code(content).replace("\r\n", "\n").replace("\r", "\n")
        # str.splitlines() also splits Unicode separators inside comments/strings,
        # whereas Python source line numbers count only physical newlines.
        lines = code.split("\n")
        line_index = int(match.group(1)) - 1
    except ValueError:
        return ""
    if not 0 <= line_index < len(lines):
        return ""

    start, end = max(0, line_index - 3), min(len(lines), line_index + 4)
    header = (
        "\n\nPrevious failed program excerpt "
        "(line numbers and '>' are annotations, not Python):\n```text\n"
    )
    footer = "\n```"
    width = len(str(end))
    # Bound every line before joining; one huge SDK string must not consume the
    # budget or hide the marked error line and its neighbouring lines.
    line_budget = (
        (_MAX_FORMAT_RETRY_EXCERPT_CHARS - len(header) - len(footer)) // (end - start) - width - 6
    )
    excerpt = []
    marker = " ... [truncated] ... "
    for index in range(start, end):
        text = lines[index]
        if len(text) > line_budget:
            head = (line_budget - len(marker)) // 2
            tail = line_budget - len(marker) - head
            text = text[:head] + marker + text[-tail:]
        indicator = ">" if index == line_index else " "
        excerpt.append(f"{indicator} {index + 1:>{width}} | {text}")
    return header + "\n".join(excerpt) + footer


def _is_string_literal_syntax_error(error: str) -> bool:
    """Heuristic: does this parse error point at a broken string literal?

    These are almost always caused by an apostrophe or quote inside prose that
    was wrapped in a single-/double-quoted literal instead of a triple-quoted
    one, so the retry should push hard on triple quotes.
    """
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in (
            "unterminated string literal",
            "unterminated triple-quoted string",
            "invalid syntax. perhaps you forgot a comma",
            "eol while scanning string literal",
            "invalid escape sequence",
        )
    )


def _format_syntax_error_source(exc: SyntaxError, *, max_len: int = 120) -> str:
    """Render the offending source line (and a caret) from a SyntaxError.

    Python's own message (e.g. "Perhaps you forgot a comma?") is often
    misleading for these programs; the real cause is usually a quote or newline
    inside prose that should have been triple-quoted. Showing the actual line
    lets the model see the break point instead of guessing.
    """
    text = (exc.text or "").rstrip("\n")
    if not text:
        return ""
    stripped = text.lstrip()
    lead = len(text) - len(stripped)
    offset = exc.offset
    caret_col = None
    if isinstance(offset, int) and offset >= 1:
        caret_col = max(0, offset - 1 - lead)
    if len(stripped) > max_len:
        stripped = stripped[:max_len] + "…"
        if caret_col is not None and caret_col > max_len:
            caret_col = None
    lines = [f"\n  {stripped}"]
    if caret_col is not None and caret_col <= len(stripped):
        lines.append("\n  " + " " * caret_col + "^")
    return "".join(lines)


def _looks_like_bare_markdown(code: str) -> bool:
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    return (
        bool(lines)
        and lines[0].startswith("# ")
        and any(line.startswith(("- ", "* ")) for line in lines[1:])
        and not any("sdk." in line for line in lines)
    )


def _visible_memory_fields(
    memory_file: MemoryFile,
    schema: MemoryTypeSchema,
    context: ExtractionOutputContext,
) -> dict[str, Any]:
    schema_fields = {name for name, _type, _description in _protocol_fields(context, schema)}
    fields = {
        key: value
        for key, value in dict(memory_file.extra_fields or {}).items()
        if key in schema_fields and key not in _HIDDEN_MEMORY_FIELDS and not key.startswith("_")
    }
    if "content" in schema_fields:
        fields["content"] = memory_file.plain_content()
    return fields


def _field_type_name(field_type: FieldType) -> str:
    return {
        FieldType.STRING: "str",
        FieldType.INT64: "int",
        FieldType.FLOAT32: "float",
        FieldType.BOOL: "bool",
    }.get(field_type, "object")


def _protocol_fields(
    context: ExtractionOutputContext, schema: MemoryTypeSchema
) -> list[tuple[str, str, str]]:
    static_fields = [
        (field.name, _field_type_name(field.field_type), field.description)
        for field in schema.fields
        if field.name not in _HIDDEN_MEMORY_FIELDS
    ]
    static_names = {field.name for field in schema.fields}
    operations_field = context.operations_model.model_fields.get(schema.memory_type)
    annotation = getattr(operations_field, "annotation", None)
    model_type = get_args(annotation)[0] if get_origin(annotation) is list else annotation
    dynamic_fields = getattr(model_type, "model_fields", {})
    extras = []
    for name, model_field in dynamic_fields.items():
        if name == "page_id" or name in static_names:
            continue
        # Non-identifier field names are aliased on the DSL surface (see
        # _identifier_alias), so they need not be valid Python identifiers here.
        extras.append(
            (name, _annotation_type_name(model_field.annotation), model_field.description or "")
        )
    return [*extras, *static_fields]


def _model_visible_identity_fields(
    context: ExtractionOutputContext, schema: MemoryTypeSchema
) -> tuple[str, ...]:
    visible_fields = {name for name, _type, _description in _protocol_fields(context, schema)}
    return tuple(name for name in schema.identity_fields() if name in visible_fields)


def _memory_identity(
    schema: MemoryTypeSchema, fields: dict[str, Any]
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (
            name,
            _normalized_peer_scope(fields.get(name)) if name == "peer_id" else fields.get(name),
        )
        for name in schema.identity_fields()
    )


def _format_identity(identity: tuple[tuple[str, Any], ...]) -> str:
    return ", ".join(
        f"{name}={'<self>' if name == 'peer_id' and value is None else value!r}"
        for name, value in identity
    )


def _annotation_type_name(annotation: Any) -> str:
    args = [item for item in get_args(annotation) if item is not type(None)]
    if args:
        annotation = args[0]
    return getattr(annotation, "__name__", "object")


def _normalized_peer_scope(value: Any) -> str | None:
    if value in (None, "", "__self"):
        return None
    return safe_peer_id(value)


def _contains_memory_object(value: Any) -> bool:
    if isinstance(value, _MemoryObject):
        return True
    if isinstance(value, dict):
        return any(
            _contains_memory_object(key) or _contains_memory_object(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_memory_object(item) for item in value)
    return False


def _require_identifier(value: str, label: str) -> None:
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(
            f"Python memory output requires {label} {value!r} to be a valid identifier"
        )


def _identifier_alias(name: str) -> str:
    """Map a schema name to a Python-identifier alias used only on the DSL surface.

    memory_type and field names keep their real value everywhere (URI, storage,
    JSON protocol); only the Python method/parameter names need to be valid
    identifiers, so non-identifier characters are folded to underscores.
    """
    alias = re.sub(r"\W", "_", name)
    if alias and alias[0].isdigit():
        alias = f"_{alias}"
    if not alias:
        alias = "_"
    if keyword.iskeyword(alias):
        alias = f"{alias}_"
    return alias


def _validate_alias_uniqueness(schemas: tuple[MemoryTypeSchema, ...]) -> None:
    """Reject schemas whose names collide after identifier-alias folding.

    Aliasing is not one-to-one (e.g. 'project-notes' and 'project_notes' both
    fold to 'project_notes'). A collision would make the DSL surface ambiguous
    and silently route calls to the wrong schema/field, so fail loudly at build
    time instead. Distinct-identifier names never collide, so real configs are
    unaffected.
    """
    type_seen: dict[str, str] = {}
    for schema in schemas:
        alias = _identifier_alias(schema.memory_type)
        if alias in type_seen and type_seen[alias] != schema.memory_type:
            raise ValueError(
                f"memory_type {type_seen[alias]!r} and {schema.memory_type!r} produce the same "
                f"Python DSL alias {alias!r}; rename one to a distinct identifier"
            )
        type_seen[alias] = schema.memory_type

        field_seen: dict[str, str] = {}
        for name in {memory_field.name for memory_field in schema.fields}:
            field_alias = _identifier_alias(name)
            if field_alias in field_seen and field_seen[field_alias] != name:
                raise ValueError(
                    f"fields {field_seen[field_alias]!r} and {name!r} in memory_type "
                    f"{schema.memory_type!r} produce the same Python DSL alias {field_alias!r}; "
                    "rename one to a distinct identifier"
                )
            field_seen[field_alias] = name
