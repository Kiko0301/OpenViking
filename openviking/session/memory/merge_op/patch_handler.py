# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Pi's text-edit engine, adapted from TypeScript to memory-field strings.

Source: earendil-works/pi@482c1381a00d8d00bc81697e4a980c593db6803e,
packages/coding-agent/src/core/tools/edit-diff.ts and edit.ts.
StrPatch blocks and the memory URI/field adapt Pi to the memory SDK. Matching,
original-snapshot batch semantics, and normalization of touched lines follow
that pinned source. Diagnostics use the SDK parameter names search/replace.

The upstream portions are used under the MIT License:

Copyright (c) 2025 Mario Zechner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import pairwise

from openviking.session.memory.merge_op.base import StrPatch


class PatchParseError(ValueError):
    """Pi edit failure, with metadata for ExtractLoop's structured tool feedback."""

    def __init__(
        self,
        message: str,
        *,
        edit_index: int | None = None,
        reason: str = "apply_error",
        match_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.edit_index = edit_index
        self.reason = reason
        self.match_count = match_count


def detect_line_ending(content: str) -> str:
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


# ECMAScript trimEnd, not Python rstrip (which also strips e.g. U+0085).
_TRAILING_WHITESPACE = re.compile(
    r"[\t\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]+$"
)
_FUZZY_TRANSLATION = str.maketrans(
    {
        **dict.fromkeys("\u2018\u2019\u201a\u201b", "'"),
        **dict.fromkeys("\u201c\u201d\u201e\u201f", '"'),
        **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"),
        **dict.fromkeys(
            "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000", " "
        ),
    }
)


def normalize_for_fuzzy_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "\n".join(_TRAILING_WHITESPACE.sub("", line) for line in text.split("\n"))
    return text.translate(_FUZZY_TRANSLATION)


@dataclass(slots=True)
class _Match:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool


def fuzzy_find_text(content: str, search: str) -> _Match:
    index = content.find(search)
    if index != -1:
        return _Match(True, index, len(search), False)
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(search)
    index = fuzzy_content.find(fuzzy_old_text)
    if index == -1:
        return _Match(False, -1, 0, False)
    return _Match(True, index, len(fuzzy_old_text), True)


def _count_occurrences(content: str, search: str) -> int:
    content = normalize_for_fuzzy_match(content)
    search = normalize_for_fuzzy_match(search)
    if not search:
        # Preserve JS String.split("").length - 1, including normalization to empty.
        return len(content.encode("utf-16-le", errors="surrogatepass")) // 2 - 1
    return content.count(search)


@dataclass(slots=True)
class _MatchedEdit:
    edit_index: int
    match_index: int
    match_length: int
    replace: str


@dataclass(slots=True)
class _LineSpan:
    start: int
    end: int


@dataclass(slots=True)
class _ReplacementGroup:
    start_line: int
    end_line: int
    replacements: list[_MatchedEdit]


def _split_lines_with_endings(content: str) -> list[str]:
    return re.findall(r"[^\n]*\n|[^\n]+", content)


def _get_line_spans(content: str) -> list[_LineSpan]:
    offset = 0
    spans = []
    for line in _split_lines_with_endings(content):
        spans.append(_LineSpan(offset, offset + len(line)))
        offset += len(line)
    return spans


def _get_replacement_line_range(
    lines: list[_LineSpan], replacement: _MatchedEdit
) -> tuple[int, int]:
    start = replacement.match_index
    end = start + replacement.match_length
    start_line = next((i for i, line in enumerate(lines) if line.start <= start < line.end), -1)
    if start_line == -1:
        raise PatchParseError("Replacement range is outside the base content.")
    end_line = start_line
    while end_line < len(lines) and lines[end_line].end < end:
        end_line += 1
    if end_line >= len(lines):
        raise PatchParseError("Replacement range is outside the base content.")
    return start_line, end_line + 1


def _apply_replacements(content: str, replacements: list[_MatchedEdit], offset: int = 0) -> str:
    result = content
    for replacement in reversed(replacements):
        index = replacement.match_index - offset
        result = result[:index] + replacement.replace + result[index + replacement.match_length :]
    return result


def _apply_replacements_preserving_unchanged_lines(
    original_content: str, base_content: str, replacements: list[_MatchedEdit]
) -> str:
    original_lines = _split_lines_with_endings(original_content)
    base_lines = _get_line_spans(base_content)
    if len(original_lines) != len(base_lines):
        raise PatchParseError(
            "Cannot preserve unchanged lines because the base content has a different line count."
        )
    groups: list[_ReplacementGroup] = []
    for replacement in sorted(replacements, key=lambda edit: edit.match_index):
        start_line, end_line = _get_replacement_line_range(base_lines, replacement)
        current = groups[-1] if groups else None
        if current is not None and start_line < current.end_line:
            current.end_line = max(current.end_line, end_line)
            current.replacements.append(replacement)
        else:
            groups.append(_ReplacementGroup(start_line, end_line, [replacement]))
    original_line_index = 0
    result = []
    for group in groups:
        result.extend(original_lines[original_line_index : group.start_line])
        start_offset = base_lines[group.start_line].start
        end_offset = base_lines[group.end_line - 1].end
        result.append(
            _apply_replacements(
                base_content[start_offset:end_offset], group.replacements, start_offset
            )
        )
        original_line_index = group.end_line
    result.extend(original_lines[original_line_index:])
    return "".join(result)


def apply_edits_to_normalized_content(content: str, patch: StrPatch, path: str) -> str:
    """Match all edits against the same LF-normalized source, as Pi does."""
    edits = [
        (normalize_to_lf(block.search), normalize_to_lf(block.replace)) for block in patch.blocks
    ]
    total = len(edits)
    for index, (search, _) in enumerate(edits):
        if not search:
            label = "search" if total == 1 else f"edits[{index}].search"
            raise PatchParseError(
                f"{label} must not be empty in {path}.", edit_index=index, reason="empty_search"
            )
    initial_matches = [fuzzy_find_text(content, search) for search, _ in edits]
    used_fuzzy_match = any(match.used_fuzzy_match for match in initial_matches)
    replacement_base = normalize_for_fuzzy_match(content) if used_fuzzy_match else content
    matched_edits = []
    for index, (search, replace) in enumerate(edits):
        match = fuzzy_find_text(replacement_base, search)
        if not match.found:
            message = (
                f"Could not find the exact text in {path}. The search must match exactly "
                "including all whitespace and newlines."
                if total == 1
                else f"Could not find edits[{index}] in {path}. The search must match exactly "
                "including all whitespace and newlines."
            )
            raise PatchParseError(message, edit_index=index, reason="not_found", match_count=0)
        occurrences = _count_occurrences(replacement_base, search)
        if occurrences > 1:
            message = (
                f"Found {occurrences} occurrences of the text in {path}. The text must be unique. "
                "Please provide more context to make it unique."
                if total == 1
                else f"Found {occurrences} occurrences of edits[{index}] in {path}. "
                "Each search must be unique. Please provide more context to make it unique."
            )
            raise PatchParseError(
                message, edit_index=index, reason="non_unique", match_count=occurrences
            )
        matched_edits.append(_MatchedEdit(index, match.index, match.match_length, replace))
    matched_edits.sort(key=lambda edit: edit.match_index)
    for previous, current in pairwise(matched_edits):
        if previous.match_index + previous.match_length > current.match_index:
            raise PatchParseError(
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions.",
                edit_index=current.edit_index,
                reason="overlap",
            )
    new_content = (
        _apply_replacements_preserving_unchanged_lines(content, replacement_base, matched_edits)
        if used_fuzzy_match
        else _apply_replacements(replacement_base, matched_edits)
    )
    if content == new_content:
        message = (
            f"No changes made to {path}. The replacement produced identical content. This might "
            "indicate an issue with special characters or the text not existing as expected."
            if total == 1
            else f"No changes made to {path}. The replacements produced identical content."
        )
        raise PatchParseError(message, reason="no_change")
    return new_content


def apply_str_patch(original_content: str, patch: StrPatch, *, path: str = "string field") -> str:
    """Adapt a memory field to Pi's edit tool, including BOM and line endings."""
    if not patch.blocks:
        raise PatchParseError(
            "Edit tool input is invalid. edits must contain at least one replacement.",
            reason="empty_edits",
        )
    bom = "\ufeff" if original_content.startswith("\ufeff") else ""
    content = original_content[len(bom) :]
    ending = detect_line_ending(content)
    new_content = apply_edits_to_normalized_content(normalize_to_lf(content), patch, path)
    return bom + restore_line_endings(new_content, ending)
