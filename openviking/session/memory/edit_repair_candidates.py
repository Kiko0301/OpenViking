# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Diagnostic candidates only; never used to apply an Edit.

The normalization, Levenshtein score and single-line substring helpers below
come from the former patch_handler implementation (RooCode-derived). Its fuzzy
replacement/80-percent acceptance path is deliberately not restored.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from rapidfuzz.distance import Levenshtein

from openviking.session.memory.merge_op.patch_handler import (
    normalize_for_fuzzy_match,
    normalize_to_lf,
)


def normalize_string(text: str) -> str:
    """Normalize string by handling smart quotes and special characters."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u200e": "",
        "\u200f": "",
        "\ufeff": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def get_similarity(original: str, search: str) -> float:
    """Calculate similarity ratio between two strings (0 to 1)."""
    if search == "":
        return 0.0

    normalized_original = normalize_string(original)
    normalized_search = normalize_string(search)

    if normalized_original == normalized_search:
        return 1.0

    dist = Levenshtein.distance(normalized_original, normalized_search)
    max_length = max(len(normalized_original), len(normalized_search))

    return 1.0 - (dist / max_length) if max_length > 0 else 1.0


def _find_best_substring_match(line: str, search_str: str) -> tuple[float, str]:
    """Find the best matching substring in a line."""
    best_score = 0.0
    best_content = ""
    search_len = len(search_str)
    line_len = len(line)

    # If search string is longer than line, just compare the whole line
    if search_len >= line_len:
        return get_similarity(line, search_str), line

    # Try sliding window for best match (limit to reasonable checks for performance)
    # First check at start, end, and a few positions in between
    positions_to_check = [0, line_len - search_len]
    if line_len > search_len * 3:
        positions_to_check.append(line_len // 2 - search_len // 2)

    for i in positions_to_check:
        if 0 <= i <= line_len - search_len:
            substring = line[i : i + search_len]
            score = get_similarity(substring, search_str)
            if score > best_score:
                best_score = score
                best_content = substring

    # Also compare with the whole line as fallback
    whole_line_score = get_similarity(line, search_str)
    if whole_line_score > best_score:
        best_score = whole_line_score
        best_content = line

    return best_score, best_content


def edit_match_hints(original: str, search: str, *, max_chars: int = 6000) -> dict[str, Any] | None:
    """Suggest up to three verbatim source spans, with a shared caller text budget.

    The old matcher supplies ranking only. Pi remains the sole execution matcher.
    Bounded scanning is advisory: no candidates does not prove no repair exists.
    """
    if not search or max_chars <= 0:
        return None
    normalized_old = normalize_for_fuzzy_match(normalize_to_lf(search))
    normalized_source = normalize_for_fuzzy_match(normalize_to_lf(original))
    count = normalized_source.count(normalized_old) if normalized_old else 0
    if count == 1:
        return None  # Matching works; the batch may instead fail overlap/content validation.
    hints: dict[str, Any] = {
        "reason": "non_unique" if count > 1 else "not_found",
        "candidates": [],
    }
    if count > 1:
        hints["match_count"] = count
        return hints
    if len(search) > 2048:
        hints["search_limited"] = True
        return hints

    prefix = original[:100_000]
    raw_lines = prefix.splitlines(keepends=True)
    limited = len(prefix) < len(original) or len(raw_lines) > 2000
    raw_lines = raw_lines[:2000]
    lines = [line.removesuffix("\n").removesuffix("\r") for line in raw_lines]
    width = len(normalize_to_lf(search).split("\n"))
    substring_match = lru_cache(maxsize=128)(_find_best_substring_match)
    similarity = lru_cache(maxsize=128)(get_similarity)
    ranked = []
    for start in range(len(lines) - width + 1):
        if width == 1:
            score, text = substring_match(lines[start], search)
        else:
            # Rebuild from RAW lines, not normalized text, so CRLF/mixed endings
            # and indentation remain copyable byte-for-byte from original_text.
            if sum(len(line) for line in raw_lines[start : start + width]) > 2050:
                continue
            text = "".join(raw_lines[start : start + width - 1]) + lines[start + width - 1]
            score = similarity(text, search)
        if text and len(text) <= 2048 and score >= 0.5:
            ranked.append((score, start, text))

    used = 0
    seen = set()
    for score, start, text in sorted(ranked, key=lambda item: (-item[0], item[1])):
        if text in seen or used + len(text) > max_chars:
            continue
        seen.add(text)
        differences = []
        # Editops transforms the supplied search INTO the candidate source text.
        # At most six character edits are enough to expose common substitutions,
        # added/omitted characters, numbering and quote mistakes without a huge diff.
        edits = Levenshtein.editops(search, text)
        for tag, old_pos, source_pos in list(edits)[:6]:
            differences.append(
                {
                    "supplied": "" if tag == "insert" else search[old_pos : old_pos + 1],
                    "original": "" if tag == "delete" else text[source_pos : source_pos + 1],
                }
            )
        hints["candidates"].append(
            {
                "text": text,
                "start_line": start + 1,
                "end_line": start + width,
                "similarity": round(score, 4),
                "match_count": normalized_source.count(
                    normalize_for_fuzzy_match(normalize_to_lf(text))
                ),
                "differences": differences,
                "differences_truncated": len(edits) > 6,
            }
        )
        used += len(text)
        if len(hints["candidates"]) == 3:
            break
    if limited:
        hints["search_limited"] = True
    return hints
