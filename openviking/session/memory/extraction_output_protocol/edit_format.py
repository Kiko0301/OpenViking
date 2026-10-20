# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Pi Edit tool description/guidelines, adapted only to the memory-field SDK.

Source: earendil-works/pi@482c1381a00d8d00bc81697e4a980c593db6803e,
packages/coding-agent/src/core/tools/edit.ts. MIT copyright/permission notice is
preserved in merge_op/patch_handler.py alongside the ported editing engine.
File targets become bound memory fields; replacement arguments use search/replace.
One field's accumulated edit/drop calls form a single original-snapshot batch.
"""

from __future__ import annotations

EDIT_TOOL_DESCRIPTION = """Edit a single memory field using exact text replacement. Every search must match a unique, non-overlapping region of the original field. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.

Usage:
- Use edit for precise changes (search must match exactly).
- When changing multiple separate locations in one field, submit all replacements together in the same response.
- Each search is matched against the original field, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.
- Keep search as small as possible while still being unique in the field. Do not pad with large unchanged regions."""
