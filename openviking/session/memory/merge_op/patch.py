# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Patch merge operation for strings, direct replacement for other field types."""

import asyncio
from typing import Any, Type

from openviking.session.memory.merge_op.base import (
    FieldType,
    MergeOp,
    MergeOpBase,
    StrPatch,
    get_python_type_for_field,
)


class PatchOp(MergeOpBase):
    """Apply SEARCH/REPLACE or DELETE blocks to strings."""

    op_type = MergeOp.PATCH

    def __init__(self, field_type: FieldType):
        self._field_type = field_type

    def get_output_schema_type(self, field_type: FieldType) -> Type[Any]:
        if field_type == FieldType.STRING:
            return StrPatch
        return get_python_type_for_field(field_type)

    def get_output_schema_description(self, field_description: str) -> str:
        if self._field_type == FieldType.STRING:
            return (
                f"PATCH operation for '{field_description}'. Follow the shared "
                "Edit instructions. DELETE is a replacement with empty text."
            )
        return f"Replace value for '{field_description}'"

    async def apply(
        self, current_value: Any, patch_value: Any, *, path: str = "string field"
    ) -> Any:
        """Apply one Pi batch; non-string and whole-value operations keep their schema behavior."""
        if self._field_type != FieldType.STRING:
            return patch_value
        if isinstance(patch_value, dict) and "blocks" in patch_value:
            patch_value = StrPatch.model_validate(patch_value)
        if current_value is None:
            return self._extract_replace_when_no_original(patch_value)
        if isinstance(patch_value, StrPatch):
            from openviking.session.memory.merge_op.patch_handler import apply_str_patch

            return await asyncio.to_thread(apply_str_patch, current_value, patch_value, path=path)
        if patch_value is None or patch_value == "":
            return current_value
        return patch_value

    async def apply_checked(
        self,
        current_value: Any,
        patch_value: Any,
        *,
        require_exact: bool = False,
        path: str = "string field",
    ) -> Any:
        """Use the same Pi batch engine, retaining the memory store's stale-read guard."""
        if self._field_type == FieldType.STRING and isinstance(patch_value, dict):
            patch_value = StrPatch.model_validate(patch_value)
        if require_exact and isinstance(patch_value, StrPatch) and current_value is not None:
            for index, block in enumerate(patch_value.blocks):
                match_count = current_value.count(block.search) if block.search else 0
                if match_count != 1:
                    raise ValueError(
                        f"edits[{index}] must match exactly once in {path} because the field "
                        f"changed since it was read; match_count={match_count}, "
                        f"search={block.search!r}, replace={block.replace!r}"
                    )
        return await self.apply(current_value, patch_value, path=path)

    def _extract_replace_when_no_original(self, patch_value: Any) -> Any:
        """
        Extract replace value from patch when there's no original content.

        Called when current_value is None - we use the replace content
        directly instead of trying to match against an empty string.

        Args:
            patch_value: The patch value (StrPatch, dict, or string)

        Returns:
            The replace content, or empty string if not available
        """
        # Case 1: StrPatch object — concatenate ALL blocks' replace content.
        # The schema instructs the model to split non-adjacent edits into
        # separate blocks, so a new memory routinely arrives as a multi-block
        # patch. Taking only blocks[0] would silently drop every subsequent
        # fact/preference the model extracted.
        if isinstance(patch_value, StrPatch):
            replaces = [b.replace for b in patch_value.blocks if b.replace is not None]
            return "\n".join(replaces) if replaces else ""

        # Complete field content for creation - use as is
        if isinstance(patch_value, str):
            return patch_value

        return ""
