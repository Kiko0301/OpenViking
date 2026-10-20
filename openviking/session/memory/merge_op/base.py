# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Merge operation base classes and registry.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field


class FieldType(str, Enum):
    """Field type enumeration."""

    STRING = "string"
    INT64 = "int64"
    FLOAT32 = "float32"
    BOOL = "bool"


# ============================================================================
# Field Type Mapping (shared across all merge operations)
# ============================================================================

_FIELD_TYPE_TO_PYTHON: Dict[FieldType, Type[Any]] = {
    FieldType.STRING: str,
    FieldType.INT64: int,
    FieldType.FLOAT32: float,
    FieldType.BOOL: bool,
}


def get_python_type_for_field(field_type: FieldType, default: Type[Any] = str) -> Type[Any]:
    """Map FieldType to corresponding Python type.

    Args:
        field_type: The FieldType enum value
        default: Default type if field_type is not recognized

    Returns:
        Corresponding Python type (str, int, float, or bool)
    """
    return _FIELD_TYPE_TO_PYTHON.get(field_type, default)


# ============================================================================
# Structured Patch Models
# ============================================================================


class SearchReplaceBlock(BaseModel):
    """Single SEARCH/REPLACE block for string patches."""

    search: str = Field(
        ...,
        description="Exact, unique text in the original field; must not overlap another block.",
    )
    replace: str = Field(
        ...,
        description="Replacement text for this targeted edit.",
    )


class DeleteBlock(BaseModel):
    """Pi deletion: replace the selected literal text with an empty string."""

    delete: str = Field(
        ...,
        min_length=1,
        description="Exact, unique text to remove. Include the newline to remove a complete line.",
    )

    @property
    def search(self) -> str:
        """Expose the text through the common patch-block interface."""
        return self.delete

    @property
    def replace(self) -> str:
        """A delete is a replacement with an empty string."""
        return ""


class StrPatch(BaseModel):
    """Non-overlapping edits matched against the same original string field."""

    blocks: List[Union[SearchReplaceBlock, DeleteBlock]] = Field(
        default_factory=list,
        description="All blocks match the original field, not earlier edits. Do not overlap or nest blocks.",
    )

    def get_first_replace(self) -> Optional[str]:
        """Get the replace content from the first block.

        Useful when there's no original content to match against,
        so we use the replace content directly.

        Returns:
            The replace content from first block, or None if no blocks
        """
        if self.blocks:
            return self.blocks[0].replace
        return None


class MergeOp(str, Enum):
    """Merge operation enumeration."""

    PATCH = "patch"
    REPLACE = "replace"
    SUM = "sum"
    IMMUTABLE = "immutable"


class MergeOpBase(ABC):
    """Abstract base class for merge operations."""

    op_type: MergeOp

    @abstractmethod
    def get_output_schema_type(self, field_type: FieldType) -> Type[Any]:
        """Get the Python type for this merge operation's output schema.

        Args:
            field_type: The underlying field type

        Returns:
            Python type to use in the Pydantic schema
        """
        pass

    @abstractmethod
    def get_output_schema_description(self, field_description: str) -> str:
        """Get the description for this merge operation's output schema.

        Args:
            field_description: The original field description

        Returns:
            Description string to use in the Pydantic schema
        """
        pass

    @abstractmethod
    async def apply(self, current_value: Any, patch_value: Any) -> Any:
        """Apply this merge operation.

        Args:
            current_value: Current field value
            patch_value: Patch value from the operation

        Returns:
            New field value after applying the merge
        """
        pass
