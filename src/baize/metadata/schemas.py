"""Metadata API schema definitions.

Provides the data structures for describing resource fields, their types,
constraints, and available options — consumed by frontend to dynamically
render forms.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class FieldOption(BaseModel):
    """A single selectable option for select / multi_select fields."""

    value: str
    label: str
    disabled: bool = False


class FieldDescriptor(BaseModel):
    """Describes one form field.

    Attributes:
        key: Dot-notation path (e.g. ``guardrails.max_tool_calls``).
        label: Display label (Chinese).
        description: Help text shown under the field.
        type: Determines which frontend widget to render.
        required: Whether the field is mandatory.
        default: Default value for new records.
        constraints: Type-specific validation rules:
            text/textarea: ``{"min_length": int, "max_length": int}``
            int: ``{"min": int, "max": int}``
            multi_select: ``{"max_items": int}``
        options: Pre-resolved selectable options (always populated for select types).
        options_ref: Semantic hint that options are dynamic (e.g. ``"llm_models_chat"``).
            Frontend may show a refresh button for such fields.
        options_source: API endpoint URL for dynamically loading options at runtime
            (e.g. ``"/api/v1/knowledge-bases"``).
    """

    key: str
    label: str
    description: str | None = None
    type: Literal["text", "textarea", "int", "bool", "select", "multi_select", "date"]
    required: bool = False
    default: Any = None
    constraints: dict[str, Any] | None = None
    options: list[FieldOption] | None = None
    options_ref: str | None = None
    options_source: str | None = None


class FieldGroup(BaseModel):
    """Visual grouping of fields for form layout."""

    key: str
    label: str
    sort: int = 0
    fields: list[FieldDescriptor]
    visible_when: dict[str, Any] | None = None


class ResourceMetadata(BaseModel):
    """Top-level response for ``GET /api/v1/metadata/{resource}``."""

    resource: str
    groups: list[FieldGroup]
