"""Shared data types for converters."""

from dataclasses import dataclass, field


@dataclass
class ConvertedDocument:
    """Result of converting a document to Markdown."""

    markdown: str
    warnings: list[str] = field(default_factory=list)
