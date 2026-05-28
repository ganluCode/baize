"""Markdown parser that builds a hierarchical section tree."""

import re
from dataclasses import dataclass, field

from langchain_text_splitters import MarkdownHeaderTextSplitter

_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_HEADER_LINE_RE = re.compile(r"^#{1,6}\s+")
_MD_DECORATION_RE = re.compile(r"[*_~`]")


@dataclass
class ParsedSection:
    """A node in the hierarchical section tree parsed from Markdown."""

    level: int
    title: str
    preface: str
    section_path: list[str]
    children: list["ParsedSection"] = field(default_factory=list)


def _clean_title(raw: str) -> str:
    """Strip markdown inline decorations (bold, italic, etc.) from a title."""
    return _MD_DECORATION_RE.sub("", raw).strip()


def _chunk_info(metadata: dict[str, str]) -> tuple[int, str]:
    """Extract the deepest heading level and its title from splitter metadata."""
    level = 0
    title = ""
    for key, value in metadata.items():
        if key.startswith("h") and key[1:].isdigit():
            lvl = int(key[1:])
            if lvl > level:
                level = lvl
                title = value
    return level, title


def _extract_preface(content: str) -> str:
    """Remove the leading header line from chunk content, returning the body."""
    lines = content.split("\n")
    if lines and _HEADER_LINE_RE.match(lines[0]):
        return "\n".join(lines[1:]).strip()
    return content.strip()


class MarkdownParser:
    """Parse Markdown content into a hierarchical ParsedSection tree.

    Splits on h1/h2/h3 headers. h4+ headers are kept in the preface text
    of their parent section.
    """

    @staticmethod
    def parse(title: str, content: str) -> ParsedSection:
        """Parse Markdown content into a hierarchical section tree.

        Args:
            title: Document title for the root node.
            content: Markdown content to parse.

        Returns:
            Root ParsedSection (level=0) with children forming the heading tree.
        """
        root = ParsedSection(level=0, title=title, preface="", section_path=[], children=[])

        if not content or not content.strip():
            return root

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        chunks = splitter.split_text(content)

        if not chunks:
            return root

        has_headers = any(c.metadata for c in chunks)
        if not has_headers:
            root.preface = content.strip()
            return root

        stack: list[tuple[int, ParsedSection]] = [(0, root)]

        for chunk in chunks:
            if not chunk.metadata:
                root.preface = chunk.page_content.strip()
                continue

            level, raw_title = _chunk_info(chunk.metadata)
            clean = _clean_title(raw_title)
            preface = _extract_preface(chunk.page_content)

            while stack and stack[-1][0] >= level:
                stack.pop()

            section_path = [node.title for lvl, node in stack if lvl > 0]
            section_path.append(clean)

            node = ParsedSection(
                level=level,
                title=clean,
                preface=preface,
                section_path=section_path,
                children=[],
            )

            stack[-1][1].children.append(node)
            stack.append((level, node))

        return root
