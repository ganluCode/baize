"""Hierarchical chunker that splits ParsedSection trees into ChunkPlan lists."""

import hashlib
import itertools
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from baize.knowledge.ingestion.parser import ParsedSection


@dataclass
class ChunkPlan:
    """A planned chunk with local parent-child linking, before DB insertion."""

    local_id: str
    parent_local_id: str | None
    level: int
    section_path: list[str]
    content: str


class HierarchicalChunker:
    """Recursively traverse a ParsedSection tree to produce ChunkPlan list.

    Each section yields one parent chunk (title + preface) and zero or more
    child chunks (paragraphs split from the preface). Long paragraphs are
    further split via RecursiveCharacterTextSplitter. Duplicate content within
    the same section_path is deduplicated by SHA-256 hash.
    """

    def __init__(self, max_tokens: int = 1000) -> None:
        self._max_tokens = max_tokens

    def chunk(self, section: ParsedSection) -> list[ChunkPlan]:
        """Convert a ParsedSection tree into a flat list of ChunkPlan."""
        plans: list[ChunkPlan] = []
        seen: set[tuple[tuple[str, ...], str]] = set()
        counter = itertools.count()
        self._walk(section, parent_local_id=None, plans=plans, seen=seen, counter=counter)
        return plans

    def _walk(
        self,
        section: ParsedSection,
        parent_local_id: str | None,
        plans: list[ChunkPlan],
        seen: set[tuple[tuple[str, ...], str]],
        counter: itertools.count,
    ) -> None:
        if section.preface:
            content = f"{section.title}\n\n{section.preface}" if section.title else section.preface
        else:
            content = section.title

        local_id = f"chunk_{next(counter)}"
        plans.append(ChunkPlan(
            local_id=local_id,
            parent_local_id=parent_local_id,
            level=section.level,
            section_path=list(section.section_path),
            content=content,
        ))

        if section.preface:
            self._split_preface(
                section.preface, local_id, section.level,
                section.section_path, plans, seen, counter,
            )

        for child in section.children:
            self._walk(child, parent_local_id=local_id, plans=plans, seen=seen, counter=counter)

    def _split_preface(
        self,
        preface: str,
        parent_id: str,
        level: int,
        section_path: list[str],
        plans: list[ChunkPlan],
        seen: set[tuple[tuple[str, ...], str]],
        counter: itertools.count,
    ) -> None:
        paragraphs = [p.strip() for p in preface.split("\n\n") if p.strip()]
        path_key = tuple(section_path)

        for para in paragraphs:
            if len(para) > self._max_tokens:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self._max_tokens,
                    chunk_overlap=0,
                )
                sub_texts = splitter.split_text(para)
            else:
                sub_texts = [para]

            for text in sub_texts:
                content_hash = hashlib.sha256(text.encode()).hexdigest()
                key = (path_key, content_hash)
                if key in seen:
                    continue
                seen.add(key)
                plans.append(ChunkPlan(
                    local_id=f"chunk_{next(counter)}",
                    parent_local_id=parent_id,
                    level=level,
                    section_path=list(section_path),
                    content=text,
                ))
