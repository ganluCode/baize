"""Tests for HierarchicalChunker — ChunkPlan generation from ParsedSection tree."""

import pytest

from baize.knowledge.ingestion.chunker import ChunkPlan, HierarchicalChunker
from baize.knowledge.ingestion.parser import MarkdownParser, ParsedSection


class TestParentChunkContent:
    """Each section produces a parent chunk with content = title + preface."""

    def test_section_parent_contains_title_and_preface(self):
        section = ParsedSection(
            level=1, title="Overview", preface="Intro text here.",
            section_path=["Overview"], children=[],
        )
        root = ParsedSection(level=0, title="Doc", preface="", section_path=[], children=[section])
        chunks = HierarchicalChunker().chunk(root)

        root_chunk = next(c for c in chunks if c.parent_local_id is None)
        section_parent = next(
            c for c in chunks if c.parent_local_id == root_chunk.local_id and c.level == 1
        )
        assert "Overview" in section_parent.content
        assert "Intro text here." in section_parent.content

    def test_root_produces_parent_chunk(self):
        root = ParsedSection(level=0, title="Doc", preface="Root preamble.", section_path=[], children=[])
        chunks = HierarchicalChunker().chunk(root)
        roots = [c for c in chunks if c.parent_local_id is None]
        assert len(roots) == 1
        assert "Doc" in roots[0].content
        assert "Root preamble." in roots[0].content


class TestChildChunksSplitByDoubleNewline:
    """Preface is split by double newline into child chunks."""

    def test_preface_paragraphs_become_children(self):
        section = ParsedSection(
            level=1, title="Section",
            preface="Paragraph one.\n\nParagraph two.\n\nParagraph three.",
            section_path=["Section"], children=[],
        )
        root = ParsedSection(level=0, title="Doc", preface="", section_path=[], children=[section])
        chunks = HierarchicalChunker().chunk(root)

        parent = next(c for c in chunks if c.level == 1 and "Section" in c.content)
        children = [c for c in chunks if c.parent_local_id == parent.local_id]
        assert len(children) == 3
        contents = {c.content for c in children}
        assert "Paragraph one." in contents
        assert "Paragraph two." in contents
        assert "Paragraph three." in contents

    def test_child_parent_local_id_points_to_section_parent(self):
        section = ParsedSection(
            level=1, title="S",
            preface="Para A.\n\nPara B.",
            section_path=["S"], children=[],
        )
        root = ParsedSection(level=0, title="Doc", preface="", section_path=[], children=[section])
        chunks = HierarchicalChunker().chunk(root)

        parent = next(c for c in chunks if c.level == 1)
        children = [c for c in chunks if c.parent_local_id == parent.local_id]
        assert len(children) == 2
        for child in children:
            assert child.parent_local_id == parent.local_id

    def test_no_children_when_preface_empty(self):
        section = ParsedSection(
            level=1, title="Empty", preface="",
            section_path=["Empty"], children=[],
        )
        root = ParsedSection(level=0, title="Doc", preface="", section_path=[], children=[section])
        chunks = HierarchicalChunker().chunk(root)

        parent = next(c for c in chunks if c.level == 1)
        children = [c for c in chunks if c.parent_local_id == parent.local_id]
        assert len(children) == 0


class TestLongParagraphSplit:
    """Paragraphs exceeding max_tokens are split via RecursiveCharacterTextSplitter."""

    def test_long_paragraph_produces_multiple_chunks(self):
        long_text = "word " * 500  # ~2500 chars
        section = ParsedSection(
            level=1, title="Section",
            preface=long_text.strip(),
            section_path=["Section"], children=[],
        )
        root = ParsedSection(level=0, title="Doc", preface="", section_path=[], children=[section])
        chunker = HierarchicalChunker(max_tokens=200)
        chunks = chunker.chunk(root)

        parent = next(c for c in chunks if c.level == 1 and "Section" in c.content)
        children = [c for c in chunks if c.parent_local_id == parent.local_id]
        assert len(children) > 1

    def test_short_paragraph_stays_single_chunk(self):
        section = ParsedSection(
            level=1, title="Section",
            preface="Short text.",
            section_path=["Section"], children=[],
        )
        root = ParsedSection(level=0, title="Doc", preface="", section_path=[], children=[section])
        chunker = HierarchicalChunker(max_tokens=1000)
        chunks = chunker.chunk(root)

        parent = next(c for c in chunks if c.level == 1)
        children = [c for c in chunks if c.parent_local_id == parent.local_id]
        assert len(children) == 1
        assert children[0].content == "Short text."


class TestDeduplication:
    """Same section_path + same content hash -> keep first only."""

    def test_duplicate_content_keeps_one(self):
        section = ParsedSection(
            level=1, title="Section",
            preface="Same para.\n\nSame para.",
            section_path=["Section"], children=[],
        )
        root = ParsedSection(level=0, title="Doc", preface="", section_path=[], children=[section])
        chunks = HierarchicalChunker().chunk(root)

        same_chunks = [c for c in chunks if c.content == "Same para."]
        assert len(same_chunks) == 1

    def test_different_section_paths_not_deduplicated(self):
        s1 = ParsedSection(level=1, title="A", preface="Shared.", section_path=["A"], children=[])
        s2 = ParsedSection(level=1, title="B", preface="Shared.", section_path=["B"], children=[])
        root = ParsedSection(level=0, title="Doc", preface="", section_path=[], children=[s1, s2])
        chunks = HierarchicalChunker().chunk(root)

        shared_chunks = [c for c in chunks if c.content == "Shared."]
        assert len(shared_chunks) == 2


class TestParentLocalIdIntegrity:
    """All child parent_local_id values resolve to existing local_id in the list."""

    def test_all_parent_ids_valid_in_deep_hierarchy(self):
        content = "# A\nText A para1.\n\nText A para2.\n\n## B\nText B.\n\n### C\nText C.\n"
        root = MarkdownParser.parse("Doc", content)
        chunks = HierarchicalChunker().chunk(root)

        all_ids = {c.local_id for c in chunks}
        for chunk in chunks:
            if chunk.parent_local_id is not None:
                assert chunk.parent_local_id in all_ids, (
                    f"parent_local_id '{chunk.parent_local_id}' not found"
                )

    def test_plain_text_children_point_to_root(self):
        root = ParsedSection(
            level=0, title="Doc",
            preface="Para one.\n\nPara two.",
            section_path=[], children=[],
        )
        chunks = HierarchicalChunker().chunk(root)

        root_chunk = next(c for c in chunks if c.parent_local_id is None)
        children = [c for c in chunks if c.parent_local_id == root_chunk.local_id]
        assert len(children) == 2

    def test_nested_sections_chain_correctly(self):
        h3 = ParsedSection(level=3, title="C", preface="Leaf.", section_path=["A", "B", "C"], children=[])
        h2 = ParsedSection(level=2, title="B", preface="Mid.", section_path=["A", "B"], children=[h3])
        h1 = ParsedSection(level=1, title="A", preface="Top.", section_path=["A"], children=[h2])
        root = ParsedSection(level=0, title="Doc", preface="", section_path=[], children=[h1])
        chunks = HierarchicalChunker().chunk(root)

        root_c = next(c for c in chunks if c.parent_local_id is None)
        h1_c = next(c for c in chunks if c.level == 1 and "A" in c.content and "Top." in c.content)
        h2_c = next(c for c in chunks if c.level == 2 and "B" in c.content and "Mid." in c.content)
        h3_c = next(c for c in chunks if c.level == 3 and "C" in c.content and "Leaf." in c.content)

        assert h1_c.parent_local_id == root_c.local_id
        assert h2_c.parent_local_id == h1_c.local_id
        assert h3_c.parent_local_id == h2_c.local_id
