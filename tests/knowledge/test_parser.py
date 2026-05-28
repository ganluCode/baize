"""Tests for MarkdownParser — hierarchical section tree from Markdown."""

import pytest

from baize.knowledge.ingestion.parser import MarkdownParser, ParsedSection


class TestParseEmptyAndPlainText:
    def test_empty_string_returns_empty_root(self):
        result = MarkdownParser.parse("Doc", "")
        assert result == ParsedSection(level=0, title="Doc", preface="", section_path=[], children=[])

    def test_whitespace_only_returns_empty_root(self):
        result = MarkdownParser.parse("Doc", "   \n\n  ")
        assert result.level == 0
        assert result.preface == ""
        assert result.children == []

    def test_plain_text_no_headers(self):
        content = "Just some text.\n\nAnother paragraph."
        result = MarkdownParser.parse("My Doc", content)
        assert result.level == 0
        assert result.title == "My Doc"
        assert result.preface == content
        assert result.children == []


class TestParseHierarchy:
    def test_h1_h2_h3_correct_parent_child(self):
        content = (
            "# Overview\nIntro\n\n"
            "## Install\nInstall info\n\n"
            "### Docker\nDocker details\n\n"
            "## Usage\nUsage info\n"
        )
        root = MarkdownParser.parse("Doc", content)

        assert root.level == 0
        assert len(root.children) == 1

        h1 = root.children[0]
        assert h1.level == 1
        assert h1.title == "Overview"
        assert "Intro" in h1.preface
        assert len(h1.children) == 2

        h2_install = h1.children[0]
        assert h2_install.level == 2
        assert h2_install.title == "Install"
        assert len(h2_install.children) == 1

        h3_docker = h2_install.children[0]
        assert h3_docker.level == 3
        assert h3_docker.title == "Docker"
        assert "Docker details" in h3_docker.preface
        assert h3_docker.children == []

        h2_usage = h1.children[1]
        assert h2_usage.level == 2
        assert h2_usage.title == "Usage"
        assert "Usage info" in h2_usage.preface

    def test_multiple_h1_sections(self):
        content = "# First\nAAA\n\n# Second\nBBB\n"
        root = MarkdownParser.parse("Doc", content)
        assert len(root.children) == 2
        assert root.children[0].title == "First"
        assert root.children[1].title == "Second"

    def test_preamble_before_first_heading(self):
        content = "Preamble text here.\n\n# Section\nContent\n"
        root = MarkdownParser.parse("Doc", content)
        assert "Preamble" in root.preface
        assert len(root.children) == 1
        assert root.children[0].title == "Section"


class TestSectionPath:
    def test_root_has_empty_path(self):
        root = MarkdownParser.parse("Doc", "# A\nText\n")
        assert root.section_path == []

    def test_leaf_path_includes_all_ancestors(self):
        content = "# 概述\nText\n\n## 安装\nText\n\n### Docker\nText\n"
        root = MarkdownParser.parse("Doc", content)
        h1 = root.children[0]
        h2 = h1.children[0]
        h3 = h2.children[0]

        assert h1.section_path == ["概述"]
        assert h2.section_path == ["概述", "安装"]
        assert h3.section_path == ["概述", "安装", "Docker"]

    def test_path_excludes_root_title(self):
        content = "# A\nText\n"
        root = MarkdownParser.parse("MyRoot", content)
        h1 = root.children[0]
        assert "MyRoot" not in h1.section_path


class TestH4MergesIntoPreface:
    def test_h4_content_in_parent_preface(self):
        content = "### Sub\nText\n\n#### Deep\nDeep text\n"
        root = MarkdownParser.parse("Doc", content)
        h3 = root.children[0]
        assert "#### Deep" in h3.preface
        assert "Deep text" in h3.preface

    def test_h4_under_h2_merges_into_h2(self):
        content = "## Section\nText\n\n#### Deep\nDeep text\n"
        root = MarkdownParser.parse("Doc", content)
        h2 = root.children[0]
        assert "Deep text" in h2.preface


class TestBoldTitleCleanup:
    def test_bold_stars_stripped(self):
        content = "## **Bold Title**\nContent\n"
        root = MarkdownParser.parse("Doc", content)
        child = root.children[0]
        assert child.title == "Bold Title"

    def test_italic_stripped(self):
        content = "## *Italic Title*\nContent\n"
        root = MarkdownParser.parse("Doc", content)
        child = root.children[0]
        assert child.title == "Italic Title"

    def test_mixed_decoration_stripped(self):
        content = "## **_Mixed_**\nContent\n"
        root = MarkdownParser.parse("Doc", content)
        child = root.children[0]
        assert child.title == "Mixed"
