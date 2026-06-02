"""Tests for answer_sheet_generator.config_exporter."""

import pytest

from answer_sheet_generator.config_exporter import export_sheet_layout
from answer_sheet_generator.schema import (
    AnswerSheetConfig,
    MetaConfig,
    PageConfig,
    SectionConfig,
    StudentIdConfig,
)


def _make_config(paper_size: str = "A4", pages: list = None) -> AnswerSheetConfig:
    """辅助函数：构造 AnswerSheetConfig。"""
    meta = MetaConfig(title="测试", paper_size=paper_size)
    student_id = StudentIdConfig(digit_count=10)
    if pages is None:
        pages = [
            PageConfig(
                sections=[
                    SectionConfig(
                        type="choice",
                        question_start=1,
                        question_count=10,
                        options=["A", "B", "C", "D"],
                        score=1.0,
                    ),
                    SectionConfig(
                        type="judge",
                        question_start=11,
                        question_count=5,
                        options=["T", "F"],
                        score=1.0,
                    ),
                ]
            )
        ]
    return AnswerSheetConfig(meta=meta, student_id=student_id, pages=pages)


class TestBasicStructure:
    def test_top_level_keys_exist(self):
        """验证导出结果包含所有顶层旧格式字段。"""
        cfg = _make_config()
        result = export_sheet_layout(cfg)

        assert "student_id" in result
        assert "choice" in result
        assert "judge" in result
        assert "layout" in result
        assert "_pages" in result

    def test_student_id_format(self):
        cfg = _make_config()
        result = export_sheet_layout(cfg)

        assert result["student_id"] == {"digit_count": 10}

    def test_choice_merged_format(self):
        cfg = _make_config()
        result = export_sheet_layout(cfg)

        choice = result["choice"]
        assert choice["rows"] == 2  # 10 题 / 5 列 = 2 行
        assert choice["cols"] == 5
        assert choice["question_start"] == 1
        assert choice["question_count"] == 10
        assert choice["options"] == ["A", "B", "C", "D"]

    def test_judge_merged_format(self):
        cfg = _make_config()
        result = export_sheet_layout(cfg)

        judge = result["judge"]
        assert judge["rows"] == 1  # 5 题 / 5 列 = 1 行
        assert judge["cols"] == 5
        assert judge["question_start"] == 11
        assert judge["question_count"] == 5
        assert judge["options"] == ["T", "F"]

    def test_no_choice_sections_omits_choice_key(self):
        cfg = _make_config(
            pages=[
                PageConfig(
                    sections=[
                        SectionConfig(
                            type="judge",
                            question_start=1,
                            question_count=5,
                            options=["T", "F"],
                            score=1.0,
                        ),
                    ]
                )
            ]
        )
        result = export_sheet_layout(cfg)

        assert "choice" not in result
        assert "judge" in result

    def test_no_judge_sections_omits_judge_key(self):
        cfg = _make_config(
            pages=[
                PageConfig(
                    sections=[
                        SectionConfig(
                            type="choice",
                            question_start=1,
                            question_count=5,
                            options=["A", "B", "C", "D"],
                            score=1.0,
                        ),
                    ]
                )
            ]
        )
        result = export_sheet_layout(cfg)

        assert "choice" in result
        assert "judge" not in result


class TestPagesField:
    def test_pages_has_correct_section_types(self):
        """验证 _pages 字段包含正确的 section 类型。"""
        cfg = _make_config()
        result = export_sheet_layout(cfg)

        pages = result["_pages"]
        assert len(pages) == 1
        assert pages[0]["page_number"] == 1

        sections = pages[0]["sections"]
        assert len(sections) == 2
        assert sections[0]["type"] == "choice"
        assert sections[1]["type"] == "judge"

    def test_pages_has_question_counts(self):
        cfg = _make_config()
        result = export_sheet_layout(cfg)

        sections = result["_pages"][0]["sections"]
        assert sections[0]["question_start"] == 1
        assert sections[0]["question_count"] == 10
        assert sections[1]["question_start"] == 11
        assert sections[1]["question_count"] == 5

    def test_pages_has_options(self):
        cfg = _make_config()
        result = export_sheet_layout(cfg)

        sections = result["_pages"][0]["sections"]
        assert sections[0]["options"] == ["A", "B", "C", "D"]
        assert sections[1]["options"] == ["T", "F"]

    def test_pages_with_essay_has_lines_per_question(self):
        cfg = _make_config(
            pages=[
                PageConfig(
                    sections=[
                        SectionConfig(
                            type="essay",
                            question_start=1,
                            question_count=2,
                            lines_per_question=3,
                            score=1.0,
                        ),
                    ]
                )
            ]
        )
        result = export_sheet_layout(cfg)

        sections = result["_pages"][0]["sections"]
        assert sections[0]["type"] == "essay"
        assert sections[0]["lines_per_question"] == 3

    def test_multi_page_config(self):
        cfg = _make_config(
            pages=[
                PageConfig(
                    sections=[
                        SectionConfig(
                            type="choice",
                            question_start=1,
                            question_count=10,
                            options=["A", "B", "C", "D"],
                            score=1.0,
                        ),
                    ]
                ),
                PageConfig(
                    sections=[
                        SectionConfig(
                            type="judge",
                            question_start=11,
                            question_count=5,
                            options=["T", "F"],
                            score=1.0,
                        ),
                    ]
                ),
            ]
        )
        result = export_sheet_layout(cfg)

        pages = result["_pages"]
        assert len(pages) == 2
        assert pages[0]["page_number"] == 1
        assert pages[1]["page_number"] == 2
        assert pages[0]["sections"][0]["type"] == "choice"
        assert pages[1]["sections"][0]["type"] == "judge"


class TestFallbackRanges:
    def test_layout_page1_fallback_has_numeric_ranges(self):
        """验证 layout.page1_fallback 包含 [0,1] 范围内的数值范围。"""
        cfg = _make_config()
        result = export_sheet_layout(cfg)

        layout = result["layout"]
        assert "page1_fallback" in layout

        fallback = layout["page1_fallback"]
        # 至少包含 choice 和 judge
        assert "choice" in fallback
        assert "judge" in fallback

        for key, value in fallback.items():
            assert isinstance(value, list)
            assert len(value) == 2
            y0, y1 = value
            assert isinstance(y0, float)
            assert isinstance(y1, float)
            assert 0.0 <= y0 <= 1.0
            assert 0.0 <= y1 <= 1.0
            assert y0 < y1

    def test_multi_page_fallback(self):
        """多页配置应产生多个 pageN_fallback。"""
        cfg = _make_config(
            pages=[
                PageConfig(
                    sections=[
                        SectionConfig(
                            type="choice",
                            question_start=1,
                            question_count=10,
                            options=["A", "B", "C", "D"],
                            score=1.0,
                        ),
                    ]
                ),
                PageConfig(
                    sections=[
                        SectionConfig(
                            type="judge",
                            question_start=11,
                            question_count=5,
                            options=["T", "F"],
                            score=1.0,
                        ),
                    ]
                ),
            ]
        )
        result = export_sheet_layout(cfg)

        layout = result["layout"]
        # 由于 paginate 可能将内容合并到一页或分到多页，
        # 我们只需验证存在的 fallback 键格式正确
        for key, fallback in layout.items():
            assert key.startswith("page") and key.endswith("_fallback")
            for section_key, value in fallback.items():
                y0, y1 = value
                assert 0.0 <= y0 <= 1.0
                assert 0.0 <= y1 <= 1.0
                assert y0 < y1

    def test_merged_choice_sections(self):
        """跨页的选择题应合并为一个 choice 条目。"""
        cfg = _make_config(
            pages=[
                PageConfig(
                    sections=[
                        SectionConfig(
                            type="choice",
                            question_start=1,
                            question_count=10,
                            options=["A", "B", "C", "D"],
                            score=1.0,
                        ),
                    ]
                ),
                PageConfig(
                    sections=[
                        SectionConfig(
                            type="choice",
                            question_start=11,
                            question_count=10,
                            options=["A", "B", "C", "D"],
                            score=1.0,
                        ),
                    ]
                ),
            ]
        )
        result = export_sheet_layout(cfg)

        choice = result["choice"]
        assert choice["question_start"] == 1
        assert choice["question_count"] == 20
        assert choice["rows"] == 4  # 20 / 5 = 4
        assert choice["cols"] == 5

    def test_a4_paper_cols(self):
        """A4 纸张应使用 5 列。"""
        cfg = _make_config(
            paper_size="A4",
            pages=[
                PageConfig(
                    sections=[
                        SectionConfig(
                            type="choice",
                            question_start=1,
                            question_count=10,
                            options=["A", "B", "C", "D"],
                            score=1.0,
                        ),
                    ]
                )
            ],
        )
        result = export_sheet_layout(cfg)

        choice = result["choice"]
        assert choice["cols"] == 5
        assert choice["rows"] == 2  # 10 / 5 = 2
