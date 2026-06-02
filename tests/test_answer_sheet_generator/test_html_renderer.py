"""Tests for answer_sheet_generator.html_renderer."""

import pytest

from answer_sheet_generator.html_renderer import generate, render_html
from answer_sheet_generator.layout_engine import paginate
from answer_sheet_generator.schema import (
    AnswerSheetConfig,
    MetaConfig,
    PageConfig,
    SectionConfig,
    StudentIdConfig,
)


def _make_config(title: str, sections: list, paper_size: str = "A4") -> AnswerSheetConfig:
    """辅助函数：构造 AnswerSheetConfig。"""
    meta = MetaConfig(title=title, paper_size=paper_size)
    student_id = StudentIdConfig(digit_count=10)
    pages = [PageConfig(sections=sections)]
    return AnswerSheetConfig(meta=meta, student_id=student_id, pages=pages)


class TestRenderHtml:
    def test_contains_title(self) -> None:
        """HTML 输出应包含 cfg.meta.title。"""
        sections = [
            SectionConfig(type="choice", question_start=1, question_count=5, options=["A", "B", "C", "D"], score=1.0),
        ]
        cfg = _make_config("期中考试答题卡", sections)
        pages = paginate(cfg)
        html = render_html(cfg, pages)
        assert "期中考试答题卡" in html
        assert "<title>期中考试答题卡</title>" in html

    def test_contains_questions(self) -> None:
        """HTML 输出应包含题号。"""
        sections = [
            SectionConfig(type="choice", question_start=1, question_count=5, options=["A", "B", "C", "D"], score=1.0),
            SectionConfig(type="judge", question_start=6, question_count=3, options=["T", "F"], score=1.0),
        ]
        cfg = _make_config("测试答题卡", sections)
        pages = paginate(cfg)
        html = render_html(cfg, pages)
        # 检查选择题号
        assert "1." in html
        assert "5." in html
        # 检查判断题号
        assert "6." in html
        assert "8." in html

    def test_multi_page(self) -> None:
        """多页情况下应生成正确的页码标记。"""
        # 100 道选择题在 A4 上会跨多页
        sections = [
            SectionConfig(
                type="choice", question_start=1, question_count=100,
                options=["A", "B", "C", "D"], score=1.0,
            ),
        ]
        cfg = _make_config("大题量测试", sections)
        pages = paginate(cfg)
        assert len(pages) >= 2
        html = render_html(cfg, pages)
        # 检查页码标记
        assert "第 1 页 / 共" in html
        assert f"共 {len(pages)} 页" in html
        # 最后一页页码也应存在
        assert f"第 {len(pages)} 页 / 共 {len(pages)} 页" in html

    def test_contains_css_rules(self) -> None:
        """HTML 应包含关键 CSS 规则。"""
        sections = [
            SectionConfig(type="choice", question_start=1, question_count=5, options=["A", "B", "C", "D"], score=1.0),
        ]
        cfg = _make_config("CSS 测试", sections)
        pages = paginate(cfg)
        html = render_html(cfg, pages)
        assert "@page" in html
        assert "size: 210mm 297mm" in html
        assert "@media screen" in html
        assert "@media print" in html
        assert "page-break-after: always" in html
        assert "position: relative" in html
        assert "page-title" in html
        assert "exam-info" in html
        assert "student-id-section" in html
        assert "choice-section" in html
        assert "judge-section" in html
        assert "essay-section" in html
        assert "choice-grid" in html
        assert "judge-grid" in html
        assert "essay-line" in html

    def test_exam_info_fields(self) -> None:
        """HTML 应包含科目、日期、姓名填写行。"""
        sections = [
            SectionConfig(type="choice", question_start=1, question_count=5, options=["A", "B", "C", "D"], score=1.0),
        ]
        cfg = _make_config("信息栏测试", sections)
        pages = paginate(cfg)
        html = render_html(cfg, pages)
        assert "科目：" in html
        assert "日期：" in html
        assert "姓名：" in html


class TestGenerate:
    def test_generate_returns_html(self) -> None:
        """generate() 应返回完整 HTML 字符串。"""
        sections = [
            SectionConfig(type="choice", question_start=1, question_count=10, options=["A", "B", "C", "D"], score=1.0),
        ]
        cfg = _make_config("生成测试", sections)
        html = generate(cfg)
        assert html.startswith("<!DOCTYPE html>")
        assert "生成测试" in html
        assert "1." in html
        assert "10." in html
