"""Tests for answer_sheet_generator.layout_engine."""

import pytest

from answer_sheet_generator.components import ChoiceComponent, EssayComponent, JudgeComponent
from answer_sheet_generator.layout_engine import (
    HEADER_HEIGHT,
    PAGE_SIZES,
    LayoutError,
    Page,
    _create_component,
    paginate,
)
from answer_sheet_generator.schema import AnswerSheetConfig, MetaConfig, PageConfig, SectionConfig, StudentIdConfig


# ---------------------------------------------------------------------------
# Page dataclass
# ---------------------------------------------------------------------------


class TestPage:
    def test_add_component(self) -> None:
        page = Page(page_number=1)
        cfg = SectionConfig(type="choice", question_start=1, question_count=5, options=["A", "B", "C", "D"], score=1.0)
        comp = ChoiceComponent(cfg)
        page.add_component(comp, 10.0)
        assert len(page.components) == 1
        assert page.components[0][1] == 10.0

    def test_total_height(self) -> None:
        page = Page(page_number=1)
        cfg = SectionConfig(type="choice", question_start=1, question_count=5, options=["A", "B", "C", "D"], score=1.0)
        comp = ChoiceComponent(cfg)
        page.add_component(comp, 0.0)
        expected = comp.estimate_height("A4")
        assert page.total_height("A4") == expected


# ---------------------------------------------------------------------------
# _create_component factory
# ---------------------------------------------------------------------------


class TestCreateComponent:
    def test_choice(self) -> None:
        cfg = SectionConfig(type="choice", question_start=1, question_count=5, options=["A", "B", "C", "D"], score=1.0)
        comp = _create_component(cfg)
        assert isinstance(comp, ChoiceComponent)

    def test_judge(self) -> None:
        cfg = SectionConfig(type="judge", question_start=1, question_count=5, options=["T", "F"], score=1.0)
        comp = _create_component(cfg)
        assert isinstance(comp, JudgeComponent)

    def test_essay(self) -> None:
        cfg = SectionConfig(type="essay", question_start=1, question_count=2, lines_per_question=3, score=1.0)
        comp = _create_component(cfg)
        assert isinstance(comp, EssayComponent)

    def test_unknown_type_raises(self) -> None:
        # SectionConfig validates type, so we bypass it by creating a valid config
        # then monkey-patching the type to trigger _create_component's error path
        cfg = SectionConfig(type="choice", question_start=1, question_count=1, options=["A", "B"], score=1.0)
        cfg.type = "unknown"  # type: ignore[misc]
        with pytest.raises(ValueError):
            _create_component(cfg)


# ---------------------------------------------------------------------------
# paginate
# ---------------------------------------------------------------------------


def _make_config(paper_size: str, sections: list) -> AnswerSheetConfig:
    """辅助函数：用给定 section 列表构造 AnswerSheetConfig。"""
    meta = MetaConfig(title="测试", paper_size=paper_size)
    student_id = StudentIdConfig(digit_count=10)
    pages = [PageConfig(sections=sections)]
    return AnswerSheetConfig(meta=meta, student_id=student_id, pages=pages)


class TestPaginateSinglePage:
    def test_single_page(self) -> None:
        """少量题目应全部放在一页内。"""
        sections = [
            SectionConfig(type="choice", question_start=1, question_count=10, options=["A", "B", "C", "D"], score=1.0),
        ]
        cfg = _make_config("A4", sections)
        pages = paginate(cfg)
        assert len(pages) == 1
        # 第一页自动包含 StudentIdComponent + 1 个 section
        assert len(pages[0].components) == 2


class TestPaginateMultiplePages:
    def test_multiple_pages(self) -> None:
        """大量题目应跨多页，且最后一题号正确。"""
        # A4 usable_height=277, page_net_height=252
        # StudentId height = 104, 剩余 148
        # 每个 choice 题 5 列，每行 12mm，header+padding=34
        # 100 题 = 20 行，高度 = 34 + 20*12 = 274，远超一页
        sections = [
            SectionConfig(
                type="choice", question_start=1, question_count=100,
                options=["A", "B", "C", "D"], score=1.0,
            ),
        ]
        cfg = _make_config("A4", sections)
        pages = paginate(cfg)
        assert len(pages) >= 2

        # 验证最后一页的最后一个组件包含第 100 题
        last_page = pages[-1]
        last_comp, _ = last_page.components[-1]
        assert isinstance(last_comp, ChoiceComponent)
        assert last_comp.question_start + last_comp.question_count - 1 == 100


class TestPaginateMixedTypes:
    def test_mixed_types_on_same_page(self) -> None:
        """选择题 + 判断题应能放在同一页。"""
        sections = [
            SectionConfig(type="choice", question_start=1, question_count=10, options=["A", "B", "C", "D"], score=1.0),
            SectionConfig(type="judge", question_start=11, question_count=10, options=["T", "F"], score=1.0),
        ]
        cfg = _make_config("A4", sections)
        pages = paginate(cfg)
        # 10 道 choice: 2 行 -> 34 + 24 = 58
        # 10 道 judge:  2 行 -> 34 + 24 = 58
        # StudentId: 104
        # 总计: 104 + 58 + 58 = 220 < 252 (A4 net height)
        assert len(pages) == 1
        from answer_sheet_generator.components import StudentIdComponent
        assert isinstance(pages[0].components[0][0], StudentIdComponent)
        assert isinstance(pages[0].components[1][0], ChoiceComponent)
        assert isinstance(pages[0].components[2][0], JudgeComponent)

    def test_mixed_types_on_same_page_instance_check(self) -> None:
        sections = [
            SectionConfig(type="choice", question_start=1, question_count=10, options=["A", "B", "C", "D"], score=1.0),
            SectionConfig(type="judge", question_start=11, question_count=10, options=["T", "F"], score=1.0),
        ]
        cfg = _make_config("A4", sections)
        pages = paginate(cfg)
        assert len(pages) == 1
        from answer_sheet_generator.components import StudentIdComponent
        assert isinstance(pages[0].components[0][0], StudentIdComponent)
        assert isinstance(pages[0].components[1][0], ChoiceComponent)
        assert isinstance(pages[0].components[2][0], JudgeComponent)


class TestPaginateOverflow:
    def test_essay_too_large_for_b5(self) -> None:
        """B5 纸张下 30 行简答题应触发 LayoutError。"""
        # B5: usable_height=230, net=205
        # StudentId=104, 剩余 101
        # essay 30 行 * 1 题: 18 + 1*(6+30*8) + 16 = 18 + 246 + 16 = 280 > 101
        # 且单题无法拆分（split 按题拆分，若 usable < q_height 则返回 None）
        sections = [
            SectionConfig(type="essay", question_start=1, question_count=1, lines_per_question=30, score=1.0),
        ]
        cfg = _make_config("B5", sections)
        with pytest.raises(LayoutError) as exc_info:
            paginate(cfg)
        assert "essay" in str(exc_info.value)


class TestPaginateContinuity:
    def test_question_continuity(self) -> None:
        """分页后所有题号应从 1 开始连续无间断。"""
        sections = [
            SectionConfig(type="choice", question_start=1, question_count=50, options=["A", "B", "C", "D"], score=1.0),
            SectionConfig(type="judge", question_start=51, question_count=20, options=["T", "F"], score=1.0),
        ]
        cfg = _make_config("A4", sections)
        pages = paginate(cfg)

        # 收集所有组件中的题号范围
        all_numbers = []
        for page in pages:
            for comp, _ in page.components:
                if hasattr(comp, "question_start") and hasattr(comp, "question_count"):
                    start = comp.question_start
                    count = comp.question_count
                    all_numbers.extend(range(start, start + count))

        all_numbers.sort()
        expected = list(range(1, len(all_numbers) + 1))
        assert all_numbers == expected


class TestPaginatePageNumbers:
    def test_page_numbers_are_sequential(self) -> None:
        """分页结果的页码应从 1 开始连续递增。"""
        sections = [
            SectionConfig(type="choice", question_start=1, question_count=100, options=["A", "B", "C", "D"], score=1.0),
        ]
        cfg = _make_config("A4", sections)
        pages = paginate(cfg)
        page_numbers = [p.page_number for p in pages]
        assert page_numbers == list(range(1, len(pages) + 1))
