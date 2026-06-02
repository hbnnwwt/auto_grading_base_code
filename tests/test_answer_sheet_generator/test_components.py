"""Tests for answer_sheet_generator.components."""

import math
from typing import Any

import pytest

from answer_sheet_generator.components import (
    ChoiceComponent,
    Component,
    EssayComponent,
    JudgeComponent,
    StudentIdComponent,
)
from answer_sheet_generator.schema import SectionConfig, StudentIdConfig


class DummyComponent(Component):
    """Concrete subclass for testing abstract base behaviour."""

    def estimate_height(self, paper_size: str = "A4") -> float:
        return 0.0

    def render(self, page_num: int, y_offset_mm: float) -> str:
        return ""

    def split(
        self, available_height: float, paper_size: str = "A4"
    ) -> Any:
        return None


# ---------------------------------------------------------------------------
# StudentIdComponent
# ---------------------------------------------------------------------------


class TestStudentIdComponent:
    def test_estimate_height_default_digits(self) -> None:
        cfg = StudentIdConfig(digit_count=10)
        comp = StudentIdComponent(cfg)
        # 10 + 8 + 8 + 7*10 + 8 = 104
        assert comp.estimate_height() == 104.0

    def test_estimate_height_custom_digits(self) -> None:
        cfg = StudentIdConfig(digit_count=12)
        comp = StudentIdComponent(cfg)
        # 10 + 8 + 8 + 7*12 + 8 = 118
        assert comp.estimate_height() == 118.0

    def test_split_returns_none(self) -> None:
        cfg = StudentIdConfig(digit_count=10)
        comp = StudentIdComponent(cfg)
        assert comp.split(200.0) is None

    def test_render_contains_title_and_grid(self) -> None:
        cfg = StudentIdConfig(digit_count=10)
        comp = StudentIdComponent(cfg)
        html = comp.render(page_num=1, y_offset_mm=0.0)
        assert "准考证号" in html
        assert "sid-grid" in html


# ---------------------------------------------------------------------------
# ChoiceComponent
# ---------------------------------------------------------------------------


class TestChoiceComponent:
    def _make(self, question_start: int = 1, question_count: int = 20, options: list[str] | None = None, score: float | None = None, scores: list[float] | None = None) -> ChoiceComponent:
        if options is None:
            options = ["A", "B", "C", "D"]
        if score is None and scores is None:
            score = 1.0
        cfg = SectionConfig(
            type="choice",
            question_start=question_start,
            question_count=question_count,
            options=options,
            score=score,
            scores=scores,
        )
        return ChoiceComponent(cfg)

    def test_estimate_height_a4_20_questions(self) -> None:
        comp = self._make(question_count=20)
        cols = 5
        rows = math.ceil(20 / cols)
        expected = 18.0 + rows * 12.0 + 16.0
        assert comp.estimate_height("A4") == expected  # 82.0

    def test_estimate_height_a4_20_questions(self) -> None:
        comp = self._make(question_count=20)
        cols = 5
        rows = math.ceil(20 / cols)
        expected = 18.0 + rows * 12.0 + 16.0
        assert comp.estimate_height("A4") == expected  # 82.0

    def test_estimate_height_b5(self) -> None:
        comp = self._make(question_count=10)
        # B5 uses same cols as A4
        cols = 5
        rows = math.ceil(10 / cols)
        expected = 18.0 + rows * 12.0 + 16.0
        assert comp.estimate_height("B5") == expected

    def test_render_contains_questions(self) -> None:
        comp = self._make(question_start=1, question_count=5)
        html = comp.render(page_num=1, y_offset_mm=0.0)
        assert "1." in html
        assert "5." in html
        assert "A" in html
        assert "choice-section" in html

    def test_split_fits_no_split(self) -> None:
        comp = self._make(question_count=20)
        # height for A4 = 82, available = 100 -> no split
        assert comp.split(100.0, "A4") is None

    def test_split_produces_correct_parts(self) -> None:
        comp = self._make(question_count=20)
        # A4 height = 82; available = 60 -> must split
        result = comp.split(60.0, "A4")
        assert result is not None
        first, second = result
        assert isinstance(first, ChoiceComponent)
        assert isinstance(second, ChoiceComponent)
        # A4: cols=5, usable=60-34=26, max_rows=2, max_questions=10
        assert first.question_count == 10
        assert second.question_count == 10
        assert first.question_start == 1
        assert second.question_start == 11

    def test_split_propagates_scores(self) -> None:
        scores = [1.0] * 20
        comp = self._make(question_count=20, scores=scores)
        result = comp.split(60.0, "A4")
        assert result is not None
        first, second = result
        assert first.scores == [1.0] * 10
        assert second.scores == [1.0] * 10
        assert first.score is None
        assert second.score is None

    def test_split_propagates_single_score(self) -> None:
        comp = self._make(question_count=20, score=2.5)
        result = comp.split(60.0, "A4")
        assert result is not None
        first, second = result
        assert first.score == 2.5
        assert second.score == 2.5
        assert first.scores is None
        assert second.scores is None

    def test_split_too_small_returns_none(self) -> None:
        comp = self._make(question_count=20)
        # available < 34 + 12 = 46, can't fit even one row
        assert comp.split(40.0, "A4") is None


# ---------------------------------------------------------------------------
# JudgeComponent
# ---------------------------------------------------------------------------


class TestJudgeComponent:
    def _make(self, question_start: int = 1, question_count: int = 10, score: float | None = None, scores: list[float] | None = None) -> JudgeComponent:
        if score is None and scores is None:
            score = 1.0
        cfg = SectionConfig(
            type="judge",
            question_start=question_start,
            question_count=question_count,
            options=["T", "F"],
            score=score,
            scores=scores,
        )
        return JudgeComponent(cfg)

    def test_estimate_height_same_as_choice(self) -> None:
        comp = self._make(question_count=10)
        cols = 5
        rows = math.ceil(10 / cols)
        expected = 18.0 + rows * 12.0 + 16.0
        assert comp.estimate_height("A4") == expected

    def test_render_contains_tf(self) -> None:
        comp = self._make(question_start=1, question_count=3)
        html = comp.render(page_num=1, y_offset_mm=0.0)
        assert "T" in html
        assert "F" in html
        assert "judge-section" in html

    def test_split_propagates_scores(self) -> None:
        scores = [2.0] * 10
        comp = self._make(question_count=10, scores=scores)
        result = comp.split(50.0, "A4")
        assert result is not None
        first, second = result
        assert first.scores == [2.0] * 5
        assert second.scores == [2.0] * 5


# ---------------------------------------------------------------------------
# EssayComponent
# ---------------------------------------------------------------------------


class TestEssayComponent:
    def _make(self, question_start: int = 1, question_count: int = 3, lines_per_question: int = 5, score: float | None = None, scores: list[float] | None = None) -> EssayComponent:
        if score is None and scores is None:
            score = 1.0
        cfg = SectionConfig(
            type="essay",
            question_start=question_start,
            question_count=question_count,
            lines_per_question=lines_per_question,
            score=score,
            scores=scores,
        )
        return EssayComponent(cfg)

    def test_estimate_height(self) -> None:
        comp = self._make(question_count=3, lines_per_question=5)
        # 18 + 3*(6 + 5*8) + 16 = 18 + 3*46 + 16 = 18 + 138 + 16 = 172
        expected = 18.0 + 3 * (6.0 + 5 * 8.0) + 16.0
        assert comp.estimate_height() == expected

    def test_render_contains_lines(self) -> None:
        comp = self._make(question_count=2, lines_per_question=3)
        html = comp.render(page_num=1, y_offset_mm=0.0)
        assert "essay-line" in html
        assert "1." in html
        assert "2." in html

    def test_split_by_whole_question(self) -> None:
        comp = self._make(question_count=5, lines_per_question=4)
        # height = 18 + 5*(6+32) + 16 = 18 + 190 + 16 = 224
        # available = 150 -> usable = 116, q_height = 38, max_questions = 3
        result = comp.split(150.0, "A4")
        assert result is not None
        first, second = result
        assert first.question_count == 3
        assert second.question_count == 2
        assert first.question_start == 1
        assert second.question_start == 4

    def test_split_propagates_scores(self) -> None:
        scores = [5.0, 4.0, 3.0, 2.0, 1.0]
        comp = self._make(question_count=5, lines_per_question=4, scores=scores)
        result = comp.split(150.0, "A4")
        assert result is not None
        first, second = result
        assert first.scores == [5.0, 4.0, 3.0]
        assert second.scores == [2.0, 1.0]

    def test_split_too_small_returns_none(self) -> None:
        comp = self._make(question_count=5, lines_per_question=4)
        # q_height = 38, available = 40 < 34 + 38 = 72
        assert comp.split(40.0, "A4") is None

    def test_split_fits_no_split(self) -> None:
        comp = self._make(question_count=2, lines_per_question=2)
        # height = 18 + 2*(6+16) + 16 = 18 + 44 + 16 = 78
        assert comp.split(100.0, "A4") is None


# ---------------------------------------------------------------------------
# Component base class
# ---------------------------------------------------------------------------


class TestComponentBase:
    def test_cols_for_paper(self) -> None:
        comp = DummyComponent(config=None)
        assert comp._cols_for_paper("A4") == 5
        assert comp._cols_for_paper("B5") == 5
