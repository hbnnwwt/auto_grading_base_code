"""答题卡渲染组件。"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from answer_sheet_generator.schema import SectionConfig, StudentIdConfig


class Component(ABC):
    """答题卡组件抽象基类。"""

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def estimate_height(self, paper_size: str = "A4") -> float:
        """估算组件在指定纸张上占用的毫米高度。"""

    @abstractmethod
    def render(self, page_num: int, y_offset_mm: float) -> str:
        """渲染为 HTML 字符串。"""

    @abstractmethod
    def split(
        self, available_height: float, paper_size: str = "A4"
    ) -> Optional[Tuple["Component", "Component"]]:
        """尝试在 available_height 处拆分为两个组件。

        返回 (first_part, remaining_part) 或 None（不可拆分）。
        """

    def _cols_for_paper(self, paper_size: str) -> int:
        """根据纸张尺寸返回每行可容纳的题数。"""
        return 5  # A4 / B5 固定 5 列


class StudentIdComponent(Component):
    """学号填涂区域组件。"""

    def __init__(self, config: StudentIdConfig) -> None:
        super().__init__(config)
        self.digit_count = config.digit_count

    def estimate_height(self, paper_size: str = "A4") -> float:
        # title 10 + instruction 8 + write row 8 + digit rows 7*10 + padding 8
        return 10.0 + 8.0 + 8.0 + 7.0 * self.digit_count + 8.0

    def render(self, page_num: int, y_offset_mm: float) -> str:
        digits = list(range(10))  # 0-9
        cells = ""
        for d in digits:
            cells += f'<div class="sid-digit-header">{d}</div>\n'
            for _ in range(self.digit_count):
                cells += '<div class="sid-cell"></div>\n'

        html = f'''<div class="student-id-section" style="top:{y_offset_mm}mm">
  <div class="sid-title">准考证号</div>
  <div class="sid-instruction">请用 2B 铅笔将对应数字涂黑</div>
  <div class="sid-write-row">
    <span>学号填写：</span>
    {' '.join(['<span class="sid-write-box"></span>' for _ in range(self.digit_count)])}
  </div>
  <div class="sid-grid" style="grid-template-columns: repeat({self.digit_count}, 1fr)">
    {cells}
  </div>
</div>
'''
        return html

    def split(
        self, available_height: float, paper_size: str = "A4"
    ) -> Optional[Tuple["Component", "Component"]]:
        return None


class ChoiceComponent(Component):
    """选择题区域组件。"""

    def __init__(self, config: SectionConfig) -> None:
        super().__init__(config)
        self.question_start = config.question_start
        self.question_count = config.question_count
        self.options = config.options or []
        self.score = config.score
        self.scores = config.scores

    def estimate_height(self, paper_size: str = "A4") -> float:
        cols = self._cols_for_paper(paper_size)
        rows = math.ceil(self.question_count / cols)
        # header 18 + rows * 12 + padding 16
        return 18.0 + rows * 12.0 + 16.0

    def render(self, page_num: int, y_offset_mm: float) -> str:
        cols = self._cols_for_paper("A4")  # render 使用 A4 列数
        questions_html = ""
        for i in range(self.question_count):
            q_num = self.question_start + i
            options_html = ""
            for opt in self.options:
                options_html += f'<span class="opt">{opt}</span>'
            questions_html += (
                f'<div class="q-item">'
                f'<span class="q-num">{q_num}.</span>{options_html}'
                f'</div>\n'
            )

        title = self.config.title or f"选择题（第 {self.question_start}~{self.question_start + self.question_count - 1} 题）"
        html = f'''<div class="choice-section" style="top:{y_offset_mm}mm">
  <div class="sec-title">{title}</div>
  <div class="choice-grid" style="grid-template-columns: repeat({cols}, 1fr)">
    {questions_html}
  </div>
</div>
'''
        return html

    def split(
        self, available_height: float, paper_size: str = "A4"
    ) -> Optional[Tuple["Component", "Component"]]:
        needed = self.estimate_height(paper_size)
        if needed <= available_height:
            return None

        cols = self._cols_for_paper(paper_size)
        # header 18 + padding 16 = 34, each row 12
        usable = available_height - 34.0
        if usable < 12.0:
            # can't fit even one row
            max_rows = 0
        else:
            max_rows = int(usable // 12.0)
        max_questions = max_rows * cols

        if max_questions < 1:
            return None
        if max_questions >= self.question_count:
            return None

        first_count = max_questions
        second_count = self.question_count - first_count

        first_scores = None
        second_scores = None
        if self.scores is not None:
            first_scores = self.scores[:first_count]
            second_scores = self.scores[first_count:]

        first_cfg = SectionConfig(
            type="choice",
            question_start=self.question_start,
            question_count=first_count,
            options=list(self.options),
            score=self.score,
            scores=first_scores,
        )
        second_cfg = SectionConfig(
            type="choice",
            question_start=self.question_start + first_count,
            question_count=second_count,
            options=list(self.options),
            score=self.score,
            scores=second_scores,
        )
        return (ChoiceComponent(first_cfg), ChoiceComponent(second_cfg))


class JudgeComponent(Component):
    """判断题区域组件。"""

    def __init__(self, config: SectionConfig) -> None:
        super().__init__(config)
        self.question_start = config.question_start
        self.question_count = config.question_count
        self.options = config.options or ["T", "F"]
        self.score = config.score
        self.scores = config.scores

    def estimate_height(self, paper_size: str = "A4") -> float:
        cols = self._cols_for_paper(paper_size)
        rows = math.ceil(self.question_count / cols)
        return 18.0 + rows * 12.0 + 16.0

    def render(self, page_num: int, y_offset_mm: float) -> str:
        cols = self._cols_for_paper("A4")
        questions_html = ""
        for i in range(self.question_count):
            q_num = self.question_start + i
            options_html = ""
            for opt in self.options:
                options_html += f'<span class="opt">{opt}</span>'
            questions_html += (
                f'<div class="q-item">'
                f'<span class="q-num">{q_num}.</span>{options_html}'
                f'</div>\n'
            )

        title = self.config.title or f"判断题（第 {self.question_start}~{self.question_start + self.question_count - 1} 题）"
        html = f'''<div class="judge-section" style="top:{y_offset_mm}mm">
  <div class="sec-title">{title}</div>
  <div class="judge-grid" style="grid-template-columns: repeat({cols}, 1fr)">
    {questions_html}
  </div>
</div>
'''
        return html

    def split(
        self, available_height: float, paper_size: str = "A4"
    ) -> Optional[Tuple["Component", "Component"]]:
        needed = self.estimate_height(paper_size)
        if needed <= available_height:
            return None

        cols = self._cols_for_paper(paper_size)
        usable = available_height - 34.0
        if usable < 12.0:
            max_rows = 0
        else:
            max_rows = int(usable // 12.0)
        max_questions = max_rows * cols

        if max_questions < 1:
            return None
        if max_questions >= self.question_count:
            return None

        first_count = max_questions
        second_count = self.question_count - first_count

        first_scores = None
        second_scores = None
        if self.scores is not None:
            first_scores = self.scores[:first_count]
            second_scores = self.scores[first_count:]

        first_cfg = SectionConfig(
            type="judge",
            question_start=self.question_start,
            question_count=first_count,
            options=list(self.options),
            score=self.score,
            scores=first_scores,
        )
        second_cfg = SectionConfig(
            type="judge",
            question_start=self.question_start + first_count,
            question_count=second_count,
            options=list(self.options),
            score=self.score,
            scores=second_scores,
        )
        return (JudgeComponent(first_cfg), JudgeComponent(second_cfg))


class EssayComponent(Component):
    """简答题/填空题区域组件。"""

    def __init__(self, config: SectionConfig) -> None:
        super().__init__(config)
        self.question_start = config.question_start
        self.question_count = config.question_count
        self.lines_per_question = config.lines_per_question or 1
        self.score = config.score
        self.scores = config.scores

    def estimate_height(self, paper_size: str = "A4") -> float:
        # header 18 + question_count * (label 6 + lines_per_question * 8) + padding 16
        return (
            18.0
            + self.question_count * (6.0 + self.lines_per_question * 8.0)
            + 16.0
        )

    def render(self, page_num: int, y_offset_mm: float) -> str:
        questions_html = ""
        for i in range(self.question_count):
            q_num = self.question_start + i
            lines_html = ""
            for _ in range(self.lines_per_question):
                lines_html += '<div class="essay-line"></div>\n'
            questions_html += (
                f'<div class="essay-item">\n'
                f'  <div class="essay-label">{q_num}.</div>\n'
                f'  <div class="essay-lines">\n{lines_html}  </div>\n'
                f'</div>\n'
            )

        title = self.config.title or f"简答题（第 {self.question_start}~{self.question_start + self.question_count - 1} 题）"
        html = f'''<div class="essay-section" style="top:{y_offset_mm}mm">
  <div class="sec-title">{title}</div>
  <div class="essay-list">
    {questions_html}
  </div>
</div>
'''
        return html

    def split(
        self, available_height: float, paper_size: str = "A4"
    ) -> Optional[Tuple["Component", "Component"]]:
        needed = self.estimate_height(paper_size)
        if needed <= available_height:
            return None

        # header 18 + padding 16 = 34, each question = 6 + lines_per_question * 8
        q_height = 6.0 + self.lines_per_question * 8.0
        usable = available_height - 34.0
        if usable < q_height:
            return None

        max_questions = int(usable // q_height)
        if max_questions < 1:
            return None
        if max_questions >= self.question_count:
            return None

        first_count = max_questions
        second_count = self.question_count - first_count

        first_scores = None
        second_scores = None
        if self.scores is not None:
            first_scores = self.scores[:first_count]
            second_scores = self.scores[first_count:]

        first_cfg = SectionConfig(
            type="essay",
            question_start=self.question_start,
            question_count=first_count,
            lines_per_question=self.lines_per_question,
            score=self.score,
            scores=first_scores,
        )
        second_cfg = SectionConfig(
            type="essay",
            question_start=self.question_start + first_count,
            question_count=second_count,
            lines_per_question=self.lines_per_question,
            score=self.score,
            scores=second_scores,
        )
        return (EssayComponent(first_cfg), EssayComponent(second_cfg))
