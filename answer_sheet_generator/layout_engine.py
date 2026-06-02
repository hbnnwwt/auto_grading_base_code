"""答题卡分页布局引擎。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from answer_sheet_generator.components import (
    ChoiceComponent,
    Component,
    EssayComponent,
    JudgeComponent,
    StudentIdComponent,
)
from answer_sheet_generator.schema import AnswerSheetConfig, SectionConfig, StudentIdConfig

PAGE_SIZES = {
    "A4": {"width": 210, "height": 297, "usable_width": 190, "usable_height": 277},
    "B5": {"width": 176, "height": 250, "usable_width": 156, "usable_height": 230},
}

HEADER_HEIGHT = 25


class LayoutError(Exception):
    """布局错误：某个 section 在单页内无法容纳且不可拆分。"""


@dataclass
class Page:
    """分页结果中的一页。"""

    page_number: int
    components: List[Tuple[Component, float]] = field(default_factory=list)

    def add_component(self, comp: Component, y_offset: float) -> None:
        """向本页添加一个组件及其 y 轴偏移量。"""
        self.components.append((comp, y_offset))

    def total_height(self, paper_size: str) -> float:
        """计算本页所有组件的总高度（不含页眉）。"""
        return sum(comp.estimate_height(paper_size) for comp, _ in self.components)


def _create_component(section: SectionConfig) -> Component:
    """根据 SectionConfig 创建对应的 Component 子类实例。"""
    if section.type == "choice":
        return ChoiceComponent(section)
    if section.type == "judge":
        return JudgeComponent(section)
    if section.type == "essay":
        return EssayComponent(section)
    raise ValueError(f"未知的 section 类型: {section.type}")


def paginate(cfg: AnswerSheetConfig) -> List[Page]:
    """将 AnswerSheetConfig 分页为 List[Page]。

    算法：
    1. 根据纸张尺寸计算每页可用净高度（usable_height - HEADER_HEIGHT）。
    2. 第一页自动放入 StudentIdComponent。
    3. 收集所有 section，按 question_start 排序。
    4. 逐个尝试放入当前页；若放不下则尝试 split；若 split 失败则抛 LayoutError。
    """
    paper_size = cfg.meta.paper_size
    if paper_size not in PAGE_SIZES:
        raise ValueError(f"不支持的纸张尺寸: {paper_size}")

    usable_height = PAGE_SIZES[paper_size]["usable_height"]
    page_net_height = usable_height - HEADER_HEIGHT

    pages: List[Page] = []
    current_page = Page(page_number=1)
    current_used = 0.0

    # 第一页自动添加学号组件
    sid_comp = StudentIdComponent(cfg.student_id)
    sid_height = sid_comp.estimate_height(paper_size)
    current_page.add_component(sid_comp, 0.0)
    current_used = sid_height

    # 收集所有 section 并按 question_start 排序
    all_sections: List[SectionConfig] = []
    for page_cfg in cfg.pages:
        all_sections.extend(page_cfg.sections)
    all_sections.sort(key=lambda s: s.question_start)

    for section in all_sections:
        comp = _create_component(section)
        comp_height = comp.estimate_height(paper_size)

        while True:
            remaining = page_net_height - current_used

            if comp_height <= remaining:
                # 直接放入当前页
                current_page.add_component(comp, current_used)
                current_used += comp_height
                break

            # 尝试拆分
            split_result = comp.split(remaining, paper_size)
            if split_result is None:
                # 无法拆分：若组件能放入全新页面，则开新页；否则报错
                if comp_height <= page_net_height:
                    pages.append(current_page)
                    current_page = Page(page_number=len(pages) + 1)
                    current_used = 0.0
                    # 继续 while 循环，将 comp 放入新页
                    continue
                raise LayoutError(
                    f"Section {section.type} (题号 {section.question_start}~"
                    f"{section.question_start + section.question_count - 1}) "
                    f"高度 {comp_height:.1f}mm 超过单页净高度 {page_net_height:.1f}mm，"
                    f"且不可拆分。"
                )

            first_part, second_part = split_result
            first_height = first_part.estimate_height(paper_size)

            # 检查拆分后的第一部分是否真的能放下
            if first_height > remaining:
                # 理论上 split 应该保证这一点，但做防御性检查
                raise LayoutError(
                    f"Section {section.type} (题号 {section.question_start}~) "
                    f"拆分后第一部分高度 {first_height:.1f}mm 仍超过剩余空间 "
                    f"{remaining:.1f}mm。"
                )

            current_page.add_component(first_part, current_used)
            current_used += first_height

            # 创建新页继续处理第二部分
            pages.append(current_page)
            current_page = Page(page_number=len(pages) + 1)
            current_used = 0.0
            comp = second_part
            comp_height = comp.estimate_height(paper_size)
            # 继续 while 循环，尝试将 second_part 放入新页

    # 别忘了把最后一页加进去
    if current_page.components:
        pages.append(current_page)

    # 重新编号页码（确保连续）
    for idx, page in enumerate(pages, start=1):
        page.page_number = idx

    return pages
