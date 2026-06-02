"""将 AnswerSheetConfig 导出为 backward-compatible 的 sheet_layout.json 格式。"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from answer_sheet_generator.components import (
    ChoiceComponent,
    Component,
    EssayComponent,
    JudgeComponent,
    StudentIdComponent,
)
from answer_sheet_generator.layout_engine import PAGE_SIZES, paginate
from answer_sheet_generator.schema import AnswerSheetConfig, SectionConfig


def _cols_for_paper(paper_size: str) -> int:
    """根据纸张尺寸返回每行可容纳的题数。"""
    return 5  # A4 / B5 固定 5 列


def _merge_sections(
    sections: List[SectionConfig], paper_size: str
) -> Optional[Dict[str, Any]]:
    """将多个同类型 section 合并为 old-format 的单一字典。

    Returns None if sections is empty.
    """
    if not sections:
        return None

    total_count = sum(s.question_count for s in sections)
    question_start = min(s.question_start for s in sections)
    cols = _cols_for_paper(paper_size)
    rows = math.ceil(total_count / cols)

    # options 取第一个 section 的（同类型 section 的 options 应一致）
    options = sections[0].options

    return {
        "rows": rows,
        "cols": cols,
        "question_start": question_start,
        "question_count": total_count,
        "options": options,
    }


def _build_old_format(
    cfg: AnswerSheetConfig,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """构建 old-format 的 student_id, choice, judge 字典。

    从原始 cfg.pages 中收集所有 section（不是 paginate 后的组件），
    按类型合并。
    """
    student_id = {"digit_count": cfg.student_id.digit_count}

    choice_sections: List[SectionConfig] = []
    judge_sections: List[SectionConfig] = []

    for page_cfg in cfg.pages:
        for sec in page_cfg.sections:
            if sec.type == "choice":
                choice_sections.append(sec)
            elif sec.type == "judge":
                judge_sections.append(sec)

    choice = _merge_sections(choice_sections, cfg.meta.paper_size)
    judge = _merge_sections(judge_sections, cfg.meta.paper_size)

    return student_id, choice, judge


def _build_fallback_layout(
    cfg: AnswerSheetConfig,
) -> Dict[str, Dict[str, List[float]]]:
    """为每页计算 fallback 的相对 y 范围。

    调用 paginate() 获取分页结果，对每个 page 中的每个 component
    （除 StudentIdComponent 外）计算相对 y 范围：
        rel_start = y_offset / page_net_height
        rel_end   = (y_offset + height) / page_net_height
    四舍五入到 4 位小数。
    """
    paper_size = cfg.meta.paper_size
    usable_height = PAGE_SIZES[paper_size]["usable_height"]
    page_net_height = usable_height - 25  # HEADER_HEIGHT = 25

    pages = paginate(cfg)
    layout: Dict[str, Dict[str, List[float]]] = {}

    for page in pages:
        page_key = f"page{page.page_number}_fallback"
        page_fallback: Dict[str, List[float]] = {}

        for comp, y_offset in page.components:
            if isinstance(comp, StudentIdComponent):
                continue

            height = comp.estimate_height(paper_size)
            rel_start = round(y_offset / page_net_height, 4)
            rel_end = round((y_offset + height) / page_net_height, 4)

            # Clamp to [0, 1]
            rel_start = max(0.0, min(1.0, rel_start))
            rel_end = max(0.0, min(1.0, rel_end))

            # Determine key based on component type
            if isinstance(comp, ChoiceComponent):
                key = "choice"
            elif isinstance(comp, JudgeComponent):
                key = "judge"
            elif isinstance(comp, EssayComponent):
                key = "essay"
            else:
                continue

            # If key already exists, merge ranges (extend to min/max)
            if key in page_fallback:
                existing = page_fallback[key]
                page_fallback[key] = [
                    min(existing[0], rel_start),
                    max(existing[1], rel_end),
                ]
            else:
                page_fallback[key] = [rel_start, rel_end]

        if page_fallback:
            layout[page_key] = page_fallback

    return layout


def _build_pages_field(cfg: AnswerSheetConfig) -> List[Dict[str, Any]]:
    """构建 _pages 字段，用于新的多页 LayoutAnalyzer。

    从原始 cfg.pages 中收集 section 信息（不是 paginate 后的组件）。
    """
    pages_field: List[Dict[str, Any]] = []

    for page_idx, page_cfg in enumerate(cfg.pages, start=1):
        sections: List[Dict[str, Any]] = []
        for sec in page_cfg.sections:
            sec_dict: Dict[str, Any] = {
                "type": sec.type,
                "question_start": sec.question_start,
                "question_count": sec.question_count,
            }
            if sec.title is not None:
                sec_dict["title"] = sec.title
            if sec.options is not None:
                sec_dict["options"] = sec.options
            if sec.lines_per_question is not None:
                sec_dict["lines_per_question"] = sec.lines_per_question
            sections.append(sec_dict)

        page_dict: Dict[str, Any] = {
            "page_number": page_idx,
            "sections": sections,
        }
        if page_cfg.title is not None:
            page_dict["title"] = page_cfg.title

        pages_field.append(page_dict)

    return pages_field


def export_sheet_layout(cfg: AnswerSheetConfig) -> dict:
    """将 AnswerSheetConfig 导出为 backward-compatible 的 sheet_layout.json 格式。

    输出包含：
    1. Old-format 字段（student_id, choice, judge, layout）供现有识别器使用。
    2. _pages 字段供新的多页 LayoutAnalyzer 使用。
    """
    student_id, choice, judge = _build_old_format(cfg)
    fallback_layout = _build_fallback_layout(cfg)
    pages_field = _build_pages_field(cfg)

    result: Dict[str, Any] = {
        "student_id": student_id,
        "layout": fallback_layout,
        "_pages": pages_field,
    }

    if choice is not None:
        result["choice"] = choice
    if judge is not None:
        result["judge"] = judge

    return result
