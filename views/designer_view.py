"""答题卡设计器视图 —— 配置编辑 + 实时预览。"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from answer_sheet_generator.schema import (
    AnswerSheetConfig,
    MetaConfig,
    PageConfig,
    SectionConfig,
    StudentIdConfig,
)
from answer_sheet_generator.html_renderer import generate
from answer_sheet_generator.config_exporter import export_sheet_layout


# ── Constants ──────────────────────────────────────────────────────
_TYPE_LABELS = {"choice": "单选", "judge": "判断", "essay": "简答"}
_TYPE_OPTIONS = ["choice", "judge", "essay"]
_DEFAULT_OPTIONS = {
    "choice": ["A", "B", "C", "D"],
    "judge": ["T", "F"],
    "essay": [],
}

_DEFAULT_CONFIG = AnswerSheetConfig(
    meta=MetaConfig(title="标准化考试答题卡", paper_size="A4"),
    student_id=StudentIdConfig(digit_count=10),
    pages=[
        PageConfig(sections=[
            SectionConfig(
                type="choice",
                question_start=1,
                question_count=20,
                options=["A", "B", "C", "D"],
                score=3,
            ),
        ]),
        PageConfig(sections=[
            SectionConfig(
                type="judge",
                question_start=21,
                question_count=10,
                options=["T", "F"],
                score=2,
            ),
            SectionConfig(
                type="essay",
                question_start=31,
                question_count=1,
                lines_per_question=10,
                score=20,
            ),
        ]),
    ],
)


# ── Session-state helpers ──────────────────────────────────────────
def _get_config() -> AnswerSheetConfig:
    """从 session_state 读取当前配置，首次加载用默认值。"""
    if "designer_config" not in st.session_state:
        st.session_state.designer_config = _dict_from_config(_DEFAULT_CONFIG)
    return _config_from_dict(st.session_state.designer_config)


def _set_config(cfg: AnswerSheetConfig) -> None:
    st.session_state.designer_config = _dict_from_config(cfg)


def _config_from_dict(d: dict) -> AnswerSheetConfig:
    return AnswerSheetConfig.from_dict(d)


def _dict_from_config(cfg: AnswerSheetConfig) -> dict:
    return cfg.to_dict()


# ── Section editor widget ──────────────────────────────────────────
def _edit_section(sec_dict: dict, page_idx: int, sec_idx: int) -> dict:
    """渲染单个 section 的编辑表单，返回修改后的字典。"""
    key_prefix = f"designer_p{page_idx}_s{sec_idx}"

    sec_type = st.selectbox(
        "题型",
        options=_TYPE_OPTIONS,
        format_func=lambda x: _TYPE_LABELS.get(x, x),
        index=_TYPE_OPTIONS.index(sec_dict.get("type", "choice")),
        key=f"{key_prefix}_type",
    )
    sec_dict["type"] = sec_type

    # Custom title (optional)
    custom_title = st.text_input(
        "题型标题（可选，留空用默认）",
        value=sec_dict.get("title", ""),
        key=f"{key_prefix}_title",
    )
    if custom_title.strip():
        sec_dict["title"] = custom_title.strip()
    else:
        sec_dict.pop("title", None)

    c1, c2 = st.columns(2)
    with c1:
        sec_dict["question_start"] = st.number_input(
            "起始题号",
            min_value=1,
            value=int(sec_dict.get("question_start", 1)),
            step=1,
            key=f"{key_prefix}_start",
        )
    with c2:
        sec_dict["question_count"] = st.number_input(
            "题目数量",
            min_value=1,
            value=int(sec_dict.get("question_count", 1)),
            step=1,
            key=f"{key_prefix}_count",
        )

    # Options (choice / judge)
    if sec_type in ("choice", "judge"):
        default_opts = sec_dict.get("options") or _DEFAULT_OPTIONS[sec_type]
        opts_str = st.text_input(
            "选项（用逗号分隔）",
            value=", ".join(default_opts),
            key=f"{key_prefix}_opts",
        )
        sec_dict["options"] = [o.strip() for o in opts_str.split(",") if o.strip()]
    else:
        sec_dict.pop("options", None)

    # Lines per question (essay)
    if sec_type == "essay":
        sec_dict["lines_per_question"] = st.number_input(
            "每题行数",
            min_value=1,
            value=int(sec_dict.get("lines_per_question", 10)),
            step=1,
            key=f"{key_prefix}_lines",
        )
    else:
        sec_dict.pop("lines_per_question", None)

    # Score
    use_individual_scores = st.checkbox(
        "逐题赋分（高级）",
        value="scores" in sec_dict,
        key=f"{key_prefix}_use_scores",
    )
    if use_individual_scores:
        count = int(sec_dict.get("question_count", 1))
        current_scores = sec_dict.get("scores")
        if current_scores is None or len(current_scores) != count:
            base_score = sec_dict.get("score", 1)
            current_scores = [base_score] * count
        scores_str = st.text_input(
            "各题分值（逗号分隔）",
            value=", ".join(str(s) for s in current_scores),
            key=f"{key_prefix}_scores",
        )
        try:
            parsed = [float(s.strip()) for s in scores_str.split(",") if s.strip()]
            if len(parsed) == count:
                sec_dict["scores"] = parsed
                sec_dict.pop("score", None)
            else:
                st.warning(f"分值数量应为 {count}，当前为 {len(parsed)}")
        except ValueError:
            st.warning("分值必须为数字")
    else:
        sec_dict["score"] = st.number_input(
            "每题分值",
            min_value=0.0,
            value=float(sec_dict.get("score", 1)),
            step=0.5,
            key=f"{key_prefix}_score",
        )
        sec_dict.pop("scores", None)

    return sec_dict


def _section_summary(sec_dict: dict) -> str:
    """生成 section 的摘要文本。"""
    t = sec_dict.get("type", "choice")
    start = sec_dict.get("question_start", 1)
    count = sec_dict.get("question_count", 1)
    label = _TYPE_LABELS.get(t, t)
    end = start + count - 1
    return f"{label}  {start}-{end}题（共{count}题）"


# ── Validation ─────────────────────────────────────────────────────
def _validate_config(cfg_dict: dict) -> Optional[str]:
    """尝试从字典构造配置，返回错误信息或 None。"""
    try:
        cfg = AnswerSheetConfig.from_dict(cfg_dict)
        # Trigger __post_init__ validation
        _ = cfg.pages
        return None
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"配置错误: {e}"


# ── Export helpers ─────────────────────────────────────────────────
def _export_json(cfg: AnswerSheetConfig) -> str:
    return json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2)


def _export_sheet_layout_json(cfg: AnswerSheetConfig) -> str:
    layout = export_sheet_layout(cfg)
    return json.dumps(layout, ensure_ascii=False, indent=2)


# ── Main render function ───────────────────────────────────────────
def render_designer() -> None:
    st.subheader("答题卡设计器")
    st.caption("配置答题卡结构，实时预览并导出")

    cfg_dict = st.session_state.get("designer_config")
    if cfg_dict is None:
        cfg_dict = _dict_from_config(_DEFAULT_CONFIG)
        st.session_state.designer_config = cfg_dict

    left_col, right_col = st.columns([3, 4])

    # ═══════════════════════════════════════════════════════════════
    # Left column: config editor
    # ═══════════════════════════════════════════════════════════════
    with left_col:
        st.markdown("#### 基本信息")
        meta = cfg_dict.get("meta", {})
        meta["title"] = st.text_input(
            "答题卡标题",
            value=meta.get("title", "标准化考试答题卡"),
            key="designer_title",
        )
        meta["paper_size"] = st.selectbox(
            "纸张尺寸",
            options=["A4", "B5"],
            index=["A4", "B5"].index(meta.get("paper_size", "A4")),
            key="designer_paper_size",
        )
        cfg_dict["meta"] = meta

        sid = cfg_dict.get("student_id", {})
        sid["digit_count"] = st.number_input(
            "学号位数",
            min_value=6,
            max_value=14,
            value=int(sid.get("digit_count", 10)),
            step=1,
            key="designer_digit_count",
        )
        cfg_dict["student_id"] = sid

        st.divider()
        st.markdown("#### 页面与题型")

        pages: List[dict] = cfg_dict.get("pages", [])

        for page_idx, page_dict in enumerate(pages):
            with st.container(border=True):
                st.markdown(f"**第 {page_idx + 1} 页**")

                # Page-level title (optional, overrides meta.title)
                page_title = st.text_input(
                    "本页标题（可选，留空用全局标题）",
                    value=page_dict.get("title", ""),
                    key=f"page_title_{page_idx}",
                )
                if page_title.strip():
                    page_dict["title"] = page_title.strip()
                else:
                    page_dict.pop("title", None)

                sections: List[dict] = page_dict.get("sections", [])
                for sec_idx, sec_dict in enumerate(sections):
                    # Use a unique key for each expander
                    expander_label = _section_summary(sec_dict)
                    with st.expander(expander_label, expanded=False):
                        sections[sec_idx] = _edit_section(
                            sec_dict, page_idx, sec_idx
                        )
                        col_del, _ = st.columns([1, 4])
                        with col_del:
                            if st.button(
                                "删除此题型",
                                key=f"del_sec_{page_idx}_{sec_idx}",
                                type="secondary",
                            ):
                                sections.pop(sec_idx)
                                page_dict["sections"] = sections
                                st.rerun()

                page_dict["sections"] = sections

                if st.button(
                    "添加题型",
                    key=f"add_sec_{page_idx}",
                    type="secondary",
                ):
                    # Auto-determine next question start from ALL pages
                    next_start = 1
                    for p in pages:
                        for s in p.get("sections", []):
                            s_end = s.get("question_start", 1) + s.get("question_count", 1) - 1
                            if s_end >= next_start:
                                next_start = s_end + 1
                    sections.append({
                        "type": "choice",
                        "question_start": next_start,
                        "question_count": 5,
                        "options": ["A", "B", "C", "D"],
                        "score": 2,
                    })
                    page_dict["sections"] = sections
                    st.rerun()

                # Delete page button (only if more than one page)
                if len(pages) > 1:
                    if st.button(
                        "删除本页",
                        key=f"del_page_{page_idx}",
                        type="secondary",
                    ):
                        pages.pop(page_idx)
                        cfg_dict["pages"] = pages
                        st.rerun()

        cfg_dict["pages"] = pages

        if st.button("添加页面", key="add_page", type="secondary"):
            # Auto-calculate next question start across all pages
            next_start = 1
            for p in pages:
                for s in p.get("sections", []):
                    s_end = s.get("question_start", 1) + s.get("question_count", 1) - 1
                    if s_end >= next_start:
                        next_start = s_end + 1
            pages.append({
                "sections": [{
                    "type": "choice",
                    "question_start": next_start,
                    "question_count": 5,
                    "options": ["A", "B", "C", "D"],
                    "score": 2,
                }]
            })
            cfg_dict["pages"] = pages
            st.rerun()

        st.divider()
        st.markdown("#### 操作")

        # Build config object for export
        error_msg = _validate_config(cfg_dict)
        cfg_obj: Optional[AnswerSheetConfig] = None
        if error_msg is None:
            try:
                cfg_obj = AnswerSheetConfig.from_dict(cfg_dict)
            except Exception:
                cfg_obj = None

        preview_clicked = st.button("预览", type="primary", key="designer_preview")

        # Track preview trigger in session state so it persists across reruns
        if preview_clicked:
            st.session_state["designer_preview_triggered"] = True

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "导出配置 (JSON)",
                data=_export_json(cfg_obj) if cfg_obj else "{}",
                file_name="answer_sheet_config.json",
                mime="application/json",
                disabled=cfg_obj is None,
                key="dl_json",
            )
        with c2:
            if cfg_obj:
                try:
                    html_content = generate(cfg_obj)
                    st.download_button(
                        "导出答题卡 (HTML)",
                        data=html_content,
                        file_name="answer_sheet.html",
                        mime="text/html",
                        key="dl_html",
                    )
                except Exception as e:
                    st.download_button(
                        "导出答题卡 (HTML)",
                        data="",
                        file_name="answer_sheet.html",
                        mime="text/html",
                        disabled=True,
                        key="dl_html",
                    )
            else:
                st.download_button(
                    "导出答题卡 (HTML)",
                    data="",
                    file_name="answer_sheet.html",
                    mime="text/html",
                    disabled=True,
                    key="dl_html",
                )
        with c3:
            st.download_button(
                "导出识别配置 (sheet_layout.json)",
                data=_export_sheet_layout_json(cfg_obj) if cfg_obj else "{}",
                file_name="sheet_layout.json",
                mime="application/json",
                disabled=cfg_obj is None,
                key="dl_layout",
            )

    # ═══════════════════════════════════════════════════════════════
    # Right column: preview
    # ═══════════════════════════════════════════════════════════════
    with right_col:
        st.markdown("#### 预览")

        if error_msg:
            st.error(f"配置有误，无法预览：\n\n{error_msg}")
        elif cfg_obj is None:
            st.error("配置解析失败，请检查参数")
        else:
            st.success("配置有效")

            # Always show preview if config is valid (or if preview was explicitly clicked)
            try:
                html_content = generate(cfg_obj)
                # Use a fixed reasonable height for the preview iframe
                components.html(html_content, height=800, scrolling=True)
            except Exception as e:
                st.error(f"HTML 生成失败: {e}")

    # Persist back to session state
    st.session_state.designer_config = cfg_dict
