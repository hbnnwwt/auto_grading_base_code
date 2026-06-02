"""空白标定向导 —— 4 步上传、确认、计算、保存。"""

import json
import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from views.components import load_image_from_bytes, image_to_bytes


# ── Constants ──────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_LAYOUT_PATH = os.path.join(_BASE_DIR, "config", "sheet_layout.json")
_BASELINE_PATH = os.path.join(_BASE_DIR, "config", "blank_baseline.json")


# ── Layout helpers ─────────────────────────────────────────────────
def _load_layout():
    """加载 sheet_layout.json，返回 dict。"""
    if os.path.exists(_LAYOUT_PATH):
        with open(_LAYOUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _get_page_count(layout):
    """根据布局配置推断页数。"""
    if layout is None:
        return 2
    pages = layout.get("_pages")
    if pages is not None:
        return len(pages)
    # 旧逻辑：有 choice 和 judge 各一页
    return 2


def _get_expected_sections(layout, page_idx):
    """返回某页期望包含的 section 类型列表。"""
    pages = layout.get("_pages") if layout else None
    if pages and page_idx < len(pages):
        return pages[page_idx]
    if page_idx == 0:
        return ["choice"]
    if page_idx == 1:
        return ["judge"]
    return []


def _get_expected_question_counts(layout):
    """返回期望的选择题和判断题数量。"""
    if layout is None:
        return 20, 10
    ch = layout.get("choice", {})
    ju = layout.get("judge", {})
    return (
        ch.get("question_count", 20),
        ju.get("question_count", 10),
    )


# ── Image helpers ──────────────────────────────────────────────────
def _save_uploaded_file(uploaded_file):
    """将 UploadedFile 保存到临时文件，返回路径。"""
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def _make_thumbnail(image_bgr, max_size=300):
    """生成缩略图（RGB），返回 bytes。"""
    h, w = image_bgr.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        thumb = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        thumb = image_bgr
    thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    return image_to_bytes(thumb_rgb, ext=".png")


# ── Step renderers ─────────────────────────────────────────────────
def _render_step_1(layout, page_count):
    """Step 1: 上传空白答卷。"""
    st.markdown("#### 第 1 步：上传空白答卷")
    st.info(
        "请打印生成的答题卡，**不要填涂任何选项**，"
        "用扫描仪或手机拍照后上传。"
    )

    uploaded = []
    for i in range(page_count):
        page_idx = i
        sections = _get_expected_sections(layout, page_idx)
        sec_labels = []
        for s in sections:
            if s == "choice":
                sec_labels.append("选择题")
            elif s == "judge":
                sec_labels.append("判断题")
            elif s == "essay":
                sec_labels.append("简答题")
            elif s == "student_id":
                sec_labels.append("学号")
            else:
                sec_labels.append(s)
        label = f"第 {page_idx + 1} 页"
        if sec_labels:
            label += f"（{' + '.join(sec_labels)}）"

        f = st.file_uploader(
            label,
            type=["png", "jpg", "jpeg"],
            key=f"calib_upload_p{page_idx}",
        )
        uploaded.append(f)

    all_uploaded = all(f is not None for f in uploaded)
    col_back, col_next = st.columns([1, 1])
    with col_back:
        st.write("")  # placeholder
    with col_next:
        if st.button("下一步", type="primary", disabled=not all_uploaded):
            # 保存到 session state
            temp_paths = []
            thumbs = []
            for f in uploaded:
                path = _save_uploaded_file(f)
                temp_paths.append(path)
                img = load_image_from_bytes(f.getvalue())
                thumbs.append(_make_thumbnail(img))
            st.session_state.calib_temp_paths = temp_paths
            st.session_state.calib_thumbnails = thumbs
            st.session_state.calib_step = 2
            st.rerun()


def _render_step_2(layout, page_count):
    """Step 2: 确认检测区域。"""
    st.markdown("#### 第 2 步：确认检测区域")

    temp_paths = st.session_state.get("calib_temp_paths", [])
    thumbs = st.session_state.get("calib_thumbnails", [])

    # 运行版面分析获取区域信息
    from modules.preprocess import ImagePreprocessor
    from modules.layout import LayoutAnalyzer

    preprocessor = ImagePreprocessor()
    analyzer = LayoutAnalyzer()

    images = []
    binaries = []
    corrected_images = []
    for path in temp_paths:
        img = preprocessor.load(path)
        corrected, _, _, binary = preprocessor.process(img)
        images.append(img)
        binaries.append(binary)
        corrected_images.append(corrected)

    pages_config = layout.get("_pages") if layout else None
    if pages_config:
        regions_list = analyzer.analyze_multipage(images, binaries)
    else:
        regions_list = []
        for idx, (corr, bin_img) in enumerate(zip(corrected_images, binaries), start=1):
            regions_list.append(analyzer.analyze(corr, bin_img, page=idx))

    # 存储供下一步使用
    st.session_state.calib_regions_list = regions_list
    st.session_state.calib_corrected_images = corrected_images

    # 显示缩略图和检测统计
    expected_ch, expected_ju = _get_expected_question_counts(layout)
    total_detected_choice = 0
    total_detected_judge = 0

    for page_idx, (thumb, regions) in enumerate(zip(thumbs, regions_list)):
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(thumb, caption=f"第 {page_idx + 1} 页")
            with c2:
                sections = _get_expected_sections(layout, page_idx)
                st.markdown(f"**期望区域**: {', '.join(sections)}")
                detected = []
                for sec in sections:
                    if regions.get(sec) is not None:
                        detected.append(sec)
                        if sec == "choice":
                            total_detected_choice += 1
                        elif sec == "judge":
                            total_detected_judge += 1
                st.markdown(f"**检测到**: {', '.join(detected) if detected else '无'}")

    # 汇总统计
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("选择题区域", f"{total_detected_choice} 页")
    with col2:
        st.metric("判断题区域", f"{total_detected_judge} 页")

    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("返回"):
            st.session_state.calib_step = 1
            st.rerun()
    with col_next:
        if st.button("下一步", type="primary"):
            st.session_state.calib_step = 3
            st.rerun()


def _render_step_3(layout):
    """Step 3: 计算基准。"""
    st.markdown("#### 第 3 步：计算空白基准")

    temp_paths = st.session_state.get("calib_temp_paths", [])
    corrected_images = st.session_state.get("calib_corrected_images", [])
    regions_list = st.session_state.get("calib_regions_list", [])

    progress_bar = st.progress(0, text="正在计算空白基准...")

    try:
        from modules.blank_calibrator import compute_blank_baseline_multipage

        # 传递 layout 和预处理好的数据
        # compute_blank_baseline_multipage 会重新预处理，这里我们直接调用
        baseline = compute_blank_baseline_multipage(temp_paths, layout=layout)

        progress_bar.progress(100, text="计算完成！")

        st.session_state.calib_baseline = baseline
        st.session_state.calib_step = 4
        st.rerun()
    except Exception as e:
        progress_bar.empty()
        st.error(f"计算失败: {e}")
        if st.button("返回重试"):
            st.session_state.calib_step = 2
            st.rerun()


def _render_step_4():
    """Step 4: 保存结果。"""
    st.markdown("#### 第 4 步：保存空白基准")

    baseline = st.session_state.get("calib_baseline", {})

    # 统计信息
    ch_questions = baseline.get("choice", {}).get("questions", {})
    ju_questions = baseline.get("judge", {}).get("questions", {})

    col1, col2 = st.columns(2)
    with col1:
        st.metric("选择题气泡数", len(ch_questions))
    with col2:
        st.metric("判断题气泡数", len(ju_questions))

    # 显示部分数据预览
    if ch_questions:
        with st.expander(f"选择题基准预览（共 {len(ch_questions)} 题）"):
            sample = dict(list(ch_questions.items())[:5])
            for q, data in sample.items():
                zones = data.get("zones", [])
                means = [f"{z['mean']:.1f}" for z in zones]
                st.text(f"题 {q}: 均值 = {', '.join(means)}")

    if ju_questions:
        with st.expander(f"判断题基准预览（共 {len(ju_questions)} 题）"):
            sample = dict(list(ju_questions.items())[:5])
            for q, data in sample.items():
                zones = data.get("zones", [])
                means = [f"{z['mean']:.1f}" for z in zones]
                st.text(f"题 {q}: 均值 = {', '.join(means)}")

    st.divider()
    col_save, col_reset = st.columns([1, 1])
    with col_save:
        if st.button("保存空白基准", type="primary"):
            try:
                from modules.blank_calibrator import save_baseline
                save_baseline(baseline, _BASELINE_PATH)
                st.success(f"已保存到 {os.path.relpath(_BASELINE_PATH, _BASE_DIR)}")
            except Exception as e:
                st.error(f"保存失败: {e}")
    with col_reset:
        if st.button("重新开始"):
            # 清理临时文件
            for path in st.session_state.get("calib_temp_paths", []):
                try:
                    os.unlink(path)
                except OSError:
                    pass
            # 清理 session state
            for key in list(st.session_state.keys()):
                if key.startswith("calib_"):
                    del st.session_state[key]
            st.session_state.calib_step = 1
            st.rerun()


# ── Main render function ───────────────────────────────────────────
def render_calibration():
    """渲染空白标定向导主界面。"""
    # 初始化步骤
    if "calib_step" not in st.session_state:
        st.session_state.calib_step = 1

    layout = _load_layout()
    page_count = _get_page_count(layout)

    step = st.session_state.calib_step

    # 步骤指示器
    steps = ["上传", "确认", "计算", "保存"]
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 < step:
                st.success(f"✓ {label}")
            elif i + 1 == step:
                st.info(f"▶ {label}")
            else:
                st.caption(f"○ {label}")

    st.divider()

    if step == 1:
        _render_step_1(layout, page_count)
    elif step == 2:
        _render_step_2(layout, page_count)
    elif step == 3:
        _render_step_3(layout)
    elif step == 4:
        _render_step_4()
