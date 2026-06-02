"""Auto Grading System - Streamlit GUI"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from modules.llm_essay_grader import load_config, save_config
from modules.essay_recognizer import SUPPORTED_ENGINES, check_engine_available
from modules.defaults import path_constants, DEFAULT_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_OCR_MODEL
from modules.blank_calibrator import (
    load_baseline, save_baseline, compute_blank_baseline,
    get_judge_baseline_dict, get_choice_baseline_dict,
    get_judge_zone_bounds, get_choice_zone_bounds,
)

from views.single_view import render_single
from views.batch_view import render_batch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = path_constants(BASE_DIR)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="智能阅卷系统",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
[data-testid="stMetricValue"] { font-size: 1.8rem; }
[data-testid="stMetricLabel"] { font-size: 0.85rem; }
.streamlit-expanderHeader { font-size: 0.95rem; }
.stDataFrame { font-size: 0.9rem; }
</style>""", unsafe_allow_html=True)


def _split_semicolon(text):
    """按英文/中文分号分割字符串，返回去空白的非空片段列表。"""
    import re
    parts = re.split(r'[;；]', text)
    return [p.strip() for p in parts if p.strip()]


def _merge_primary_and_fallback(primary, fallback_list):
    """合并主值和备用列表，去重，保持顺序。支持 str | list[str]。

    单元素时转回字符串，保持下游兼容。
    """
    result = []
    if primary:
        if isinstance(primary, str):
            result.append(primary)
        else:
            result.extend(primary)
    if isinstance(fallback_list, list):
        for item in fallback_list:
            if item and item not in result:
                result.append(item)
    if len(result) == 1:
        return result[0]
    return result if result else ""


# ── Utility functions ──────────────────────────────────────────────
def _load_grading_service():
    if os.path.exists(PATHS['answer_key']):
        try:
            from modules.grading import GradingService
            return GradingService.from_xlsx(PATHS['answer_key'])
        except Exception as e:
            st.error(f"参考答案文件加载失败: {e}")
    return None


# ── Config validation (show warnings in sidebar) ───────────────────
from modules.config_validator import validate_all
_warnings = validate_all(BASE_DIR)

# ── Blank baseline loading ─────────────────────────────────────────
_BLANK_BASELINE_PATH = os.path.join(BASE_DIR, "config", "blank_baseline.json")
_blank_baseline_raw = load_baseline(_BLANK_BASELINE_PATH)
_choice_baseline = get_choice_baseline_dict(_blank_baseline_raw)
_judge_baseline = get_judge_baseline_dict(_blank_baseline_raw)
_choice_zone_bounds = get_choice_zone_bounds(_blank_baseline_raw)
_judge_zone_bounds = get_judge_zone_bounds(_blank_baseline_raw)


# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    if _warnings:
        with st.expander("配置警告", expanded=True):
            for w in _warnings:
                st.warning(w)

    st.header("参数设置")
    threshold = st.slider(
        "填涂阈值", 0.02, 0.30, 0.06, 0.01,
        help="黑色像素占比超过此值则判定为已填涂"
    )

    st.divider()
    st.header("API Key 配置")
    API_KEYS_PATH = PATHS['api_keys']
    MODEL_CFG_PATH = PATHS['model_config']
    _keys = load_config(API_KEYS_PATH)
    _mcfg = load_config(MODEL_CFG_PATH)
    _saved_api_key = _keys.get("api_key", "")
    _saved_ocr_key = _keys.get("ocr_api_key", "")

    _api_key_input = st.text_input(
        "ModelScope API Key",
        value=_saved_api_key, type="password",
        help="用于 LLM 评分和在线 OCR（同一个 Key 即可）")
    with st.expander("备用 API Key（可选）"):
        _api_keys_fallback_input = st.text_area(
            "备用 Key 列表（用 ; 或 ； 分隔，限流时自动轮询）",
            value="; ".join(_keys.get("api_keys", [])),
            key="fallback_api_keys",
            height=80)
    _use_same_key = st.checkbox(
        "在线 OCR 使用同一个 Key",
        value=not _saved_ocr_key.strip() or _saved_ocr_key == _saved_api_key)
    if _use_same_key:
        _ocr_key_input = _api_key_input
    else:
        _ocr_key_input = st.text_input(
            "OCR 专用 API Key",
            value=_saved_ocr_key, type="password")
    st.caption("提示：主 Key 已保存时无需重复填写，可直接保存备用 Key")
    if st.button("保存 API Key", type="primary"):
        _fallback_keys = _split_semicolon(_api_keys_fallback_input)
        _cfg_to_save = {"api_keys": _fallback_keys}
        if _api_key_input.strip():
            _cfg_to_save["api_key"] = _api_key_input
        if _ocr_key_input.strip():
            _cfg_to_save["ocr_api_key"] = _ocr_key_input
        save_config(_cfg_to_save, API_KEYS_PATH)
        st.success(f"已保存 {len(_fallback_keys)} 个备用 Key" if _fallback_keys else "已清空备用 Key")
    # 刷新读取（保存后立即生效）
    _keys = load_config(API_KEYS_PATH)
    _api_key = _keys.get("api_key", "")
    _api_key = _merge_primary_and_fallback(_api_key, _keys.get("api_keys", []))
    # OCR key：优先专用 key，否则复用 LLM key 池；都合并备用 key 列表
    _ocr_primary = _keys.get("ocr_api_key", "")
    if _ocr_primary.strip():
        _ocr_key = _merge_primary_and_fallback(_ocr_primary, _keys.get("api_keys", []))
    else:
        _ocr_key = _api_key

    st.divider()
    st.header("OCR 引擎")
    _available = [(e, f"{e} {'✓' if check_engine_available(e) else '✗ 未安装'}")
                  for e in SUPPORTED_ENGINES]
    _eng_labels = [label for _, label in _available]
    _eng_values = [e for e, _ in _available]
    _eng_default = 0
    _selected_idx = st.selectbox("简答题 OCR 引擎", range(len(_eng_labels)),
                                  format_func=lambda i: _eng_labels[i],
                                  index=_eng_default)
    ocr_engine = _eng_values[_selected_idx]
    online_ocr_model = _mcfg.get("ocr_model", DEFAULT_OCR_MODEL)
    if ocr_engine == 'online':
        online_ocr_model = st.text_input(
            "在线 OCR 模型",
            value=_mcfg.get("ocr_model", DEFAULT_OCR_MODEL),
            help="视觉模型 ID，需支持 image_url 输入")
        _has_ocr_key = bool(_ocr_key.strip() if isinstance(_ocr_key, str) else _ocr_key)
        if not _has_ocr_key:
            st.warning("在线 OCR 需要配置 API Key")
    elif not check_engine_available(ocr_engine):
        st.warning(f"`{ocr_engine}` 未安装，将回退到 paddleocr")
        ocr_engine = 'paddleocr'

    st.divider()
    st.header("LLM 评分设置")
    _ocr_api_config = {}
    llm_enabled = st.checkbox("启用 LLM 简答题评分",
                               value=bool(_api_key.strip() if isinstance(_api_key, str) else _api_key),
                               help="勾选后自动评分简答题，不勾选则需人工评分")
    llm_api_key = _api_key
    llm_base_url = _merge_primary_and_fallback(
        _mcfg.get("base_url", DEFAULT_BASE_URL),
        _mcfg.get("fallback_base_urls", []))
    llm_model = _merge_primary_and_fallback(
        _mcfg.get("llm_model", DEFAULT_LLM_MODEL),
        _mcfg.get("llm_models", []))
    llm_max_tokens = _mcfg.get("llm_max_tokens", 256)
    llm_temperature = _mcfg.get("llm_temperature", 0.3)
    _has_key = bool(llm_api_key.strip() if isinstance(llm_api_key, str) else llm_api_key)
    if llm_enabled:
        if not _has_key:
            st.warning("未配置 API Key，请在上方填写")
        llm_base_url = st.text_input(
            "Base URL",
            value=_mcfg.get("base_url", DEFAULT_BASE_URL),
            help="API 端点地址")
        # st.text_input 的 value 必须是字符串；若配置中 llm_model 为列表
        # （多模型 fallback 合并后的结果），取第一个元素作为主模型显示
        _primary_model = _mcfg.get("llm_model", DEFAULT_LLM_MODEL)
        if isinstance(_primary_model, list):
            _primary_model = _primary_model[0] if _primary_model else DEFAULT_LLM_MODEL
        llm_model = st.text_input(
            "模型", value=_primary_model,
            help="模型 ID")
        with st.expander("备用模型（可选）"):
            _llm_models_fallback_input = st.text_area(
                "备用模型列表（用 ; 或 ； 分隔，限流时自动降级）",
                value="; ".join(_mcfg.get("llm_models", ["deepseek-ai/DeepSeek-V4-Pro"])),
                height=80)
        # 用 UI 上的备用模型（含默认值）重新构建 llm_model，无需点击保存即可生效
        _ui_fallback_models = _split_semicolon(_llm_models_fallback_input)
        llm_model = _merge_primary_and_fallback(llm_model, _ui_fallback_models)
        if st.button("保存模型配置"):
            _fallback_models = _split_semicolon(_llm_models_fallback_input)
            save_config({"base_url": llm_base_url,
                          "llm_model": llm_model,
                          "llm_models": _fallback_models,
                          "ocr_model": online_ocr_model}, MODEL_CFG_PATH)
            st.success("模型配置已保存")
    _ocr_api_config = {
        'api_key': _ocr_key,
        'base_url': llm_base_url,
        'ocr_model': online_ocr_model if ocr_engine == 'online' else '',
        'ocr_max_tokens': _mcfg.get("ocr_max_tokens", 1024),
        'ocr_prompt': _mcfg.get("ocr_prompt",
            "请逐行识别图片中的所有文字内容，只输出文字，不要添加解释。"),
    }
    if llm_enabled and _has_key:
        st.success("LLM 评分已启用")
    elif not llm_enabled:
        st.info("未启用 LLM 评分，简答题需人工评分")

    st.divider()
    st.header("参考答案")
    if os.path.exists(PATHS['answer_key']):
        st.success("参考答案已加载")
    else:
        st.warning("参考答案文件未找到")

    st.divider()
    st.header("空白答卷校准")
    _baseline_loaded = bool(_choice_baseline or _judge_baseline)
    if _baseline_loaded:
        _ch_cnt = len(_choice_baseline) if _choice_baseline else 0
        _ju_cnt = len(_judge_baseline) if _judge_baseline else 0
        st.success(f"已加载空白基准（选择 {_ch_cnt} 题 / 判断 {_ju_cnt} 题）")
    else:
        st.info("未配置空白基准")
    with st.expander("上传空白答卷"):
        st.markdown("**第1页（选择题）**")
        _blank_file_p1 = st.file_uploader(
            "选择空白答题卡第1页",
            type=["png", "jpg", "jpeg"],
            key="blank_sheet_p1",
            help="上传未填涂的答题卡第1页，用于计算选择题灰度基准")
        if _blank_file_p1 is not None and st.button("生成选择题基准", key="calib_p1"):
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix="." + _blank_file_p1.name.split(".")[-1]) as tmp:
                    tmp.write(_blank_file_p1.getvalue())
                    tmp_path = tmp.name
                baseline = compute_blank_baseline(tmp_path, page=1)
                # 合并到现有基准（保留判断题数据）
                existing = load_baseline(_BLANK_BASELINE_PATH) or {}
                existing["choice"] = baseline.get("choice", {})
                save_baseline(existing, _BLANK_BASELINE_PATH)
                _choice_baseline = get_choice_baseline_dict(existing)
                _choice_zone_bounds = get_choice_zone_bounds(existing)
                os.unlink(tmp_path)
                st.success(f"选择题基准已保存（{len(baseline.get('choice', {}).get('questions', {}))} 题）")
                st.rerun()
            except Exception as e:
                st.error(f"校准失败: {e}")

        st.markdown("**第2页（判断题）**")
        _blank_file_p2 = st.file_uploader(
            "选择空白答题卡第2页",
            type=["png", "jpg", "jpeg"],
            key="blank_sheet_p2",
            help="上传未填涂的答题卡第2页，用于计算判断题灰度基准")
        if _blank_file_p2 is not None and st.button("生成判断题基准", key="calib_p2"):
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix="." + _blank_file_p2.name.split(".")[-1]) as tmp:
                    tmp.write(_blank_file_p2.getvalue())
                    tmp_path = tmp.name
                baseline = compute_blank_baseline(tmp_path, page=2)
                existing = load_baseline(_BLANK_BASELINE_PATH) or {}
                existing["judge"] = baseline.get("judge", {})
                save_baseline(existing, _BLANK_BASELINE_PATH)
                _judge_baseline = get_judge_baseline_dict(existing)
                _judge_zone_bounds = get_judge_zone_bounds(existing)
                os.unlink(tmp_path)
                st.success(f"判断题基准已保存（{len(baseline.get('judge', {}).get('questions', {}))} 题）")
                st.rerun()
            except Exception as e:
                st.error(f"校准失败: {e}")
        st.caption("说明：如果淡铅笔填涂被误判为未填涂，上传空白答卷可提升识别精度。")

    st.divider()
    with st.expander("使用说明"):
        st.markdown(
            "**单套识别**: 上传一份答题卡的正反两页图片，"
            "系统分步展示预处理、版面分析、识别、评分全过程。\n\n"
            "**批量阅卷**: 将图片放入文件夹，文件按名称排序——奇数位"
            "(第1、3、…张)为**第1页**(学号+选择题)，"
            "偶数位(第2、4、…张)为**第2页**(判断题+简答题)。\n\n"
            "**提示**: 如果识别结果偏多或偏少，请调整填涂阈值滑块。"
        )


# ── Main area ──────────────────────────────────────────────────────
st.title("智能阅卷系统")
st.caption("上传答题卡图片，自动识别填涂并评分  |  计算机视觉课程设计")

tab_single, tab_batch = st.tabs(["单套识别", "批量阅卷"])

with tab_single:
    render_single(
        threshold=threshold,
        llm_enabled=llm_enabled,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_max_tokens=llm_max_tokens,
        llm_temperature=llm_temperature,
        ocr_engine=ocr_engine,
        ocr_api_config=_ocr_api_config,
        PATHS=PATHS,
        choice_baseline=_choice_baseline,
        judge_baseline=_judge_baseline,
        choice_zone_bounds=_choice_zone_bounds,
        judge_zone_bounds=_judge_zone_bounds,
    )

with tab_batch:
    render_batch(
        threshold=threshold,
        llm_enabled=llm_enabled,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_max_tokens=llm_max_tokens,
        llm_temperature=llm_temperature,
        ocr_engine=ocr_engine,
        ocr_api_config=_ocr_api_config,
        PATHS=PATHS,
        choice_baseline=_choice_baseline,
        judge_baseline=_judge_baseline,
        choice_zone_bounds=_choice_zone_bounds,
        judge_zone_bounds=_judge_zone_bounds,
    )
