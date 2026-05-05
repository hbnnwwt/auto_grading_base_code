"""Auto Grading System - Streamlit GUI"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from modules.llm_essay_grader import load_config, save_config
from modules.essay_recognizer import SUPPORTED_ENGINES, check_engine_available
from modules.defaults import path_constants, DEFAULT_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_OCR_MODEL

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
    _use_same_key = st.checkbox(
        "在线 OCR 使用同一个 Key",
        value=not _saved_ocr_key.strip() or _saved_ocr_key == _saved_api_key)
    if _use_same_key:
        _ocr_key_input = _api_key_input
    else:
        _ocr_key_input = st.text_input(
            "OCR 专用 API Key",
            value=_saved_ocr_key, type="password")
    if st.button("保存 API Key"):
        save_config({"api_key": _api_key_input,
                      "ocr_api_key": _ocr_key_input}, API_KEYS_PATH)
        st.success("API Key 已保存")
    # 刷新读取（保存后立即生效）
    _keys = load_config(API_KEYS_PATH)
    _api_key = _keys.get("api_key", "")
    _ocr_key = _keys.get("ocr_api_key", _api_key)

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
        if not _ocr_key.strip():
            st.warning("在线 OCR 需要配置 API Key")
    elif not check_engine_available(ocr_engine):
        st.warning(f"`{ocr_engine}` 未安装，将回退到 paddleocr")
        ocr_engine = 'paddleocr'

    st.divider()
    st.header("LLM 评分设置")
    _ocr_api_config = {}
    llm_enabled = st.checkbox("启用 LLM 简答题评分",
                               value=bool(_api_key.strip()),
                               help="勾选后自动评分简答题，不勾选则需人工评分")
    llm_api_key = _api_key
    llm_base_url = _mcfg.get("base_url", DEFAULT_BASE_URL)
    llm_model = _mcfg.get("llm_model", DEFAULT_LLM_MODEL)
    llm_max_tokens = _mcfg.get("llm_max_tokens", 256)
    llm_temperature = _mcfg.get("llm_temperature", 0.3)
    if llm_enabled:
        if not llm_api_key.strip():
            st.warning("未配置 API Key，请在上方填写")
        llm_base_url = st.text_input(
            "Base URL",
            value=_mcfg.get("base_url", DEFAULT_BASE_URL),
            help="API 端点地址")
        llm_model = st.text_input(
            "模型", value=_mcfg.get("llm_model", DEFAULT_LLM_MODEL),
            help="模型 ID")
        if st.button("保存模型配置"):
            save_config({"base_url": llm_base_url,
                          "llm_model": llm_model,
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
    if llm_enabled and llm_api_key.strip():
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
    )
