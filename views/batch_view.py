"""批量阅卷 view。"""
import glob
import io
import json as _json
import os
import time
import uuid

import openpyxl
import streamlit as st
import cv2

from modules.pipeline import (
    preprocess_and_analyze, extract_student_id,
    recognize_choices, recognize_judges, recognize_essay,
    get_essay_questions, LAYOUT,
)
from modules.preprocess import ImagePreprocessor
from modules.layout import LayoutAnalyzer
from modules.grading import GradingService
from modules.llm_essay_grader import LLMEssayGrader
from modules.marker import mark_and_save
from modules.defaults import IMAGE_EXTS, natural_sort_key
from views.components import (
    load_image, render_score_metrics, render_question_table,
    check_and_save_rejected,
)


# ── Session State Keys ──
_PHASE = "batch_phase"
_PAIRS = "batch_pairs"
_RESULTS = "batch_results"
_CURRENT = "batch_current_idx"
_FOLDER = "batch_folder_path"
_OUTPUT_DIR = "batch_output_dir"
_TIMESTAMP = "batch_timestamp"
_SVC = "batch_svc"
_PREPROCESSOR = "batch_preprocessor"
_ANALYZER = "batch_analyzer"
_PROCESSED_DIR = "batch_processed_dir"
_CKPT_PATH = "batch_ckpt_path"


def _reset_state(keep_folder=False):
    """重置批量处理状态。"""
    keys = [_PHASE, _PAIRS, _RESULTS, _CURRENT,
            _OUTPUT_DIR, _TIMESTAMP, _SVC, _PREPROCESSOR, _ANALYZER,
            _PROCESSED_DIR, _CKPT_PATH]
    if not keep_folder:
        keys.append(_FOLDER)
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state[_PHASE] = 'idle'


def _collect_images(folder_path):
    """收集文件夹中的图像文件，按文件名自然排序。"""
    all_files = glob.glob(os.path.join(folder_path, "*"))
    all_files.sort(key=natural_sort_key)
    return [f for f in all_files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]


def _build_pairs(img_files):
    """按奇偶配对图像文件。"""
    return [(img_files[2 * i], img_files[2 * i + 1])
            for i in range(len(img_files) // 2)]


def _load_svc(PATHS, llm_enabled, llm_api_key, llm_base_url,
              llm_model, llm_max_tokens, llm_temperature):
    """加载评分服务。"""
    svc = None
    if os.path.exists(PATHS['answer_key']):
        try:
            svc = GradingService.from_xlsx(PATHS['answer_key'])
        except Exception as e:
            st.warning(f"参考答案文件加载失败: {e}")

    _has_key = bool(llm_api_key) if isinstance(llm_api_key, list) else bool(llm_api_key.strip())
    if svc and llm_enabled and _has_key:
        svc.essay_grader = LLMEssayGrader(
            api_key=llm_api_key, base_url=llm_base_url,
            model=llm_model,
            max_tokens=llm_max_tokens,
            temperature=llm_temperature)

    return svc


def _retry_essay_grading(r, idx, svc, threshold=0.06):
    """对单个学生的简答题重新调用 LLM 评分，更新 session state 和标注图。"""
    if not svc or not svc.essay_grader:
        st.warning("评分服务未初始化，无法重试")
        return

    result = r.get("result")
    if not result:
        st.warning("无评分结果，无法重试")
        return

    essay_text = r.get("essay", "")
    if not essay_text:
        st.warning("无简答题文本，无法重试")
        return

    with st.spinner("正在重新评分简答题..."):
        new_essay_detail = {}
        new_essay_score = 0
        for q, ref_text in svc.answer_key.get('essay', {}).items():
            s, mx, fb = svc.essay_grader.score(q, ref_text, essay_text,
                                                svc.essay_max_score)
            new_essay_detail[q] = {'score': s, 'max_score': mx, 'feedback': fb}
            new_essay_score += s

    new_result = dict(result)
    new_result['essay_detail'] = new_essay_detail
    new_result['essay'] = {q: essay_text for q in svc.answer_key.get('essay', {})} if essay_text else {}
    new_result['essay_total'] = new_essay_score
    new_result['total'] = (new_result['choice_total'] + new_result['judge_total']
                           + new_essay_score)

    try:
        report = svc.generate_report(new_result)
    except Exception as e:
        report = f"报告生成失败: {e}"

    # ── 重新生成标注图 ──
    marked_p1_path = r.get("marked_p1_path")
    marked_p2_path = r.get("marked_p2_path")
    page1_path = r.get("page1_full_path")
    page2_path = r.get("page2_full_path")

    if page1_path and page2_path and marked_p1_path and marked_p2_path:
        try:
            preprocessor = st.session_state.get(_PREPROCESSOR) or ImagePreprocessor()
            analyzer = st.session_state.get(_ANALYZER) or LayoutAnalyzer()

            page1 = load_image(page1_path)
            page2 = load_image(page2_path)

            regions1, page1 = preprocess_and_analyze(page1, 1, preprocessor, analyzer)
            regions2, page2 = preprocess_and_analyze(page2, 2, preprocessor, analyzer)

            choice_answers, choice_cells = recognize_choices(
                page1, regions1, threshold, return_details=True)
            judge_answers, judge_cells = recognize_judges(
                page2, regions2, threshold, return_details=True)

            choice_max = len(svc.answer_key.get('choice', {})) * svc.choice_score
            judge_max = len(svc.answer_key.get('judge', {})) * svc.judge_score
            essay_max = len(svc.answer_key.get('essay', {})) * svc.essay_max_score

            output_dir = os.path.dirname(marked_p1_path)
            p1_path, p2_path, _, _ = mark_and_save(
                r.get("student_id"), page1, page2,
                regions1, regions2,
                choice_cells, judge_cells, new_result,
                choice_max=choice_max, judge_max=judge_max,
                essay_max=essay_max, output_dir=output_dir)

            marked_p1_path = p1_path
            marked_p2_path = p2_path
        except Exception as e:
            st.warning(f"标注图更新失败: {e}")

    results = st.session_state.get(_RESULTS, [])
    if 0 <= idx < len(results):
        results[idx]['result'] = new_result
        results[idx]['report'] = report
        results[idx]['marked_p1_path'] = marked_p1_path
        results[idx]['marked_p2_path'] = marked_p2_path
        st.session_state[_RESULTS] = results
        st.success("简答题评分已更新")
    else:
        st.warning("结果索引已失效，无法更新")


def _render_student_result(r, idx, svc, llm_enabled, threshold=0.06):
    """渲染单份试卷的结果（在 expander 中）。"""
    if "error" in r:
        with st.expander(f"[失败] 学生 {idx + 1}: {r['page1']}", expanded=False):
            st.error(r["error"])
        return

    if r.get("rejected"):
        with st.expander(f"[废卷] 学生 {idx + 1}: {r['student_id']}", expanded=False):
            st.warning(r.get("report", "学号未填涂或整页无任何填涂"))
        return

    res = r["result"]
    if res and svc:
        total = res["total"]
        pct = total / svc.max_total * 100 if svc.max_total > 0 else 0
        cs = res["choice_total"]
        js = res["judge_total"]
        es = res.get("essay_total", 0)
        choice_max = len(svc.answer_key.get('choice', {})) * svc.choice_score
        judge_max = len(svc.answer_key.get('judge', {})) * svc.judge_score
        essay_max = len(svc.answer_key.get('essay', {})) * svc.essay_max_score
        label = (f"学生 {idx + 1}: {r['student_id']}  "
                 f"总分 {total}/{svc.max_total} ({pct:.0f}%)  "
                 f"选择 {cs}/{choice_max}  判断 {js}/{judge_max}  简答 {es}/{essay_max}")
    else:
        label = f"学生 {idx + 1}: {r['student_id']}"

    # 识别过程中，最新完成的默认展开；完成后全部折叠
    results = st.session_state.get(_RESULTS, [])
    is_latest = (idx == len(results) - 1) and st.session_state.get(_PHASE) == 'processing'

    with st.expander(label, expanded=is_latest):
        if res and svc:
            render_score_metrics(res, svc)

        if r.get("marked_p1_path") and r.get("marked_p2_path"):
            st.divider()
            st.markdown("**批改标注图（红色 × = 错题）**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("第1页（选择题）")
                st.image(r["marked_p1_path"], width='stretch')
            with c2:
                st.caption("第2页（判断题）")
                st.image(r["marked_p2_path"], width='stretch')

        dt1, dt2, dt3 = st.tabs(["选择题", "判断题", "简答题"])

        with dt1:
            _ch = LAYOUT['choice']
            rows = render_question_table(
                _ch['question_start'], _ch['question_start'] + _ch['question_count'] - 1,
                r["choice"], res["choice"] if res else None)
            st.dataframe(rows, width='stretch', hide_index=True)

        with dt2:
            _ju = LAYOUT['judge']
            rows = render_question_table(
                _ju['question_start'], _ju['question_start'] + _ju['question_count'] - 1,
                r["judge"], res["judge"] if res else None)
            st.dataframe(rows, width='stretch', hide_index=True)

        with dt3:
            st.text_area("OCR 文本", r.get("essay", "(空)"),
                        height=100, key=f"essay_batch_{idx}")
            if llm_enabled and res:
                essay_detail = res.get("essay_detail", {})
                _LL_FAILURE_MARKERS = (
                    "LLM 调用失败", "LLM 未配置", "LLM 返回为空",
                    "LLM 返回无法解析")
                has_llm_failure = any(
                    any(m in str(d.get("feedback", "")) for m in _LL_FAILURE_MARKERS)
                    for d in essay_detail.values()
                )
                for q, d in essay_detail.items():
                    st.markdown(f"**第{q}题: {d['score']}/{d['max_score']}分**")
                    st.caption(f"反馈: {d['feedback']}")
                if has_llm_failure and svc and svc.essay_grader:
                    if st.button("重试 LLM 评分", key=f"retry_llm_{idx}"):
                        _retry_essay_grading(r, idx, svc, threshold)
                        st.rerun()
            else:
                st.info(f"未启用 LLM 评分，需人工评分（{svc.essay_max_score if svc else '?'}分）")

        if r.get("report"):
            with st.expander("详细报告"):
                st.code(r["report"], language=None)


def _render_summary(results, svc):
    """渲染成绩汇总表格。"""
    st.subheader("成绩汇总")
    summary = []
    for r in results:
        if "error" in r:
            summary.append({
                "学号": r["student_id"], "总分": "ERROR",
                "选择题": "-", "判断题": "-", "简答题": "-",
            })
        elif r.get("rejected"):
            summary.append({
                "学号": r["student_id"], "总分": "【废卷】",
                "选择题": "-", "判断题": "-", "简答题": "-",
            })
        else:
            res = r["result"]
            if res and svc:
                cs = res["choice_total"]
                js = res["judge_total"]
                es = res.get("essay_total", 0)
                choice_max = len(svc.answer_key.get('choice', {})) * svc.choice_score
                judge_max = len(svc.answer_key.get('judge', {})) * svc.judge_score
                essay_max = len(svc.answer_key.get('essay', {})) * svc.essay_max_score
                summary.append({
                    "学号": r["student_id"],
                    "总分": f"{res['total']}/{svc.max_total}",
                    "选择题": f"{cs}/{choice_max}",
                    "判断题": f"{js}/{judge_max}",
                    "简答题": f"{es}/{essay_max}",
                })
            else:
                summary.append({
                    "学号": r["student_id"], "总分": "-",
                    "选择题": "-", "判断题": "-",
                    "简答题": r.get("essay", ""),
                })
    st.dataframe(summary, width='stretch', hide_index=True)
    return summary


def _build_xlsx(results, svc):
    """构建 xlsx 下载数据。"""
    buf = io.BytesIO()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"
    headers = ["学号", "总分", "选择题得分", "判断题得分", "简答题得分"]
    _ch = LAYOUT['choice']
    _ju = LAYOUT['judge']
    choice_end = _ch['question_start'] + _ch['question_count'] - 1
    judge_end = _ju['question_start'] + _ju['question_count'] - 1
    for q in range(_ch['question_start'], judge_end + 1):
        headers.append(f"Q{q}")
    headers += ["简答题内容", "简答题反馈"]
    ws.append(headers)

    for r in results:
        if "error" in r or r.get("rejected"):
            continue
        res = r["result"]
        cs = res["choice_total"] if res else 0
        js = res["judge_total"] if res else 0
        es = res.get("essay_total", 0) if res else 0
        row = [r["student_id"], cs + js + es, cs, js, es]
        for q in range(_ch['question_start'], judge_end + 1):
            if q <= choice_end:
                row.append(r["choice"].get(q, "-"))
            else:
                row.append(r["judge"].get(q, "-"))
        essay_text = r.get("essay", "")
        essay_feedback = ""
        if res and res.get("essay_detail"):
            for q, d in res["essay_detail"].items():
                essay_feedback = d.get("feedback", "")
        row += [essay_text, essay_feedback]
        ws.append(row)

    wb.save(buf)
    buf.seek(0)
    return buf


def render_batch(threshold, llm_enabled, llm_api_key, llm_base_url,
                 llm_model, llm_max_tokens, llm_temperature,
                 ocr_engine, ocr_api_config, PATHS,
                 choice_baseline=None, judge_baseline=None,
                 choice_zone_bounds=None, judge_zone_bounds=None):
    """批量阅卷视图。

    流程：idle（选择文件夹）→ paired（确认配对）→ processing（增量识别）→ done（结果汇总）
    """
    st.subheader("批量阅卷")

    # ── 状态初始化 ──
    if _PHASE not in st.session_state:
        st.session_state[_PHASE] = 'idle'

    phase = st.session_state[_PHASE]

    # ═══════════════════════════════════════════════════════════════
    # Phase: idle — 选择文件夹
    # ═══════════════════════════════════════════════════════════════
    if phase == 'idle':
        st.markdown(
            "图片按文件名排序配对：奇数位(第1、3、…)为**第1页**(学号+选择题)，"
            "偶数位(第2、4、…)为**第2页**(判断题+简答题)。"
        )

        default_path = PATHS['default_folder'] if os.path.isdir(PATHS['default_folder']) else ""
        folder_path = st.text_input(
            "图片文件夹路径",
            value=st.session_state.get(_FOLDER, default_path),
            help="答题卡图片所在文件夹的绝对或相对路径",
            key="batch_folder_input"
        )

        if folder_path:
            st.session_state[_FOLDER] = folder_path

        col1, _ = st.columns([1, 4])
        with col1:
            load_btn = st.button("加载并配对", type="primary")

        # 显示文件计数（预览）
        if folder_path and os.path.isdir(folder_path):
            img_files = _collect_images(folder_path)
            pair_count = len(img_files) // 2
            remainder = len(img_files) % 2

            info_text = f"找到 {len(img_files)} 张图片 = {pair_count} 份试卷"
            if remainder:
                info_text += "（奇数，最后 1 张将被忽略）"
            st.info(info_text)

            if load_btn:
                if len(img_files) < 2:
                    st.error("至少需要 2 张图片（1 对试卷）")
                    st.stop()

                pairs = _build_pairs(img_files)
                st.session_state[_PAIRS] = pairs
                st.session_state[_PHASE] = 'paired'
                # 保存路径配置供后续阶段使用
                st.session_state[_PROCESSED_DIR] = PATHS['processed_dir']
                st.session_state[_CKPT_PATH] = PATHS['batch_checkpoint']
                st.rerun()
        elif folder_path and load_btn:
            st.error("请输入有效的文件夹路径")

    # ═══════════════════════════════════════════════════════════════
    # Phase: paired — 确认配对
    # ═══════════════════════════════════════════════════════════════
    elif phase == 'paired':
        pairs = st.session_state.get(_PAIRS, [])

        if not pairs:
            st.error("配对数据丢失，请重新加载")
            _reset_state()
            st.rerun()

        st.success(f"共配对 {len(pairs)} 份试卷")

        # 配对表格
        table_data = []
        for i, (p1, p2) in enumerate(pairs, 1):
            table_data.append({
                "序号": i,
                "第1页（学号+选择题）": os.path.basename(p1),
                "第2页（判断题+简答题）": os.path.basename(p2),
            })
        st.dataframe(table_data, hide_index=True, width='stretch')

        # 操作按钮
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            start_btn = st.button("开始识别", type="primary")
        with col2:
            reset_btn = st.button("重新选择")

        if start_btn:
            # 初始化处理状态
            st.session_state[_RESULTS] = []
            st.session_state[_CURRENT] = 0

            # 创建时间戳子文件夹
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(st.session_state[_PROCESSED_DIR], timestamp)
            os.makedirs(output_dir, exist_ok=True)
            st.session_state[_OUTPUT_DIR] = output_dir
            st.session_state[_TIMESTAMP] = timestamp

            # 加载评分服务
            svc = _load_svc(PATHS, llm_enabled, llm_api_key, llm_base_url,
                           llm_model, llm_max_tokens, llm_temperature)
            st.session_state[_SVC] = svc

            # 初始化预处理器
            st.session_state[_PREPROCESSOR] = ImagePreprocessor()
            st.session_state[_ANALYZER] = LayoutAnalyzer()

            st.session_state[_PHASE] = 'processing'
            st.rerun()

        if reset_btn:
            _reset_state(keep_folder=True)
            st.rerun()

    # ═══════════════════════════════════════════════════════════════
    # Phase: processing — 增量识别
    # ═══════════════════════════════════════════════════════════════
    elif phase == 'processing':
        pairs = st.session_state[_PAIRS]
        current = st.session_state[_CURRENT]
        results = st.session_state[_RESULTS]
        total_pairs = len(pairs)
        svc = st.session_state.get(_SVC)
        output_dir = st.session_state.get(_OUTPUT_DIR)

        # ── 显示总体进度 ──
        st.progress(current / total_pairs if total_pairs > 0 else 0,
                    text=f"处理进度: {current}/{total_pairs}")

        # ── 显示已处理的结果 ──
        if results:
            st.subheader("识别结果")
            for idx, r in enumerate(results):
                _render_student_result(r, idx, svc, llm_enabled, threshold)

        # ── 处理下一份 ──
        if current < total_pairs:
            page1_path, page2_path = pairs[current]

            with st.status(
                f"正在识别: {os.path.basename(page1_path)} + {os.path.basename(page2_path)}",
                expanded=True
            ):
                st.write(f"第 {current + 1}/{total_pairs} 份")

                preprocessor = st.session_state[_PREPROCESSOR]
                analyzer = st.session_state[_ANALYZER]

                try:
                    # 加载图像
                    page1 = load_image(page1_path)
                    page2 = load_image(page2_path)

                    # 处理第1页
                    st.write("处理第1页（学号+选择题）...")
                    regions1, page1 = preprocess_and_analyze(page1, 1, preprocessor, analyzer)
                    student_id = extract_student_id(page1, regions1, threshold=threshold)
                    choice_answers, choice_cells = recognize_choices(
                        page1, regions1, threshold, return_details=True,
                        blank_baseline=choice_baseline,
                        zone_bounds=choice_zone_bounds)

                    # 废卷检测
                    uuid_str = uuid.uuid4().hex[:8]
                    reject_result = check_and_save_rejected(
                        student_id, page1, page2,
                        uuid_str, output_dir)

                    if reject_result:
                        reject_result["page1"] = os.path.basename(page1_path)
                        reject_result["page2"] = os.path.basename(page2_path)
                        results.append(reject_result)
                        st.write("检测结果：废卷（学号未填涂），已跳过")
                    else:
                        # 处理第2页
                        st.write("处理第2页（判断题+简答题）...")
                        regions2, page2 = preprocess_and_analyze(page2, 2, preprocessor, analyzer)
                        judge_answers, judge_cells = recognize_judges(
                            page2, regions2, threshold, return_details=True,
                            blank_baseline=judge_baseline,
                            zone_bounds=judge_zone_bounds)
                        essay_text = recognize_essay(page2, regions2, ocr_engine,
                                                    api_config=ocr_api_config)

                        recognized = {
                            "choice": choice_answers,
                            "judge": judge_answers,
                            "essay": {q: essay_text for q in get_essay_questions(svc.answer_key)} if essay_text else {},
                        }

                        # 评分
                        result = None
                        report = ""
                        marked_p1_path = None
                        marked_p2_path = None

                        if svc:
                            result = svc.grade(recognized)
                            report = svc.generate_report(result)

                            # 生成标注图
                            try:
                                choice_max = len(svc.answer_key.get('choice', {})) * svc.choice_score
                                judge_max = len(svc.answer_key.get('judge', {})) * svc.judge_score
                                essay_max = len(svc.answer_key.get('essay', {})) * svc.essay_max_score

                                p1_path, p2_path, marked_p1, marked_p2 = mark_and_save(
                                    student_id, page1, page2,
                                    regions1, regions2,
                                    choice_cells, judge_cells, result,
                                    choice_max=choice_max, judge_max=judge_max,
                                    essay_max=essay_max, output_dir=output_dir)
                                marked_p1_path = p1_path
                                marked_p2_path = p2_path
                                st.write("批改标注图已生成")
                            except Exception as e:
                                st.warning(f"错题标注失败: {e}")

                        results.append({
                            "student_id": student_id or "??????????",
                            "page1": os.path.basename(page1_path),
                            "page2": os.path.basename(page2_path),
                            "page1_full_path": page1_path,
                            "page2_full_path": page2_path,
                            "choice": choice_answers,
                            "judge": judge_answers,
                            "essay": essay_text,
                            "result": result,
                            "report": report,
                            "marked_p1_path": marked_p1_path,
                            "marked_p2_path": marked_p2_path,
                        })
                        st.write(f"学号: {student_id or '??????????'} | 识别完成")

                except Exception as e:
                    results.append({
                        "student_id": "ERROR",
                        "page1": os.path.basename(page1_path),
                        "page2": os.path.basename(page2_path),
                        "error": str(e),
                    })
                    st.error(f"处理失败: {e}")

            # 保存状态
            st.session_state[_RESULTS] = results
            st.session_state[_CURRENT] = current + 1

            # 增量保存 checkpoint
            ckpt_path = st.session_state.get(_CKPT_PATH)
            if ckpt_path:
                with open(ckpt_path, 'w', encoding='utf-8') as f:
                    _json.dump(results, f, ensure_ascii=False, default=str)

            # 自动继续下一份
            st.rerun()

        else:
            # 全部完成
            st.session_state[_PHASE] = 'done'
            st.rerun()

    # ═══════════════════════════════════════════════════════════════
    # Phase: done — 结果汇总
    # ═══════════════════════════════════════════════════════════════
    elif phase == 'done':
        results = st.session_state.get(_RESULTS, [])
        svc = st.session_state.get(_SVC)
        output_dir = st.session_state.get(_OUTPUT_DIR)

        if not results:
            st.warning("未产生任何结果")
            if st.button("重新开始"):
                _reset_state()
                st.rerun()
            st.stop()

        st.success(f"已完成 {len(results)} 份试卷的识别!")

        # ── 统计卡片 ──
        if svc:
            scores = []
            error_count = 0
            for r in results:
                if "error" in r:
                    error_count += 1
                elif r["result"]:
                    scores.append(r["result"]["total"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("总份数", len(results))
            c2.metric("成功", len(results) - error_count)
            c3.metric("失败", error_count)
            if scores:
                avg = sum(scores) / len(scores)
                c4.metric("平均分", f"{avg:.1f}", f"满分 {svc.max_total}")
            st.divider()

        # ── 成绩汇总表 ──
        _render_summary(results, svc)
        st.divider()

        # ── 逐份详情 ──
        st.subheader("逐份详情")
        for idx, r in enumerate(results):
            _render_student_result(r, idx, svc, llm_enabled, threshold)

        st.divider()

        # ── 下载按钮 ──
        if svc:
            buf = _build_xlsx(results, svc)
            st.download_button(
                "导出成绩 (xlsx)", buf,
                file_name="grading_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # ── 输出目录信息 ──
        if output_dir:
            st.info(f"标注图已保存至: {output_dir}")

        # ── 重新开始 ──
        if st.button("重新开始"):
            _reset_state()
            st.rerun()
