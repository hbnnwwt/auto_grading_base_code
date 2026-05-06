"""Single-sheet recognition view."""
import glob
import os
import uuid

import cv2
import numpy as np
import streamlit as st

from modules.pipeline import (
    preprocess_and_analyze, extract_student_id,
    recognize_choices, recognize_judges, recognize_essay,
    get_essay_questions, LAYOUT,
)
from modules.preprocess import ImagePreprocessor
from modules.layout import LayoutAnalyzer
from modules.grading import GradingService
from modules.llm_essay_grader import LLMEssayGrader
from modules.student_id_recognizer import StudentIdRecognizer
from modules.choice_recognizer import ChoiceRecognizer
from modules.judge_recognizer import JudgeRecognizer
from modules import essay_recognizer as _essay_mod
from modules.marker import mark_and_save
from views.components import (
    load_image_from_bytes, imwrite,
    render_score_metrics, render_question_table,
)


def render_single(threshold, llm_enabled, llm_api_key, llm_base_url,
                   llm_model, llm_max_tokens, llm_temperature,
                   ocr_engine, ocr_api_config, PATHS):
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        page1_file = st.file_uploader(
            "第1页（正面）：学号 + 选择题",
            type=["png", "jpg", "jpeg", "bmp"], key="page1_upload",
        )
    with col_up2:
        page2_file = st.file_uploader(
            "第2页（反面）：判断题 + 简答题",
            type=["png", "jpg", "jpeg", "bmp"], key="page2_upload",
        )

    if page1_file is not None and page2_file is not None:
        st.divider()
        if st.button("开始识别", type="primary", width='stretch'):
            try:
                page1_img = load_image_from_bytes(page1_file.read())
                page2_img = load_image_from_bytes(page2_file.read())
            except Exception as e:
                st.error(f"图片读取失败: {e}")
                st.stop()

            preprocessor = ImagePreprocessor()
            analyzer = LayoutAnalyzer()

            # ── Step 1: 方向矫正 ──
            with st.expander("Step 1  方向矫正", expanded=True):
                p1_orig, p1_gray, p1_enhanced, p1_binary = preprocessor.process(page1_img)
                p1_correction = preprocessor.last_correction
                p1_before = preprocessor.before_correction
                p1_viz = preprocessor.detection_viz

                p2_orig, p2_gray, p2_enhanced, p2_binary = preprocessor.process(page2_img)
                p2_correction = preprocessor.last_correction
                p2_before = preprocessor.before_correction
                p2_viz = preprocessor.detection_viz

                # ── 1a: 角度检测 ──
                st.markdown("##### 1a  角度检测")
                st.caption(
                    "对原始图像做 OTSU 二值化 → 提取最大外轮廓 → "
                    "minAreaRect 拟合 → 计算倾斜角"
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**第1页**  检测角度: `{p1_correction:+.1f}°`")
                    st.image(p1_viz, width='stretch',
                             caption="绿色=轮廓  橙色=minAreaRect")
                with c2:
                    st.markdown(f"**第2页**  检测角度: `{p2_correction:+.1f}°`")
                    st.image(p2_viz, width='stretch',
                             caption="绿色=轮廓  橙色=minAreaRect")

                # ── 1b: 矫正效果 ──
                st.markdown("##### 1b  矫正效果")
                need_fix1 = abs(p1_correction) > 0.5
                need_fix2 = abs(p2_correction) > 0.5
                if not need_fix1 and not need_fix2:
                    st.success("两页均无需矫正，角度偏差 < 0.5°")
                c1, c2 = st.columns(2)
                with c1:
                    if need_fix1:
                        st.success(f"第1页已矫正 `{p1_correction:+.1f}°`")
                        bc, ac = st.columns(2)
                        with bc:
                            st.caption("矫正前")
                            st.image(p1_before, width='stretch')
                        with ac:
                            st.caption("矫正后")
                            st.image(p1_orig, width='stretch')
                    else:
                        st.info("第1页 方向正确")
                        st.image(p1_orig, width='stretch', caption="原图")
                with c2:
                    if need_fix2:
                        st.success(f"第2页已矫正 `{p2_correction:+.1f}°`")
                        bc, ac = st.columns(2)
                        with bc:
                            st.caption("矫正前")
                            st.image(p2_before, width='stretch')
                        with ac:
                            st.caption("矫正后")
                            st.image(p2_orig, width='stretch')
                    else:
                        st.info("第2页 方向正确")
                        st.image(p2_orig, width='stretch', caption="原图")

            # ── Step 2: 图像增强 ──
            with st.expander("Step 2  图像增强", expanded=True):
                st.caption(
                    "灰度化 → 高斯去噪 → CLAHE 对比度增强 → OTSU 二值化"
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**第1页**")
                    g1, e1, b1 = st.columns(3)
                    with g1:
                        st.caption("去噪灰度")
                        st.image(p1_gray, width='stretch')
                    with e1:
                        st.caption("CLAHE 增强")
                        st.image(p1_enhanced, width='stretch')
                    with b1:
                        st.caption("OTSU 二值化")
                        st.image(p1_binary, width='stretch')
                with c2:
                    st.markdown("**第2页**")
                    g2, e2, b2 = st.columns(3)
                    with g2:
                        st.caption("去噪灰度")
                        st.image(p2_gray, width='stretch')
                    with e2:
                        st.caption("CLAHE 增强")
                        st.image(p2_enhanced, width='stretch')
                    with b2:
                        st.caption("OTSU 二值化")
                        st.image(p2_binary, width='stretch')

            # ── Step 3: 版面分析 ──
            output_dir = PATHS['output_dir']
            os.makedirs(output_dir, exist_ok=True)

            regions1 = analyzer.analyze(p1_orig, p1_binary, page=1)
            p1_morph = analyzer.morph_image
            p1_layout_viz = analyzer.debug_image

            regions2 = analyzer.analyze(p2_orig, p2_binary, page=2)
            p2_morph = analyzer.morph_image
            p2_layout_viz = analyzer.debug_image

            # 保存各区域图像（临时文件名，识别后重命名）
            _saved = []
            uuid_str = uuid.uuid4().hex[:8]
            for code, roi_info, src_img in [
                ('student_id', regions1.get('student_id'), p1_orig),
                ('choice', regions1.get('choice'), p1_orig),
            ]:
                if roi_info:
                    x, y, w, h = roi_info
                    roi = src_img[y:y + h, x:x + w]
                    fpath = os.path.join(output_dir, f"tmp_{uuid_str}_{code}.png")
                    ok = imwrite(fpath, roi)
                    _saved.append(f"{code}: {ok} → {fpath}")
            for code, roi_info, src_img in [
                ('judge', regions2.get('judge'), p2_orig),
                ('essay', regions2.get('essay'), p2_orig),
            ]:
                if roi_info:
                    x, y, w, h = roi_info
                    roi = src_img[y:y + h, x:x + w]
                    fpath = os.path.join(output_dir, f"tmp_{uuid_str}_{code}.png")
                    ok = imwrite(fpath, roi)
                    _saved.append(f"{code}: {ok} → {fpath}")
            if _saved:
                st.caption("区域保存: " + " | ".join(_saved))
            else:
                st.warning(f"未保存任何区域。output_dir={output_dir}")

            with st.expander("Step 3  版面分析", expanded=True):
                names_cn = {
                    'student_id': '学号',
                    'choice': '选择题',
                    'judge': '判断题',
                    'essay': '简答题',
                }

                # ── 2a: 形态学闭运算 ──
                st.markdown("##### 2a  形态学闭运算")
                st.caption(
                    "反转二值图 → 形态学闭运算合并边框与内部内容 → "
                    "外轮廓检测 → 面积过滤 → 取最大2个区域"
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**第1页** 闭运算结果")
                    st.image(p1_morph, width='stretch')
                with c2:
                    st.markdown("**第2页** 闭运算结果")
                    st.image(p2_morph, width='stretch')

                # ── 2b: 区域定位 ──
                st.markdown("##### 2b  区域定位")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**第1页区域定位**")
                    st.image(p1_layout_viz, width='stretch')
                    for k in ['student_id', 'choice']:
                        r = regions1.get(k)
                        if r:
                            st.caption(f"{names_cn[k]}: x={r[0]} y={r[1]} w={r[2]} h={r[3]}")
                        else:
                            st.caption(f"{names_cn[k]}: 未检测到")
                with c2:
                    st.markdown("**第2页区域定位**")
                    st.image(p2_layout_viz, width='stretch')
                    for k in ['judge', 'essay']:
                        r = regions2.get(k)
                        if r:
                            st.caption(f"{names_cn[k]}: x={r[0]} y={r[1]} w={r[2]} h={r[3]}")
                        else:
                            st.caption(f"{names_cn[k]}: 未检测到")

            # ── Step 4: OMR 识别 ──
            with st.expander("Step 4  OMR 识别（填涂检测）", expanded=True):
                # ── 4a: 学号识别 ──
                st.markdown("##### 4a  学号识别")

                student_id = None
                if regions1.get('student_id'):
                    x, y, w, h = regions1['student_id']
                    sid_roi = p1_orig[y:y + h, x:x + w]
                    sid_rec = StudentIdRecognizer(
                        digit_count=10, threshold=threshold)
                    student_id, sid_viz, sid_details = \
                        sid_rec.recognize_with_viz(sid_roi)

                    # ── 4a-1 轮廓检测定位网格（中间结果可视化） ──
                    st.markdown("**4a-1  轮廓检测定位网格**")
                    st.caption(
                        "Canny → 膨胀 → findContours(RETR_TREE) → 去掉最大轮廓后在剩余轮廓中取第三大作为填涂网格"
                    )
                    if sid_rec.canny_image is not None:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("**① Canny 边缘**")
                            st.image(sid_rec.canny_image,
                                     width='stretch',
                                     caption="Canny(50, 150)")
                        with c2:
                            st.markdown("**② 面积 Top 3 轮廓**")
                            st.image(sid_rec.top10_contours_image,
                                     width='stretch',
                                     caption=f"检测到 {sid_rec.contour_count} 个轮廓")
                        with c3:
                            st.markdown("**③ 去#1取#3**")
                            st.image(sid_rec.third_selected_image,
                                     width='stretch',
                                     caption="红框=#1最大(去掉)  橙框=#3填涂网格")
                    if sid_rec.grid_bounds:
                        gx, gy, gw, gh = sid_rec.grid_bounds
                        st.caption(
                            f"网格边框: x={gx}  y={gy}  w={gw}  h={gh}")

                    # ── 4a-2 网格分割 ──
                    st.markdown("**4a-2  网格分割**")
                    st.caption(
                        "按网格边框等分为 N列×11行（1行序号+10行数字），"
                        "逐格统计黑色像素占比（像素值 < 128）得到填充率"
                    )
                    if sid_rec.grid_image is not None:
                        st.image(sid_rec.grid_image,
                                 width='stretch',
                                 caption="橙色=网格边框  灰线=单元格")

                    # ── 4a-3 网格扫描结果 ──
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.markdown(
                            f"**4a-3  网格扫描**  识别结果: `{student_id}`")
                        st.image(sid_viz, width='stretch',
                                 caption="绿框=选中  橙框=半填充")
                    with c2:
                        sid_rows = []
                        for i, d in enumerate(sid_details):
                            sid_rows.append({
                                '位': i + 1,
                                '识别': d['digit'],
                                '最高': (f"{d['best_row']}号 "
                                         f"({d['best_fill']:.0%})"),
                                '次高': (f"{d['runner_up_row']}号 "
                                         f"({d['runner_up_fill']:.0%})"),
                            })
                        st.dataframe(sid_rows, width='stretch',
                                     hide_index=True)

                    # ── 保存切分区域图像到 data/output ──
                    if student_id:
                        # 重命名临时文件为学号文件名
                        sid = student_id.replace('?', '_')
                        for code in ['student_id', 'choice', 'judge', 'essay']:
                            # 匹配 tmp_{uuid}_{code}.png 或 tmp_{code}.png
                            candidates = glob.glob(
                                os.path.join(output_dir, f"tmp_*_{code}.png"))
                            candidates += glob.glob(
                                os.path.join(output_dir, f"tmp_{code}.png"))
                            for tmp_path in candidates:
                                new_path = os.path.join(
                                    output_dir, f"{sid}_{code}.png")
                                os.replace(tmp_path, new_path)
                                break  # 只取第一个匹配
                        st.caption(f"区域图像已保存至 data/output/ → {sid}_*.png")
                else:
                    st.warning("未检测到学号区域")

                st.divider()

                # ── 4b: 选择题识别 ──
                choice_all_result = None
                judge_all_result = None
                _ch = LAYOUT['choice']
                st.markdown(f"##### 4b  选择题识别 ({_ch['question_start']}-{_ch['question_start'] + _ch['question_count'] - 1})")

                choice_answers = {}
                if regions1.get('choice'):
                    x, y, w, h = regions1['choice']
                    choice_roi = p1_orig[y:y + h, x:x + w]
                    choice_rec = ChoiceRecognizer(threshold=threshold)
                    choice_all_result = choice_rec.recognize_all_with_viz(
                        choice_roi,
                        question_count=_ch['question_count'],
                        question_start=_ch['question_start'],
                        fixed_grid=(_ch.get('rows', 5), _ch.get('cols', 4)))
                    choice_answers = choice_all_result['answers']

                    # 4b-1 网格切分
                    st.markdown("**4b-1  网格切分**")
                    st.caption(
                        f"将选择题区域均匀切分为 {_ch['rows']}行×{_ch['cols']}列"
                    )
                    st.image(choice_all_result['grid_viz'], width='stretch',
                             caption="数字=题号  灰线=网格分割")

                    st.divider()

                    # 4b-2 选项识别
                    st.markdown("**4b-2  选项识别**")
                    st.caption(
                        "逐格做连通域分析 → 各选项区最大填充率 → 选最高"
                    )
                    viz_cells = [r['cell_viz'] for r in choice_all_result['cell_results']]
                    if viz_cells:
                        # 统一单元格图像尺寸（找最大宽高后 padding）
                        max_h = max(v.shape[0] for v in viz_cells)
                        max_w = max(v.shape[1] for v in viz_cells)
                        padded = []
                        for v in viz_cells:
                            if v.shape[0] < max_h or v.shape[1] < max_w:
                                pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                                pad[:v.shape[0], :v.shape[1]] = v
                                padded.append(pad)
                            else:
                                padded.append(v)
                        full_viz = np.vstack(padded)
                        choice_table = []
                        for r in choice_all_result['cell_results']:
                            fills = r['zone_fills']
                            choice_table.append({
                                '题号': r['question'],
                                'A': f"{fills[0]:.0%}",
                                'B': f"{fills[1]:.0%}",
                                'C': f"{fills[2]:.0%}",
                                'D': f"{fills[3]:.0%}",
                                '识别': r['answer'] or '-',
                            })
                        c1, c2 = st.columns([3, 2])
                        with c1:
                            st.image(full_viz, width='stretch',
                                     caption="绿框=选中  橙框=半填充  灰线=选项分割")
                        with c2:
                            st.dataframe(choice_table,
                                         width='stretch',
                                         hide_index=True)
                        # ── 备查：选择题识别结果汇总 ──
                        st.markdown("**选择题识别结果（备查）**")
                        choice_summary = [f"第{r['question']}题: {r['answer'] or '未识别'}"
                                           for r in choice_all_result['cell_results']]
                        st.markdown("  |  ".join(choice_summary))
                else:
                    st.warning("未检测到选择题区域")

                st.divider()

                # ── 4c: 判断题识别 ──
                _ju = LAYOUT['judge']
                st.markdown(f"##### 4c  判断题识别 ({_ju['question_start']}-{_ju['question_start'] + _ju['question_count'] - 1})")

                judge_answers = {}
                if regions2.get('judge'):
                    x, y, w, h = regions2['judge']
                    judge_roi = p2_orig[y:y + h, x:x + w]
                    judge_rec = JudgeRecognizer(threshold=threshold)
                    judge_all_result = judge_rec.recognize_all_with_viz(
                        judge_roi,
                        question_count=_ju['question_count'],
                        question_start=_ju['question_start'],
                        rows_n=_ju.get('rows', 3), cols_n=_ju.get('cols', 4))
                    judge_answers = judge_all_result['answers']

                    # 4c-1 网格切分
                    st.markdown("**4c-1  网格切分**")
                    st.caption(
                        f"将判断题区域均匀切分为 {_ju['rows']}行×{_ju['cols']}列"
                    )
                    st.image(judge_all_result['grid_viz'], width='stretch',
                             caption="数字=题号  灰线=网格分割")

                    st.divider()

                    # 4c-2 选项识别
                    st.markdown("**4c-2  选项识别**")
                    st.caption(
                        "逐格做连通域分析 → T/F 区域最大填充率 → 选最高"
                    )
                    viz_cells = [r['cell_viz'] for r in judge_all_result['cell_results']]
                    if viz_cells:
                        # 统一单元格图像尺寸（找最大宽高后 padding）
                        max_h = max(v.shape[0] for v in viz_cells)
                        max_w = max(v.shape[1] for v in viz_cells)
                        padded = []
                        for v in viz_cells:
                            if v.shape[0] < max_h or v.shape[1] < max_w:
                                pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                                pad[:v.shape[0], :v.shape[1]] = v
                                padded.append(pad)
                            else:
                                padded.append(v)
                        full_viz = np.vstack(padded)
                        judge_table = []
                        for r in judge_all_result['cell_results']:
                            fills = r['zone_fills']
                            judge_table.append({
                                '题号': r['question'],
                                'T(对)': f"{fills[0]:.0%}",
                                'F(错)': f"{fills[1]:.0%}",
                                '识别': r['answer'] or '-',
                            })
                        c1, c2 = st.columns([3, 2])
                        with c1:
                            st.image(full_viz, width='stretch',
                                     caption="绿框=选中  橙框=半填充  灰线=选项分割")
                        with c2:
                            st.dataframe(judge_table,
                                         width='stretch',
                                         hide_index=True)
                        # ── 备查：判断题识别结果汇总 ──
                        st.markdown("**判断题识别结果（备查）**")
                        judge_summary = [f"第{r['question']}题: {r['answer'] or '未识别'}"
                                          for r in judge_all_result['cell_results']]
                        st.markdown("  |  ".join(judge_summary))
                else:
                    st.warning("未检测到判断题区域")

            # ── Step 5: OCR 识别 ──
            with st.expander("Step 5  OCR 识别（简答题）", expanded=True):
                if regions2.get('essay'):
                    ex, ey, ew, eh = regions2['essay']
                    essay_roi = p2_orig[ey:ey + eh, ex:ex + ew]
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown("**简答题区域原图**")
                        st.image(essay_roi, width='stretch')
                    with c2:
                        essay_text = recognize_essay(p2_orig, regions2,
                                                     ocr_engine,
                                                     api_config=ocr_api_config)
                        st.markdown("**第31题 OCR 结果**")
                        _display_text = essay_text or "(空)"
                        st.text_area("识别文本", _display_text,
                                     height=max(150, len(_display_text.split('\n')) * 30))
                        if not essay_text and ocr_engine == 'online':
                            if _essay_mod.last_error:
                                st.error(f"在线 OCR 错误: {_essay_mod.last_error}")
                else:
                    essay_text = ""
                    st.warning("未检测到简答题区域")

            # ── Step 6: 评分 ──
            svc = None
            if os.path.exists(PATHS['answer_key']):
                try:
                    svc = GradingService.from_xlsx(PATHS['answer_key'])
                except Exception as e:
                    st.error(f"参考答案文件加载失败: {e}")
            if svc and llm_enabled:
                svc.essay_grader = LLMEssayGrader(
                    api_key=llm_api_key, base_url=llm_base_url,
                    model=llm_model,
                    max_tokens=llm_max_tokens,
                    temperature=llm_temperature)
            with st.expander("Step 6  评分", expanded=True):
                if not svc:
                    st.warning("参考答案未加载，无法评分。")
                    st.stop()

                recognized = {
                    "choice": choice_answers,
                    "judge": judge_answers,
                    "essay": {q: essay_text for q in get_essay_questions(svc.answer_key)} if essay_text else {},
                }
                try:
                    result = svc.grade(recognized)
                except Exception as e:
                    st.error(f"评分失败: {e}")
                    result = None

                render_score_metrics(result, svc)
                st.divider()

                st.markdown("**选择题逐题对比**")
                _ch = LAYOUT['choice']
                rows = render_question_table(_ch['question_start'], _ch['question_start'] + _ch['question_count'] - 1, choice_answers, result["choice"])
                st.dataframe(rows, width='stretch', hide_index=True)

                st.markdown("**判断题逐题对比**")
                _ju = LAYOUT['judge']
                rows = render_question_table(_ju['question_start'], _ju['question_start'] + _ju['question_count'] - 1, judge_answers, result["judge"])
                st.dataframe(rows, width='stretch', hide_index=True)

                st.divider()
                st.markdown("**简答题评分**")
                if llm_enabled:
                    essay_detail = result.get("essay_detail", {})
                    for q, d in essay_detail.items():
                        st.markdown(f"第{q}题: **{d['score']}/{d['max_score']}分**")
                        st.caption(f"反馈: {d['feedback']}")
                else:
                    st.info(f"未启用 LLM 评分，简答题需人工评分（{svc.essay_max_score}分）")

                # ── 错题标注 ──
                processed_dir = PATHS['processed_dir']
                _choice_cells = (choice_all_result.get('cell_results', [])
                                 if choice_all_result else [])
                _judge_cells = (judge_all_result.get('cell_results', [])
                                if judge_all_result else [])
                try:
                    _, _, marked_p1, marked_p2 = mark_and_save(
                        student_id, p1_orig, p2_orig,
                        regions1, regions2,
                        _choice_cells, _judge_cells, result,
                        choice_max=len(svc.answer_key.get('choice', {})) * svc.choice_score,
                        judge_max=len(svc.answer_key.get('judge', {})) * svc.judge_score,
                        essay_max=len(svc.answer_key.get('essay', {})) * svc.essay_max_score,
                        output_dir=processed_dir)
                except Exception as e:
                    st.warning(f"错题标注失败: {e}")
                    marked_p1, marked_p2 = p1_orig, p2_orig
                st.divider()
                st.markdown("**批改标注图（红色 × = 错题）**")
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("第1页（选择题）")
                    st.image(cv2.cvtColor(marked_p1, cv2.COLOR_BGR2RGB),
                             width='stretch')
                with c2:
                    st.caption("第2页（判断题）")
                    st.image(cv2.cvtColor(marked_p2, cv2.COLOR_BGR2RGB),
                             width='stretch')

            # ── Step 7: 评分报告 ──
            with st.expander("Step 7  评分报告", expanded=True):
                report = svc.generate_report(result)
                st.code(report, language=None)

            # P35: 清理残留临时文件
            for tmp_file in glob.glob(os.path.join(output_dir, "tmp_*.png")):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass
    else:
        st.info("请上传答题卡的**两页**图片开始识别（支持 PNG、JPG、BMP）")
