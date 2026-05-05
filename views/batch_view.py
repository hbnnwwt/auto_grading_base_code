"""批量阅卷 view。"""
import glob
import io
import json as _json
import os

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
from modules.defaults import IMAGE_EXTS
from views.components import (
    load_image, render_score_metrics, render_question_table,
)


def render_batch(threshold, llm_enabled, llm_api_key, llm_base_url,
                 llm_model, llm_max_tokens, llm_temperature,
                 ocr_engine, ocr_api_config, PATHS):
    """批量阅卷视图。

    参数:
        threshold: 填涂阈值
        llm_enabled: 是否启用 LLM 评分
        llm_api_key: LLM API 密钥
        llm_base_url: LLM API 基础 URL
        llm_model: LLM 模型名称
        llm_max_tokens: LLM 最大 token 数
        llm_temperature: LLM 温度参数
        ocr_engine: OCR 引擎类型
        ocr_api_config: OCR API 配置
        PATHS: 路径配置字典
    """
    st.subheader("批量阅卷")
    st.markdown(
        "图片按文件名排序配对：奇数位(第1、3、…)为**第1页**(学号+选择题)，"
        "偶数位(第2、4、…)为**第2页**(判断题+简答题)。"
    )

    default_path = PATHS['default_folder'] if os.path.isdir(PATHS['default_folder']) else ""
    folder_path = st.text_input(
        "图片文件夹路径",
        value=default_path,
        help="答题卡图片所在文件夹的绝对或相对路径",
    )

    img_count = 0
    if folder_path and os.path.isdir(folder_path):
        all_files = sorted(glob.glob(os.path.join(folder_path, "*")))
        img_files = [f for f in all_files
                     if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        img_count = len(img_files)
        st.info(f"找到 {img_count} 张图片 = {img_count // 2} 份试卷")
    elif folder_path:
        st.warning("文件夹不存在")

    if st.button("开始批量识别", type="primary",
                 disabled=not folder_path or not os.path.isdir(folder_path)
                 or img_count < 2):
        if img_count < 2:
            st.error("至少需要 2 张图片（1 对试卷）")
            st.stop()

        # 加载评分服务
        svc = None
        if os.path.exists(PATHS['answer_key']):
            try:
                svc = GradingService.from_xlsx(PATHS['answer_key'])
            except Exception as e:
                st.warning(f"参考答案文件加载失败: {e}")

        if svc and llm_enabled:
            svc.essay_grader = LLMEssayGrader(
                api_key=llm_api_key, base_url=llm_base_url,
                model=llm_model,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature)

        all_files_sorted = sorted(glob.glob(os.path.join(folder_path, "*")))
        images = [f for f in all_files_sorted
                  if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        results = []
        pairs = len(images) // 2
        if len(images) % 2 != 0:
            st.warning(f"共 {len(images)} 张图片（奇数），最后 1 张将被忽略")
        preprocessor = ImagePreprocessor()
        analyzer = LayoutAnalyzer()

        progress = st.progress(0, text="扫描中...")
        for p in range(pairs):
            progress.progress(
                (p * 2) / (pairs * 2),
                text=f"学生 {p + 1}/{pairs}: 处理第1页..."
            )
            page1_path = images[p * 2]
            page2_path = images[p * 2 + 1]

            try:
                page1 = load_image(page1_path)
                regions1, page1 = preprocess_and_analyze(page1, 1, preprocessor, analyzer)
                student_id = extract_student_id(page1, regions1, threshold=threshold)
                choice_answers, choice_cells = recognize_choices(
                    page1, regions1, threshold, return_details=True)

                progress.progress(
                    (p * 2 + 1) / (pairs * 2),
                    text=f"学生 {p + 1}/{pairs}: 处理第2页..."
                )
                page2 = load_image(page2_path)
                regions2, page2 = preprocess_and_analyze(page2, 2, preprocessor, analyzer)
                judge_answers, judge_cells = recognize_judges(
                    page2, regions2, threshold, return_details=True)
                essay_text = recognize_essay(page2, regions2, ocr_engine,
                                            api_config=ocr_api_config)

                recognized = {
                    "choice": choice_answers,
                    "judge": judge_answers,
                    "essay": {q: essay_text for q in get_essay_questions(svc.answer_key)} if essay_text else {},
                }

                result = None
                report = ""
                if svc:
                    result = svc.grade(recognized)
                    report = svc.generate_report(result)
                    processed_dir = PATHS['processed_dir']
                    mark_and_save(student_id, page1, page2,
                                  regions1, regions2,
                                  choice_cells, judge_cells, result,
                                  choice_max=len(svc.answer_key.get('choice', {})) * svc.choice_score,
                                  judge_max=len(svc.answer_key.get('judge', {})) * svc.judge_score,
                                  essay_max=len(svc.answer_key.get('essay', {})) * svc.essay_max_score,
                                  output_dir=processed_dir)

                results.append({
                    "student_id": student_id or "??????????",
                    "page1": os.path.basename(page1_path),
                    "page2": os.path.basename(page2_path),
                    "choice": choice_answers,
                    "judge": judge_answers,
                    "essay": essay_text,
                    "result": result,
                    "report": report,
                })
            except Exception as e:
                results.append({
                    "student_id": "ERROR",
                    "page1": os.path.basename(page1_path),
                    "page2": os.path.basename(page2_path),
                    "error": str(e),
                })

            # P33: 增量保存，中途崩溃时已处理结果不丢失
            _ckpt = PATHS['batch_checkpoint']
            with open(_ckpt, 'w', encoding='utf-8') as _f:
                _json.dump(results, _f, ensure_ascii=False, default=str)

        progress.progress(1.0, text="完成!")

        if not results:
            st.warning("未产生任何结果")
            st.stop()

        st.success(f"已完成 {len(results)} 份试卷的识别!")

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

        # ── Summary table ──
        st.subheader("成绩汇总")
        summary = []
        for r in results:
            if "error" in r:
                summary.append({
                    "学号": r["student_id"], "总分": "ERROR",
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
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.divider()

        # ── Per-student details ──
        st.subheader("逐份详情")
        for i, r in enumerate(results):
            if "error" in r:
                with st.expander(f"[失败] 学生 {i + 1}: {r['page1']}"):
                    st.error(r["error"])
                continue

            res = r["result"]
            if res and svc:
                total = res["total"]
                pct = total / svc.max_total * 100 if svc.max_total > 0 else 0
                label = (f"学生 {i + 1}: {r['student_id']}  "
                         f" {total}/{svc.max_total} ({pct:.0f}%)")
            else:
                label = f"学生 {i + 1}: {r['student_id']}"

            with st.expander(label):
                if res and svc:
                    render_score_metrics(res, svc)

                dt1, dt2, dt3 = st.tabs(["选择题", "判断题", "简答题"])

                with dt1:
                    _ch = LAYOUT['choice']
                    rows = render_question_table(
                        _ch['question_start'], _ch['question_start'] + _ch['question_count'] - 1,
                        r["choice"], res["choice"] if res else None)
                    st.dataframe(rows, use_container_width=True, hide_index=True)

                with dt2:
                    _ju = LAYOUT['judge']
                    rows = render_question_table(
                        _ju['question_start'], _ju['question_start'] + _ju['question_count'] - 1,
                        r["judge"], res["judge"] if res else None)
                    st.dataframe(rows, use_container_width=True, hide_index=True)

                with dt3:
                    st.text_area("OCR 文本", r.get("essay", "(空)"),
                                 height=100, key=f"essay_{i}")
                    if llm_enabled and res:
                        essay_detail = res.get("essay_detail", {})
                        for q, d in essay_detail.items():
                            st.markdown(
                                f"**第{q}题: {d['score']}/{d['max_score']}分**")
                            st.caption(f"反馈: {d['feedback']}")
                    else:
                        st.info(f"未启用 LLM 评分，需人工评分（{svc.essay_max_score if svc else '?'}分）")

                if r.get("report"):
                    with st.expander("详细报告"):
                        st.code(r["report"], language=None)

        # ── Download ──
        st.divider()
        if svc:
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
                if "error" in r:
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
            st.download_button(
                "导出成绩 (xlsx)", buf,
                file_name="grading_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
