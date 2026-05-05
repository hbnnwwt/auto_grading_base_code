"""Shared UI components."""
import os

import cv2
import numpy as np
import streamlit as st


def load_image(path):
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def load_image_from_bytes(data):
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img


def render_question_table(q_start, q_end, answers, result_detail):
    rows = []
    for q in range(q_start, q_end + 1):
        given = answers.get(q, "-")
        if result_detail and q in result_detail:
            d = result_detail[q]
            correct = d["correct"] or "-"
            mark = "O" if d["score"] > 0 else "X"
        else:
            correct, mark = "-", "-"
        rows.append({
            "题号": q, "作答": given,
            "正确答案": correct, "结果": mark,
        })
    return rows


def render_score_bar(total, max_total):
    if max_total <= 0:
        return
    bar_ratio = total / max_total
    bar_color = ("#22c55e" if bar_ratio >= 0.6
                 else "#eab308" if bar_ratio >= 0.4
                 else "#ef4444")
    pct = int(bar_ratio * 100)
    st.markdown(
        f'<div role="progressbar" aria-valuenow="{pct}" '
        f'aria-valuemin="0" aria-valuemax="100" '
        f'aria-label="得分 {total}/{max_total}" '
        f'style="background:#e2e8f0;border-radius:6px;height:8px;overflow:hidden;">'
        f'<div style="background:{bar_color};width:{pct}%;height:100%;"></div>'
        f'</div>',
        unsafe_allow_html=True
    )


def render_score_metrics(result, svc):
    if not result or not svc:
        return
    cs = result["choice_total"]
    js = result["judge_total"]
    es = result.get("essay_total", 0)
    choice_max = len(svc.answer_key.get('choice', {})) * svc.choice_score
    judge_max = len(svc.answer_key.get('judge', {})) * svc.judge_score
    essay_max = len(svc.answer_key.get('essay', {})) * svc.essay_max_score
    total = result["total"]
    pct = total / svc.max_total * 100 if svc.max_total > 0 else 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总分", f"{total}", f"{svc.max_total} 满分")
    c2.metric("正确率", f"{pct:.0f}%")
    c3.metric("选择题", f"{cs}/{choice_max}")
    c4.metric("判断题", f"{js}/{judge_max}")
    c5.metric("简答题", f"{es}/{essay_max}")
    render_score_bar(total, svc.max_total)


def image_to_bytes(image, ext='.png'):
    _, buf = cv2.imencode(ext, image)
    return buf.tobytes()


def imwrite(path, image):
    """cv2.imwrite 不支持中文路径，用 imencode + 写文件绕过。"""
    ext = os.path.splitext(path)[1] or '.png'
    ok, buf = cv2.imencode(ext, image)
    if ok:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
    return ok
