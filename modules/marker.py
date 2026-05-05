"""在旋转校正后的答题卡图片上标注错题红色 X 及各部分得分。"""

import os

import cv2
import numpy as np

from modules.pipeline import LAYOUT, _valid_region

_CHOICE_LABELS = list("ABCD")[:LAYOUT['choice']['cols']]
_JUDGE_LABELS = list("TF")[:LAYOUT['judge']['cols']]


def _imwrite(path, image):
    ext = os.path.splitext(path)[1] or '.png'
    ok, buf = cv2.imencode(ext, image)
    if ok:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
    return ok


def _put_score(image, text, region, color=(0, 0, 255), font_scale=1.2):
    """在区域正上方居中写分数。"""
    x, y, w, h = region
    thickness = max(1, int(font_scale * 2))
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = x + (w - tw) // 2
    ty = max(y - 15, th + 10)
    cv2.rectangle(image, (tx - 6, ty - th - 6),
                  (tx + tw + 6, ty + baseline + 6), (255, 255, 255), -1)
    cv2.putText(image, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
                cv2.LINE_AA)


def mark_wrong_on_page(page, region, cell_results, grading_detail,
                       option_labels, color=(0, 0, 255), thickness=3):
    """在错误选项的气泡位置画红色 X。"""
    if not _valid_region(region):
        return page.copy()
    marked = page.copy()
    rx, ry = region[0], region[1]

    for cell in cell_results:
        q_num = cell['question']
        if q_num not in grading_detail:
            continue
        detail = grading_detail[q_num]
        if detail['score'] > 0:
            continue
        if detail['given'] is None or detail['given'] == '':
            continue

        given = detail['given']
        if given not in option_labels:
            continue

        option_idx = option_labels.index(given)
        zone_bounds = cell.get('zone_bounds', [])
        if option_idx >= len(zone_bounds):
            continue

        zx0, zx1 = zone_bounds[option_idx]
        y0, y1, x0, _ = cell['cell_bounds']

        bx1 = rx + x0 + zx0
        by1 = ry + y0
        bx2 = rx + x0 + zx1
        by2 = ry + y1

        pad = 2
        cv2.line(marked, (bx1 + pad, by1 + pad), (bx2 - pad, by2 - pad),
                 color, thickness)
        cv2.line(marked, (bx2 - pad, by1 + pad), (bx1 + pad, by2 - pad),
                 color, thickness)

    return marked


def mark_and_save(student_id, page1, page2, regions1, regions2,
                  choice_cells, judge_cells, grading_result,
                  choice_max, judge_max, essay_max,
                  output_dir='data/processed'):
    """标注两页错题、写上各部分得分并保存。

    Args:
        choice_max/judge_max/essay_max: 各部分满分
    Returns:
        (p1_path, p2_path, marked_p1, marked_p2)
    """
    os.makedirs(output_dir, exist_ok=True)
    sid = (student_id or 'unknown').replace('?', '_')

    cs = grading_result.get('choice_total', 0)
    js = grading_result.get('judge_total', 0)
    es = grading_result.get('essay_total', 0)

    # ── 第1页：选择题 × + 得分 ──
    marked_p1 = page1.copy()
    if regions1.get('choice') and choice_cells:
        marked_p1 = mark_wrong_on_page(
            marked_p1, regions1['choice'], choice_cells,
            grading_result.get('choice', {}),
            _CHOICE_LABELS)
        _put_score(marked_p1, f'Choice: {cs}/{choice_max}',
                   regions1['choice'])

    # ── 第2页：判断题 × + 得分 + 简答题得分 ──
    if page2 is not None:
        marked_p2 = page2.copy()
    else:
        marked_p2 = np.zeros_like(page1)
    if page2 is not None and regions2.get('judge') and judge_cells:
        marked_p2 = mark_wrong_on_page(
            marked_p2, regions2['judge'], judge_cells,
            grading_result.get('judge', {}),
            _JUDGE_LABELS)
        _put_score(marked_p2, f'Judge: {js}/{judge_max}',
                   regions2['judge'])

    if _valid_region(regions2.get('essay')):
        _put_score(marked_p2, f'Essay: {es}/{essay_max}',
                   regions2['essay'])

    p1_path = os.path.join(output_dir, f'{sid}_page1_marked.png')
    p2_path = os.path.join(output_dir, f'{sid}_page2_marked.png')
    # 防止学号重复覆盖：追加序号直到两个文件都不存在
    for suffix in range(100):
        if not os.path.exists(p1_path) and not os.path.exists(p2_path):
            break
        p1_path = os.path.join(output_dir, f'{sid}_page1_marked_{suffix}.png')
        p2_path = os.path.join(output_dir, f'{sid}_page2_marked_{suffix}.png')
    _imwrite(p1_path, marked_p1)
    _imwrite(p2_path, marked_p2)

    return p1_path, p2_path, marked_p1, marked_p2
