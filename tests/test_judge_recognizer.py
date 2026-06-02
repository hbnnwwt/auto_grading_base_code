"""判断题识别模块测试。

验证 JudgeRecognizer 能正确识别 T/F 填涂。
运行: pytest tests/test_judge_recognizer.py -v
"""

import cv2
import numpy as np
import pytest

from modules.judge_recognizer import JudgeRecognizer


# ---------------------------------------------------------------------------
# fixture: 生成含 T/F 两个气泡的模拟图像
# ---------------------------------------------------------------------------

def make_judge_image(filled='T', bubble_size=40, gap=60):
    """生成判断题 T/F 填涂图像。

    Args:
        filled: 'T' 填涂"对"，'F' 填涂"错"，None 未填涂
        bubble_size: 气泡直径
        gap: 气泡间距

    Returns:
        灰度图像 (numpy.ndarray)
    """
    w = 2 * bubble_size + 3 * gap
    h = bubble_size + 2 * gap
    img = np.ones((h, w), dtype=np.uint8) * 255

    positions = {
        'T': (gap + bubble_size // 2, h // 2),
        'F': (2 * gap + bubble_size + bubble_size // 2, h // 2),
    }

    for label, (cx, cy) in positions.items():
        if label == filled:
            cv2.circle(img, (cx, cy), bubble_size // 2, 0, -1)
        else:
            cv2.circle(img, (cx, cy), bubble_size // 2, 0, 1)

    return img


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestJudgeRecognizer:
    def test_init_default(self):
        """默认参数初始化。"""
        rec = JudgeRecognizer()
        assert rec.threshold == 0.06

    def test_recognize_true(self):
        """填涂 T 应识别为 'T'。"""
        img = make_judge_image(filled='T')
        rec = JudgeRecognizer(threshold=0.06)
        result = rec.recognize(img)
        assert result == 'T'

    def test_recognize_false(self):
        """填涂 F 应识别为 'F'。"""
        img = make_judge_image(filled='F')
        rec = JudgeRecognizer(threshold=0.06)
        result = rec.recognize(img)
        assert result == 'F'

    def test_recognize_no_fill_returns_none(self):
        """未填涂应返回 None。"""
        img = make_judge_image(filled=None)
        rec = JudgeRecognizer(threshold=0.06)
        result = rec.recognize(img)
        assert result is None

    def test_multiple_questions(self):
        """多题连续识别。"""
        rec = JudgeRecognizer(threshold=0.06)
        cases = [
            ('T', 'T'),
            ('F', 'F'),
            ('T', 'T'),
            (None, None),
        ]
        for filled, expected in cases:
            img = make_judge_image(filled=filled)
            result = rec.recognize(img)
            assert result == expected, (
                f"填涂 {filled} 应识别为 {expected}，实际为 {result}"
            )


# ---------------------------------------------------------------------------
# 多行区域测试
# ---------------------------------------------------------------------------

def make_judge_region_image(filled_labels, bubble_size=30, gap=20,
                            row_gap=15, num_width=30):
    """生成多行判断题区域图像（含题号）。

    Args:
        filled_labels: 每题填涂标签列表，如 ['T', 'F', 'T', None]。
    """
    question_count = len(filled_labels)
    cell = bubble_size + gap
    row_h = bubble_size + 2 * gap
    options_w = 2 * cell + gap
    w = num_width + options_w
    h = question_count * row_h + (question_count - 1) * row_gap

    img = np.ones((h, w), dtype=np.uint8) * 255

    positions = {'T': 0, 'F': 1}
    for q_idx, filled in enumerate(filled_labels):
        y_off = q_idx * (row_h + row_gap)
        cv2.putText(img, str(21 + q_idx),
                    (2, y_off + row_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
        for label, opt_idx in positions.items():
            cx = num_width + gap + opt_idx * cell + bubble_size // 2
            cy = y_off + row_h // 2
            if label == filled:
                cv2.circle(img, (cx, cy), bubble_size // 2, 0, -1)
            else:
                cv2.circle(img, (cx, cy), bubble_size // 2, 0, 1)

    return img


class TestJudgeDetectFillStart:
    def test_detect_fill_start_blank(self):
        img = np.ones((300, 400), dtype=np.uint8) * 255
        rec = JudgeRecognizer(threshold=0.06)
        assert rec._detect_fill_start(img) == 0

    def test_detect_cells_fixed_count(self):
        img = np.ones((300, 400), dtype=np.uint8) * 255
        rec = JudgeRecognizer(threshold=0.06)
        cell_mapping = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, None, None]
        cells = rec._detect_cells_fixed(img, 3, 4, cell_mapping)
        assert len(cells) == 12
        assert cells[0][4] == 21
        assert cells[-1][4] is None


class TestJudgeRecognizeAll:
    def test_returns_all_keys(self):
        img = np.ones((300, 400), dtype=np.uint8) * 255
        rec = JudgeRecognizer(threshold=0.06)
        result = rec.recognize_all_with_viz(img)
        for key in ['answers', 'grid_viz', 'cell_results']:
            assert key in result

    def test_cell_results_count(self):
        img = np.ones((300, 400), dtype=np.uint8) * 255
        rec = JudgeRecognizer(threshold=0.06)
        result = rec.recognize_all_with_viz(img)
        # 3×4 grid, 10 questions (21-30), 2 None cells skipped
        assert len(result['cell_results']) == 10

    def test_cell_results_structure(self):
        img = np.ones((300, 400), dtype=np.uint8) * 255
        rec = JudgeRecognizer(threshold=0.06)
        result = rec.recognize_all_with_viz(img)
        for r in result['cell_results']:
            assert 'question' in r
            assert 'answer' in r
            assert 'zone_fills' in r
