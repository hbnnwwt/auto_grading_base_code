"""选择题识别模块测试。

验证 ChoiceRecognizer 能正确识别模拟的填涂气泡。
运行: pytest tests/test_choice_recognizer.py -v
"""

import cv2
import numpy as np
import pytest

from modules.choice_recognizer import ChoiceRecognizer


# ---------------------------------------------------------------------------
# fixture: 生成含 A/B/C/D 四个选项气泡的模拟图像
# ---------------------------------------------------------------------------

def make_choice_image(filled_index, total_options=4, bubble_size=40, gap=20):
    """生成选择题选项图像。

    Args:
        filled_index: 填涂的选项索引 (0=A, 1=B, 2=C, 3=D)，None 表示未填涂
        total_options: 选项总数
        bubble_size: 气泡直径
        gap: 气泡间距

    Returns:
        灰度图像 (numpy.ndarray)
    """
    w = total_options * bubble_size + (total_options + 1) * gap
    h = bubble_size + 2 * gap
    img = np.ones((h, w), dtype=np.uint8) * 255

    for i in range(total_options):
        cx = gap + i * (bubble_size + gap) + bubble_size // 2
        cy = h // 2
        if i == filled_index:
            # 填涂：实心黑色圆
            cv2.circle(img, (cx, cy), bubble_size // 2, 0, -1)
        else:
            # 未填涂：空心圆
            cv2.circle(img, (cx, cy), bubble_size // 2, 0, 1)

    return img


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestChoiceRecognizer:
    def test_init_default(self):
        """默认参数初始化。"""
        rec = ChoiceRecognizer()
        assert rec.threshold == 0.06
        assert rec.option_count == 4

    def test_recognize_filled_A(self):
        """填涂 A 选项应识别为 'A'。"""
        img = make_choice_image(filled_index=0)
        rec = ChoiceRecognizer(threshold=0.06)
        result = rec.recognize(img)
        assert result == 'A'

    def test_recognize_filled_B(self):
        """填涂 B 选项应识别为 'B'。"""
        img = make_choice_image(filled_index=1)
        rec = ChoiceRecognizer(threshold=0.06)
        result = rec.recognize(img)
        assert result == 'B'

    def test_recognize_filled_C(self):
        """填涂 C 选项应识别为 'C'。"""
        img = make_choice_image(filled_index=2)
        rec = ChoiceRecognizer(threshold=0.06)
        result = rec.recognize(img)
        assert result == 'C'

    def test_recognize_filled_D(self):
        """填涂 D 选项应识别为 'D'。"""
        img = make_choice_image(filled_index=3)
        rec = ChoiceRecognizer(threshold=0.06)
        result = rec.recognize(img)
        assert result == 'D'

    def test_recognize_no_fill_returns_none(self):
        """未填涂任何选项应返回 None。"""
        img = make_choice_image(filled_index=None)
        rec = ChoiceRecognizer(threshold=0.06)
        result = rec.recognize(img)
        assert result is None

    def test_recognize_custom_options(self):
        """自定义选项标签。"""
        img = make_choice_image(filled_index=0)
        rec = ChoiceRecognizer(threshold=0.06)
        result = rec.recognize(img, options=['甲', '乙', '丙', '丁'])
        assert result == '甲'

    def test_multiple_questions(self):
        """多题连续识别：模拟 5 道选择题。"""
        rec = ChoiceRecognizer(threshold=0.06)
        answers = [0, 1, 2, 3, 0]  # A, B, C, D, A
        expected = ['A', 'B', 'C', 'D', 'A']
        for filled_idx, expected_ans in zip(answers, expected):
            img = make_choice_image(filled_index=filled_idx)
            result = rec.recognize(img)
            assert result == expected_ans, (
                f"填涂索引 {filled_idx} 应识别为 {expected_ans}，实际为 {result}"
            )


# ---------------------------------------------------------------------------
# 多行区域测试：行检测 + recognize_all_with_viz
# ---------------------------------------------------------------------------

def make_choice_region_image(filled_indices, num_options=4,
                             bubble_size=30, gap=8, row_gap=15,
                             num_width=30):
    """生成多行选择题区域图像（含题号、网格线、边框）。

    Args:
        filled_indices: 每题填涂的选项索引列表，如 [0, 2, 1, ...]。
                        None 表示该题未填涂。
        num_options: 每题选项数
        bubble_size: 气泡直径
        gap: 气泡间距
        row_gap: 行间距
        num_width: 左侧题号区域宽度
    """
    question_count = len(filled_indices)
    cell = bubble_size + gap
    row_h = bubble_size + 2 * gap
    options_w = num_options * cell + gap
    w = num_width + options_w
    h = question_count * row_h + (question_count - 1) * row_gap

    img = np.ones((h, w), dtype=np.uint8) * 255

    for q_idx, filled in enumerate(filled_indices):
        y_off = q_idx * (row_h + row_gap)
        # 题号
        cv2.putText(img, str(q_idx + 1),
                    (5, y_off + row_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
        # 选项气泡
        for opt in range(num_options):
            cx = num_width + gap + opt * cell + bubble_size // 2
            cy = y_off + row_h // 2
            if opt == filled:
                cv2.circle(img, (cx, cy), bubble_size // 2, 0, -1)
            else:
                cv2.circle(img, (cx, cy), bubble_size // 2, 0, 1)

    return img


class TestDetectRowsFixed:
    """测试固定网格切分方法。"""

    def test_correct_cell_count(self):
        img = np.ones((300, 400), dtype=np.uint8) * 255
        rec = ChoiceRecognizer(threshold=0.06)
        cells = rec._detect_rows_fixed(img, 5, 4)
        assert len(cells) == 20

    def test_cells_cover_full_image(self):
        img = np.ones((300, 400), dtype=np.uint8) * 255
        rec = ChoiceRecognizer(threshold=0.06)
        cells = rec._detect_rows_fixed(img, 5, 4)
        assert cells[0][0] == 0
        assert cells[0][2] == 0
        assert cells[-1][1] == 300
        assert cells[-1][3] == 400

    def test_detect_fill_start_blank(self):
        img = np.ones((300, 400), dtype=np.uint8) * 255
        rec = ChoiceRecognizer(threshold=0.06)
        assert rec._detect_fill_start(img) == 0


class TestRecognizeAllWithViz:
    def test_returns_all_keys(self):
        img = np.ones((300, 400), dtype=np.uint8) * 255
        rec = ChoiceRecognizer(threshold=0.06)
        result = rec.recognize_all_with_viz(img, question_count=20)
        for key in ['answers', 'grid_viz', 'cell_results']:
            assert key in result

    def test_cell_results_count(self):
        img = np.ones((300, 400), dtype=np.uint8) * 255
        rec = ChoiceRecognizer(threshold=0.06)
        result = rec.recognize_all_with_viz(img, question_count=20,
                                            fixed_grid=(5, 4))
        assert len(result['cell_results']) == 20

    def test_cell_results_structure(self):
        img = np.ones((300, 400), dtype=np.uint8) * 255
        rec = ChoiceRecognizer(threshold=0.06)
        result = rec.recognize_all_with_viz(img, question_count=20)
        for r in result['cell_results']:
            assert 'question' in r
            assert 'answer' in r
            assert 'zone_fills' in r
