import cv2
import numpy as np

from modules.bubble_base import BubbleRecognizerBase
from modules.defaults import MORPH_KERNEL


class JudgeRecognizer(BubbleRecognizerBase):
    """判断题填涂识别模块。

    调用者（pipeline.py）使用方式：
        rec = JudgeRecognizer(threshold=0.06)
        result = rec.recognize_all_with_viz(
            roi, question_count=10, question_start=21,
            rows_n=3, cols_n=4)
        answers = result['answers']  # {21: 'T', 22: 'F', ...}

    目标：给定判断题填涂区域图像，返回每道题的选择（T/F）。
    与选择题的区别：只有 T/F 两个选项，且填涂密度更高。
    """

    def __init__(self, threshold=0.06, margin=5, multi_threshold=0.10):
        """
        Args:
            threshold: 填充率阈值
            margin: 裁剪边距像素数
            multi_threshold: 多选检测阈值（判断题填涂密度更高，默认0.10）
        """
        super().__init__(
            threshold=threshold, margin=margin,
            zone_count=2,
            option_labels=['T', 'F'],
            multi_threshold=multi_threshold)

    def _detect_zone_boundaries(self, binary, cell_w, cell_h):
        """检测单元格内 T/F 两个选项的水平边界。

        Args:
            binary: ndarray, 单元格的二值化图像
            cell_w: int, 单元格宽度
            cell_h: int, 单元格高度

        Returns:
            tuple: (zone_bounds, num_w)
                - zone_bounds: list[tuple], 每个选项的 (x_start, x_end) 边界
                - num_w: int, 题号区域宽度

        思路提示：
            - 判断题只有 T 和 F 两个选项，它们在单元格内是怎么排列的？
            - 相比选择题（4个选项），判断题的切分有什么简化之处？
            - 如果不依赖连通域，直接把单元格二等分，识别效果会差多少？
        """
        raise NotImplementedError("TODO: 请实现 _detect_zone_boundaries() 方法")

    def recognize_all_with_viz(self, region_image, question_count=None,
                               question_start=21, rows_n=3, cols_n=4):
        """识别所有判断题并返回可视化结果。

        Args:
            region_image: ndarray, 判断题区域的 BGR 图像
            question_count: int, 题目数量（默认10）
            question_start: int, 起始题号（默认21）
            rows_n: int, 网格行数（默认3）
            cols_n: int, 网格列数（默认4）

        Returns:
            dict: {
                'answers': {题号: 'T'/'F'/None},
                'grid_viz': ndarray,
                'cell_results': [...]
            }

        思路提示：
            - 判断题的 T 和 F 选项，哪个涂出来更"实"？这会影响阈值的选择吗？
            - 判断题网格（3行4列）和选择题网格（5行4列），
              题号排列顺序是一样的吗？画个草图验证一下。
        """
        raise NotImplementedError("TODO: 请实现 recognize_all_with_viz() 方法")
