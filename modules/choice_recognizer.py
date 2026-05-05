import cv2
import numpy as np

from modules.bubble_base import BubbleRecognizerBase
from modules.defaults import MORPH_KERNEL


class ChoiceRecognizer(BubbleRecognizerBase):
    """选择题填涂识别模块。

    调用者（pipeline.py）使用方式：
        rec = ChoiceRecognizer(threshold=0.06)
        result = rec.recognize_all_with_viz(
            roi, question_count=20, question_start=1,
            fixed_grid=(5, 4))
        answers = result['answers']  # {1: 'A', 2: 'C', ...}

    目标：给定选择题填涂区域图像，返回每道题的选择（A/B/C/D）。
    """

    def __init__(self, threshold=0.06, option_count=4, margin=5,
                 multi_threshold=0.12):
        """
        Args:
            threshold: 填充率阈值，低于此值认为未填涂
            option_count: 选项数量（默认4: A/B/C/D）
            margin: 裁剪边距像素数
            multi_threshold: 多选检测阈值
        """
        super().__init__(
            threshold=threshold, margin=margin,
            zone_count=option_count,
            option_labels=[chr(ord('A') + i) for i in range(option_count)],
            multi_threshold=multi_threshold)

    def _detect_zone_boundaries(self, binary, cell_w, cell_h):
        """检测单元格内各选项（气泡）的水平边界。

        Args:
            binary: ndarray, 单元格的二值化图像
            cell_w: int, 单元格宽度
            cell_h: int, 单元格高度

        Returns:
            tuple: (zone_bounds, num_w)
                - zone_bounds: list[tuple], 每个选项的 (x_start, x_end) 边界
                - num_w: int, 题号区域宽度

        思路提示：
            - 气泡是什么形状的？它的轮廓有什么特征可以用来把它和背景区分开？
            - 如果一个单元格内有 4 个气泡，它们在位置上有什么规律？
            - 如何用连通域分析找到这些气泡的中心点？
            - 如果某个气泡因为打印质量问题断了，该怎么修复？这种修复会不会影响识别？
            - 气泡的位置是严格等距的吗？能否用气泡间距的统计特征（而非固定值）来切分？
        """
        raise NotImplementedError("TODO: 请实现 _detect_zone_boundaries() 方法")

    def recognize_all_with_viz(self, region_image, question_count=20,
                               question_start=1, options=None,
                               fixed_grid=None):
        """识别所有选择题并返回可视化结果。

        Args:
            region_image: ndarray, 选项区域的 BGR 图像
            question_count: int, 题目数量（默认20）
            question_start: int, 起始题号（默认1）
            options: list[str], 选项标签（默认 ['A','B','C','D']）
            fixed_grid: tuple(int, int), 网格 (rows, cols)，默认 (5, 4)

        Returns:
            dict: {
                'answers': {题号: 'A'/'B'/'C'/'D'/None},
                'grid_viz': ndarray, 标注了网格和答案的可视化图像,
                'cell_results': [...]
            }

        思路提示：
            - 整个选择题区域中，所有气泡排列成什么形状？有没有现成的线条可以作为参考？
            - 如果拍摄角度偏了，气泡形状会怎么变化？还能用矩形网格切分吗？
            - 什么叫"填涂"？填涂后的像素分布和未填涂有什么本质区别？
            - 多选的情况怎么处理？学生同时涂了 A 和 C，和只涂 A 的像素分布有什么不同？
            - 什么是"多阈值检测"？用一个阈值和两个阈值分别怎么判断多选？
        """
        raise NotImplementedError("TODO: 请实现 recognize_all_with_viz() 方法")
