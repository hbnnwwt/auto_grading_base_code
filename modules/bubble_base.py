"""选择题/判断题识别器的共享基类。

抽取 _trim_margin、_analyze_zones、_detect_fill_start、
recognize_with_viz 等在 ChoiceRecognizer 和 JudgeRecognizer 中
完全重复的方法。
"""

import cv2
import numpy as np

from modules.defaults import MORPH_KERNEL, FILL_BAND_THRESHOLD


class BubbleRecognizerBase:
    """气泡填涂识别器基类。"""

    COLOR_SELECTED = (0, 200, 0)
    COLOR_HALF = (0, 160, 255)
    COLOR_LINE = (180, 180, 180)

    def __init__(self, threshold=0.06, margin=5, zone_count=4,
                 option_labels=None, multi_threshold=None):
        self.threshold = threshold
        self.margin = margin
        self.zone_count = zone_count
        self.option_labels = option_labels or [
            chr(ord('A') + i) for i in range(zone_count)]
        self.multi_threshold = multi_threshold

    def _trim_margin(self, image):
        """裁剪图像四周边距。"""
        m = self.margin
        if m <= 0:
            return image
        h, w = image.shape[:2]
        if h <= 2 * m or w <= 2 * m:
            return image
        return image[m:-m, m:-m]

    def _analyze_zones(self, image):
        """分析各区域像素密度。

        Returns:
            dict: zone_fills, best_idx, above_threshold, is_multi

        启发式问题：
        - 填涂了的气泡和空白气泡，在二值化图像中分别呈现为什么颜色？
        - 如果直接把整列的像素都统计进去，印刷文字和边框会产生什么干扰？
          如何用形态学操作消除这些干扰？
        - "多选"在像素分布上有什么特征？如何设计一个阈值来检测多选？
        """
        raise NotImplementedError("请实现区域填充密度分析")

    def _detect_fill_start(self, gray):
        """通过水平投影频带检测填涂区域起始行。

        启发式问题：
        - 答题卡区域上方通常有标题/题号栏，下方才是填涂区域。
          水平投影如何区分"标题区域"和"填涂区域"？
        - 为什么可以用"黑色像素占比"来判断一行是否属于填涂区域？
        """
        raise NotImplementedError("请实现填涂起始行检测")

    def _refine_cell_boundaries(self, gray_fill, expected_rows, expected_cols):
        """基于高阈值间隙检测精修实际的行列边界。

        1. 高阈值二值化（230）只保留极暗像素（格线、填涂），
           间隙行/列几乎全白。
        2. 提取连续间隙区域，在预期边界附近匹配最佳间隙中心。
        3. 列边界保守精修（最大偏移 10px），避免气泡干扰导致列切分失常。

        Returns:
            tuple: (row_boundaries, col_boundaries)，各为像素坐标列表

        启发式问题：
        - 均匀切分网格可能因印刷/扫描偏差导致边界不准。
          如何用灰度特征找到"真正的"行列分界线？
        - 格线通常比填涂和文字都亮（在白纸上是空白），
          高阈值二值化后格线会呈现为什么？间隙呢？
        """
        raise NotImplementedError("请实现单元格边界精修")

    def recognize(self, image, options=None):
        """识别填涂区域，返回选中的选项。多选时返回 None。

        启发式问题：
        - 识别单个填涂区域时，需要调用哪些已实现的辅助方法？
        - 什么情况下应该返回 None（未填涂或多选）？
        """
        raise NotImplementedError("请实现单区域识别")

    def recognize_with_viz(self, image, options=None):
        """识别填涂区域并返回可视化标注图。多选时返回 None。

        Returns:
            tuple: (result, viz_image, zone_fills)

        启发式问题：
        - 可视化标注图需要显示哪些信息？选中选项、半填充选项、多选警告分别用什么颜色？
        - 如何在图像上绘制选项分隔线？
        """
        raise NotImplementedError("请实现带可视化的识别")
