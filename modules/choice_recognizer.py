import cv2
import numpy as np

from modules.bubble_base import BubbleRecognizerBase
from modules.defaults import MORPH_KERNEL


class ChoiceRecognizer(BubbleRecognizerBase):
    """选择题填涂识别模块。

    对选项区域进行形态学处理 + 连通域分析 + 像素密度统计，
    识别填涂的选项。
    """

    def __init__(self, threshold=0.06, option_count=4, margin=5,
                 multi_threshold=0.75,
                 blank_baseline=None, gray_darkening_threshold=8.0,
                 zone_bounds_template=None):
        self.bubble_min_w = 0.08
        self.bubble_max_w = 0.30
        self.bubble_min_h = 0.40
        super().__init__(
            threshold=threshold, margin=margin,
            zone_count=option_count,
            option_labels=[chr(ord('A') + i) for i in range(option_count)],
            multi_threshold=multi_threshold)
        self.blank_baseline = blank_baseline or {}
        self.gray_darkening_threshold = gray_darkening_threshold
        # zone_bounds_template: {q: [(left_rel, right_rel), ...]}
        self.zone_bounds_template = zone_bounds_template or {}

    @property
    def option_count(self):
        return self.zone_count

    def _detect_zone_boundaries_projection(self, gray, cell_w, cell_h):
        """基于灰度垂直投影的峰值检测定位气泡位置。

        直接在原始灰度图上工作，不依赖二值化连通域质量。
        气泡区域更暗，在反转后的投影中表现为峰值。
        在预期位置附近搜索局部最大值，然后以峰值间中点划分 zone。

        Returns:
            tuple: (zone_bounds, num_w)

        启发式问题：
        - 气泡区域通常比背景更暗。如何在垂直投影中体现这一特征？
        - 平滑后的投影曲线可能有多个局部峰值，如何在预期位置附近找到"真正的"气泡中心？
        - 找到气泡中心后，如何计算相邻气泡之间的分界线？
        """
        raise NotImplementedError("请实现投影峰值检测")

    def _detect_zone_boundaries(self, gray, binary, cell_w, cell_h):
        """基于气泡实际位置划分选项区域边界。

        先用连通域检测；若气泡不足（淡铅笔/空白时常见），
        回退到基于灰度投影的峰值检测，比均匀切分更精确。

        Returns:
            tuple: (zone_bounds, num_w)

        启发式问题：
        - 如果每个选项都有一个圆形/椭圆形的气泡，二值化后气泡会呈现为什么？
        - 连通域分析能直接找到气泡位置，但如果填涂很淡（气泡不明显），
          连通域方法会失效。如何设计一个"主方案+备用方案"的策略？
        - 当检测到的气泡中心数量超过 4 个时，如何选出"最均匀分布"的 4 个？
        """
        raise NotImplementedError("请实现选项边界检测")

    @staticmethod
    def _select_best_bubbles(sorted_cx, count):
        """从已排序的气泡中心中选出间距最均匀的 count 个。

        启发式问题：
        - "间距最均匀"如何用数学方法量化？方差？标准差？还是绝对偏差之和？
        - 为什么只需要检查连续子序列，而不需要考虑非连续的跳跃选择？
        """
        raise NotImplementedError("请实现最优气泡选择")

    def _detect_rows_fixed(self, region_image, expected_rows, expected_cols):
        """切分为 expected_rows × expected_cols 的网格。

        先用均匀切分获得初始边界，再通过投影精修行列边界。

        启发式问题：
        - 均匀切分可能因印刷偏差导致不准。如何在均匀切分的基础上，
          利用投影特征精修行列边界？
        """
        raise NotImplementedError("请实现固定网格切分")

    def recognize_all_with_viz(self, region_image, question_count,
                               question_start=1, options=None,
                               fixed_grid=None):
        """批量识别选择题区域。

        Args:
            region_image: 整个选择题区域图像
            question_count: 题目数量
            question_start: 起始题号
            options: 选项标签列表
            fixed_grid: (rows, cols)，默认 (5, 4)

        启发式问题：
        - 如何将整个选择题区域切分为 (rows, cols) 的网格？每格对应一道题。
        - 每道题有 A/B/C/D 四个选项，如何调用基类方法分析每个选项的填充密度？
        - 多选检测：如果多个选项的填充率都很高，如何判断是真正的多选还是污渍/阴影？
        - 如何在网格可视化图上标注每道题的识别结果？答对的绿色、答错/多选/未答的红色？
        """
        raise NotImplementedError("请实现批量选择题识别")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        recognizer = ChoiceRecognizer()
        result = recognizer.recognize(img)
        print(f"识别结果: {result}")
