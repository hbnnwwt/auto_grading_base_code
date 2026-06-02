import cv2
import numpy as np

from modules.bubble_base import BubbleRecognizerBase
from modules.defaults import (MORPH_KERNEL, JUDGE_MIN_FILL, JUDGE_STAIN_FILL_LOW,
                              JUDGE_STAIN_FILL_HIGH, JUDGE_STAIN_RATIO, JUDGE_VALID_RATIO,
                              JUDGE_MULTI_FILL, JUDGE_MULTI_RATIO, JUDGE_BLOB_AREA_MIN,
                              JUDGE_BLOB_AREA_MAX, JUDGE_BLOB_ASPECT_MIN, JUDGE_BLOB_ASPECT_MAX,
                              JUDGE_SIDE_MARGIN_RATIO, JUDGE_VERT_MARGIN_RATIO)


class JudgeRecognizer(BubbleRecognizerBase):
    """判断题填涂识别模块。

    识别"对"/"错"两个填涂气泡的密度。
    支持空白答卷基准校准，用于检测淡铅笔痕迹。
    """

    def __init__(self, threshold=0.06, margin=5, multi_threshold=0.10,
                 blank_baseline=None, gray_darkening_threshold=8.0,
                 zone_bounds_template=None):
        super().__init__(
            threshold=threshold, margin=margin,
            zone_count=2,
            option_labels=['T', 'F'],
            multi_threshold=multi_threshold)
        # blank_baseline: {question_number: [zone0_gray_mean, zone1_gray_mean, ...]}
        self.blank_baseline = blank_baseline or {}
        self.gray_darkening_threshold = gray_darkening_threshold
        self.zone_bounds_template = zone_bounds_template or {}

    def _detect_zone_boundaries(self, gray, cell_w, cell_h,
                                 bubble_data=None):
        """基于实际气泡位置或灰度投影划分 T/F zone 边界。

        zone 宽度严格限制为可用宽度的 30%，避免把非气泡区域纳入统计。
        先获取 T/F 中心位置（连通域或灰度峰值），再以中点为界对称收窄。

        启发式问题：
        - 判断题只有 T/F 两个选项，zone 宽度如果太大，会包含非气泡区域。
          如何限制 zone 宽度，使其只覆盖气泡本身？
        - 如果连通域检测找到了气泡，如何利用气泡中心位置确定分界线？
          如果找不到，如何回退到投影峰值检测？
        """
        raise NotImplementedError("请实现 T/F 边界检测")

    def _detect_bubbles_in_cell(self, binary, cell_w, cell_h):
        """检测格子内的方形气泡区域。

        Returns:
            dict: {
                'T': (zx0, zy0, zx1, zy1),  # T 气泡边界
                'F': (zx0, zy0, zx1, zy1),  # F 气泡边界
                'blobs': [(x, y, w, h), ...] # 所有检测到的 blob
            }

        启发式问题：
        - 与选择题不同，判断题的气泡更大、更方。连通域的面积和宽高比应该落在什么范围？
        - 如何区分"气泡"和"格线残余"？位置信息（中心 vs 边缘）如何帮助过滤？
        - 如果完全没有找到气泡，如何设计一个合理的 fallback 区域？
        """
        raise NotImplementedError("请实现气泡检测")

    def _detect_cells_fixed(self, region_image, rows_n, cols_n, cell_mapping):
        """切分网格 + 自定义题号映射。

        先用均匀切分获得初始边界，再通过投影精修行列边界。

        启发式问题：
        - 判断题的题号通常从 21 开始，而不是 1。如何在切分网格时映射正确的题号？
        - 如果总网格数（rows_n × cols_n）大于实际题目数量，多余的格子应该如何处理？
        """
        raise NotImplementedError("请实现网格切分与题号映射")

    def recognize_all_with_viz(self, region_image, question_count=None,
                               question_start=21, rows_n=3, cols_n=4):
        """批量识别判断题区域。

        Args:
            region_image: 整个判断题区域图像
            question_count: 题目数量，默认从参数推断
            question_start: 起始题号
            rows_n: 网格行数
            cols_n: 网格列数

        启发式问题：
        - 判断题的识别流程与选择题有什么相似之处？有什么不同之处？
        - T/F 两个选项的填涂密度差异如何判断？什么情况下应该判定为"多选"（TF 同时填涂）？
        - 空白基准校准（blank_baseline）如何利用灰度差异提升淡铅笔识别率？
        """
        raise NotImplementedError("请实现批量判断题识别")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        recognizer = JudgeRecognizer()
        result = recognizer.recognize(img)
        print(f"识别结果: {result}")
