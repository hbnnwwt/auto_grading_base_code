"""选择题/判断题识别器的共享基类。

提供气泡填涂识别的通用工具：
- _trim_margin: 裁剪边距
- _analyze_zones: 计算各选项区域的像素填充率
- _detect_fill_start: 通过水平投影频带检测填涂区域起始行
- recognize: 识别单个单元格
- recognize_with_viz: 带可视化输出的识别

调用者（choice_recognizer.py / judge_recognizer.py）继承本基类后，
在 recognize_all_with_viz() 中调用上述方法完成识别。
"""

import cv2
import numpy as np

from modules.defaults import MORPH_KERNEL, FILL_BAND_THRESHOLD


class BubbleRecognizerBase:
    """气泡填涂识别器基类。子类必须实现所有以 _ 开头的方法。"""

    COLOR_SELECTED = (0, 200, 0)
    COLOR_HALF = (0, 160, 255)
    COLOR_LINE = (180, 180, 180)

    def __init__(self, threshold=0.06, margin=5, zone_count=4,
                 option_labels=None, multi_threshold=None):
        """
        Args:
            threshold: 填充率阈值，低于此值判定为未填涂
            margin: 裁剪边距像素数
            zone_count: 选项数量（选择题=4，判断题=2）
            option_labels: 选项标签列表，默认 ['A','B','C','D'] 或 ['T','F']
            multi_threshold: 多选检测阈值，超过此填充率的选项数>=2时判定为多选
        """
        self.threshold = threshold
        self.margin = margin
        self.zone_count = zone_count
        self.option_labels = option_labels or [
            chr(ord('A') + i) for i in range(zone_count)]
        self.multi_threshold = multi_threshold

    # ── 以下方法子类必须实现 ──────────────────────────────────────

    def _analyze_zones(self, image):
        """分析各区域（气泡）的像素填充率。

        Args:
            image: ndarray, 单个单元格内裁剪后的图像（BGR 或灰度）

        Returns:
            dict: {
                'zone_fills': list[float],   # 每个区域的黑色像素填充率 [0.0~1.0]
                'best_idx': int,             # 填充率最高的区域索引
                'above_threshold': bool,      # 最高填充率是否超过 threshold
                'is_multi': bool,           # 是否为多选（>=2个选项超过multi_threshold）
            }

        思路提示：
            - 什么叫"填涂"？填涂后黑色像素在整個区域中占多大比例？
            - 未填涂区域和填涂区域的像素分布有什么本质差异？
            - 如果有噪点（比如纸张褶皱、墨迹渗透），单纯用二值化会有什么后果？
              形态学开运算帮我们过滤掉什么类型的噪声？
            - 为什么需要 OTSU 而不是固定阈值？固定阈值（比如127）在这类图像上可能出什么错？
            - 多选检测的核心问题：如果学生同时涂了 A 和 C，那这两个气泡的填充率会怎么变化？
              用一个阈值（比如 0.06）能区分"涂了一个"和"涂了两个"吗？
              两个阈值（base threshold 和 multi threshold）分别负责什么判断？
        """
        raise NotImplementedError("TODO: 请实现 _analyze_zones() 方法")

    def _detect_fill_start(self, gray):
        """通过水平投影频带检测填涂区域的起始行。

        Args:
            gray: ndarray, 灰度图像

        Returns:
            int: 填涂区域起始行 y 坐标

        思路提示：
            - 什么是"水平投影"？把二维图像沿水平方向压扁后得到的一维信号能告诉我们什么？
            - 填涂区域和空白区域在投影信号上有什么区别？
            - 如果图像中有多个填涂区域（比如选择题的多个气泡行），投影信号会有几个"波峰"？
            - 如何过滤掉过窄的噪声频带？过窄的频带通常对应什么（气泡线？噪点？）？
            - 为什么最后取的是最大频带的"起始行"而不是中心？这是故意的吗？
        """
        raise NotImplementedError("TODO: 请实现 _detect_fill_start() 方法")

    def recognize(self, image, options=None):
        """识别填涂区域，返回选中的选项标签。多选或未填涂时返回 None。

        Args:
            image: ndarray, 单个单元格的 BGR 图像
            options: list[str], 选项标签，默认使用 self.option_labels

        Returns:
            str or None: 选中的选项（如 'A'），多选或未填涂返回 None
        """
        raise NotImplementedError("TODO: 请实现 recognize() 方法")

    def recognize_with_viz(self, image, options=None):
        """识别填涂区域并返回带标注的可视化图像。

        Args:
            image: ndarray, 单个单元格的 BGR 图像
            options: list[str], 选项标签，默认使用 self.option_labels

        Returns:
            tuple: (result, viz_image, zone_fills)
                - result: str or None, 识别结果
                - viz_image: ndarray, BGR 可视化图像（标注各区域填充率）
                - zone_fills: list[float], 各区域填充率

        思路提示：
            - 可视化图像的作用是什么？（不是给程序看，是给人调试用的）
            - 怎么在图上同时展示"每个区域的填充率数值"和"程序认为的答案"？
            - 如果程序判断是多选，可视化图像应该怎么让用户一眼看出来？
        """
        raise NotImplementedError("TODO: 请实现 recognize_with_viz() 方法")

    # ── 以下为已实现的工具方法，子类可直接调用 ──────────────────

    def _trim_margin(self, image):
        """裁剪图像四周边距。

        Args:
            image: ndarray, 输入图像

        Returns:
            ndarray: 裁剪后的图像
        """
        m = self.margin
        if m <= 0:
            return image
        h, w = image.shape[:2]
        if h <= 2 * m or w <= 2 * m:
            return image
        return image[m:-m, m:-m]
