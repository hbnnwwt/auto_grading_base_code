import cv2
import numpy as np


class StudentIdRecognizer:
    """学号气泡填涂识别模块。

    调用者（pipeline.py）使用方式：
        rec = StudentIdRecognizer(digit_count=10, threshold=0.3)
        student_id = rec.recognize(roi)

    目标：从学号填涂区域图像中识别出学生填涂的 10 位数字。
    """

    def __init__(self, digit_count=10, threshold=0.2, margin=0,
                 canny_low=50, canny_high=150):
        """
        Args:
            digit_count: 学号位数，默认10
            threshold: 填充率阈值，用于判断格子是否被填涂
            margin: 裁剪边距
            canny_low: Canny 边缘检测低阈值
            canny_high: Canny 边缘检测高阈值
        """
        self.digit_count = digit_count
        self.threshold = threshold
        self.margin = margin
        self.canny_low = canny_low
        self.canny_high = canny_high

    def recognize(self, roi):
        """识别学号。

        Args:
            roi: ndarray (H, W, 3), 学号区域的 BGR 图像

        Returns:
            str: 学号字符串，如 "2025811008"
                 无法识别的数字用 '?' 替代

        思路提示：
            - 学号区域的网格线是什么颜色的？它和我们想识别的数字颜色有何不同？
            - 如何把网格线去掉，只留下数字（或数字的痕迹）？
            - 填涂后格子的黑色像素占比大概在什么范围？未填涂的格子呢？
            - 每列有 11 行（0-10），其中哪一行永远不会被填涂？为什么？
            - 如果某个数字完全没被填涂，程序应该返回什么？
        """
        raise NotImplementedError("TODO: 请实现学号识别 recognize() 方法")

    def recognize_with_viz(self, roi):
        """识别学号并返回可视化结果。

        Args:
            roi: ndarray (H, W, 3), 学号区域的 BGR 图像

        Returns:
            tuple: (student_id, viz_image, digit_details)
                - student_id: str, 识别到的学号
                - viz_image: ndarray, 标注了网格和识别结果的可视化图像
                - digit_details: list[dict], 每位数字的详细信息
        """
        raise NotImplementedError("TODO: 请实现 recognize_with_viz() 方法")
