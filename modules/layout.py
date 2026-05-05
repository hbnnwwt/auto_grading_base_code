"""答题卡版面分析模块：基于轮廓检测定位各题型区域。

调用者（pipeline.py）使用方式：
    analyzer = LayoutAnalyzer()
    regions = analyzer.analyze(corrected_image, binary, page=1)
    # regions = {'student_id': (x,y,w,h), 'choice': (x,y,w,h), ...}

功能：
    - 第1页：检测学号区域 + 选择题区域
    - 第2页：检测判断题区域 + 简答题区域
    - 优先使用轮廓检测，失败时回退到固定比例（fallback）

参考: week05 形态学运算、week06 轮廓检测、连通域分析
"""

import json
import os

import cv2
import numpy as np

# 加载配置文件（避免循环引用）
_LAYOUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "config", "sheet_layout.json")
if os.path.exists(_LAYOUT_PATH):
    with open(_LAYOUT_PATH, "r", encoding="utf-8") as _f:
        _LAYOUT = json.load(_f)
else:
    _LAYOUT = {
        "layout": {
            "page1_fallback": {"student_id": [0.06, 0.26], "choice": [0.28, 0.80]},
            "page2_fallback": {"judge": [0.06, 0.46], "essay": [0.50, 0.90]},
        },
    }


class LayoutAnalyzer:
    """答题卡版面区域定位器。"""

    # fallback 比率（当轮廓检测失败时使用）
    _layout_cfg = _LAYOUT.get('layout', {})
    PAGE1_FALLBACK = {
        k: tuple(v) for k, v in _layout_cfg.get('page1_fallback', {
            'student_id': [0.06, 0.26], 'choice': [0.28, 0.80]
        }).items()
    }
    PAGE2_FALLBACK = {
        k: tuple(v) for k, v in _layout_cfg.get('page2_fallback', {
            'judge': [0.06, 0.46], 'essay': [0.50, 0.90]
        }).items()
    }

    def __init__(self, min_area_ratio=0.03, max_area_ratio=0.92,
                 kernel_ratio=0.02):
        """
        Args:
            min_area_ratio: 最小有效轮廓面积（占图像总面积的比例）
            max_area_ratio: 最大有效轮廓面积
            kernel_ratio: 形态学闭运算核大小（占图像高度的比例）
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.kernel_ratio = kernel_ratio
        self._debug_image = None
        self._morph_image = None

    def _detect_regions(self, binary):
        """从二值图中检测带边框的区域轮廓。

        Args:
            binary: ndarray, 二值化图像（255=白/背景，0=黑/内容）

        Returns:
            tuple: (boxes, morph, all_contours)
                - boxes: list[tuple], [(x, y, w, h), ...]，最多2个最大区域
                - morph: ndarray, 形态学闭运算后的图像（用于调试）
                - all_contours: list, 所有检测到的外轮廓（用于调试）

        思路提示：
            - 答题卡页面的边框是什么颜色的？在二值图上它是什么灰度值？
              如果直接找轮廓，为什么需要先反转图像？
            - 形态学闭运算在这里解决了什么问题？答题卡打印/扫描后边框可能有哪些缺陷？
            - 为什么用 RETR_EXTERNAL 而不是 RETR_LIST 或 RETR_TREE？
              我们只想找什么类型的轮廓？
            - 过滤轮廓面积时，上限（max_area_ratio）和下限（min_area_ratio）
              分别帮我们排除了哪些误检？
            - 宽高比过滤的根据是什么？答题卡的各个区域在正常拍摄角度下大概是什么形状？
            - 最后为什么要按 y 坐标排序而不是按面积排序？
              （提示：学号在图片上方，选择题在下方，这个顺序是固定的吗？）
        """
        raise NotImplementedError("TODO: 请实现 _detect_regions() 方法")

    def analyze(self, image, binary, page=1):
        """分析答题卡版面，返回各题型的区域坐标。

        Args:
            image: ndarray, 旋转校正后的 BGR 彩色图像
            binary: ndarray, 二值化图像
            page: int, 1=第1页（学号+选择题），2=第2页（判断题+简答）

        Returns:
            dict: {
                'student_id': (x, y, w, h) or None,
                'choice': (x, y, w, h) or None,
                'judge': (x, y, w, h) or None,
                'essay': (x, y, w, h) or None,
                'image_size': (w, h),
            }

        思路提示：
            - 第1页和第2页的区域排列有什么规律？y坐标大小关系是什么？
            - 当轮廓检测找到的区域不足2个时，说明什么？
              这时候 fallback 比率的假设是什么（提示：和答题卡的标准布局有关）？
            - 为什么 fallback 要用图像高度（h）的比例而不是绝对像素值？
              这解决了什么问题？
        """
        raise NotImplementedError("TODO: 请实现 analyze() 方法")
