import json
import os

import cv2
import numpy as np

# 直接加载配置文件，避免从 pipeline 导入导致循环引用
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
    """答题卡版面分析模块：基于轮廓检测定位各题型区域。

    答题卡的每个区域（学号、选择题、判断题、简答题）都有可见的矩形边框，
    通过形态学闭运算合并边框与内部内容为一个连通区域，再用轮廓检测定位。
    """

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
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.kernel_ratio = kernel_ratio
        self._debug_image = None
        self._morph_image = None

    @staticmethod
    def _extract_vertical_edges(gray):
        """提取垂直线（含噪声过滤），委托给 ImagePreprocessor。"""
        from modules.preprocess import ImagePreprocessor
        return ImagePreprocessor.extract_vertical_edges(gray)

    @staticmethod
    def _extract_horizontal_edges(gray):
        """提取水平线（含噪声过滤），委托给 ImagePreprocessor。"""
        from modules.preprocess import ImagePreprocessor
        return ImagePreprocessor.extract_horizontal_edges(gray)

    def _detect_regions(self, binary, gray_enhanced=None):
        """从二值图中检测带边框的区域。

        流程：
        1. 反转二值图
        2. 形态学闭运算(水平扁核)连接水平断线
        3. Sobel 提取垂直线(补充垂直断线)
        4. 闭运算结果 + 垂直线叠加
        5. 外轮廓检测 → 面积过滤 → 取最大2个

        自适应策略：优先小核(避免联通相邻区域)，检测不到足够区域时逐步增大核。
        最后回退到不做闭运算(用原始反转图)。

        启发式问题：
        - 形态学中的"闭运算"（MORPH_CLOSE）有什么效果？
          为什么它能帮助连接断开的边框线？
        - 如果闭运算的核太大，可能会把上下相邻的两个区域连成一个，
          如何避免这种情况？（提示：考虑核的形状）
        - Sobel 算子提取的垂直线/水平线，如何与闭运算结果互补？
        - 检测到的轮廓很多，面积过滤和长宽比过滤分别排除了什么噪声？
        """
        raise NotImplementedError("请实现区域检测")

    def _filter_boxes(self, contours, total_area):
        """从轮廓列表中过滤出符合面积和长宽比的候选框。"""
        min_area = total_area * self.min_area_ratio
        max_area = total_area * self.max_area_ratio
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / max(bh, 1)
            if aspect < 0.3 or aspect > 5.0:
                continue
            boxes.append((x, y, bw, bh))
        return boxes

    def analyze(self, image, binary, page=1):
        """分析答题卡版面，返回各题型的区域坐标。

        优先使用轮廓检测，失败时回退到固定比例。

        Args:
            image: 原始图像（BGR）
            binary: 二值化图像
            page: 1 = 第1页（学号+选择题），2 = 第2页（判断题+简答题）

        启发式问题：
        - 检测到的区域框如何与"学号区"、"选择题区"等语义对应？
          第1页和第2页的区域排列顺序有什么不同？
        - 如果轮廓检测失败（比如拍照时光线不均导致边框不清晰），
          如何回退到基于图像比例的固定分割？
        """
        raise NotImplementedError("请实现版面分析")

    def analyze_multipage(self, images, binaries):
        """多页答题卡版面分析，返回每页的区域坐标列表。

        根据 sheet_layout.json 中的 _pages 配置，按页面顺序匹配检测到的区域。
        若 _pages 不存在，则回退到单页 analyze() 行为。

        Args:
            images: 原始图像列表（BGR），每页一张
            binaries: 二值化图像列表，与 images 一一对应

        Returns:
            list: 每页一个 result dict，包含 student_id/choice/judge/essay/
                  image_size/boxes 等键

        启发式问题：
        - 多页模式下，如何根据配置知道"第1页应该有学号+选择题，第2页应该有判断题+简答题"？
        - 如果某页检测到的区域数量不足，如何优雅地回退到固定比例？
        """
        raise NotImplementedError("请实现多页版面分析")

    def _fallback_region(self, h, w, ratio_range):
        start, end = ratio_range
        return (0, int(h * start), w, int(h * (end - start)))

    def _build_debug_image(self, image, boxes, all_contours):
        viz = image.copy()
        cv2.drawContours(viz, all_contours, -1, (180, 180, 180), 1)

        names = {0: '学号', 1: '选择题', 2: '判断题', 3: '简答题'}
        colors = {
            0: (0, 200, 0),
            1: (255, 100, 0),
            2: (0, 150, 255),
            3: (200, 0, 200),
        }
        for i, (x, y, bw, bh) in enumerate(boxes):
            color = colors.get(i, (200, 200, 200))
            name = names.get(i, f'区域{i}')
            cv2.rectangle(viz, (x, y), (x + bw, y + bh), color, 3)
            cv2.putText(viz, name, (x + 5, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return viz

    @property
    def debug_image(self):
        """最近的调试可视化图像（原图 + 检测框）。"""
        return self._debug_image

    @property
    def morph_image(self):
        """形态学闭运算后的图像（用于展示中间步骤）。"""
        return self._morph_image
