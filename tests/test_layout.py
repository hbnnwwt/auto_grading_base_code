"""版面分析模块测试。

验证 LayoutAnalyzer 的轮廓检测和版面分析。
运行: pytest tests/test_layout.py -v
"""

import cv2
import numpy as np
import pytest

from modules.layout import LayoutAnalyzer


# ---------------------------------------------------------------------------
# fixture: 模拟有 2 个水平区域的答题卡二值图
# ---------------------------------------------------------------------------

@pytest.fixture
def two_region_binary():
    """生成二值图：上下两个黑色矩形区域，模拟答题卡的两个区域框。"""
    h, w = 900, 600
    img = np.ones((h, w), dtype=np.uint8) * 255
    img[50:250, 50:550] = 0
    img[400:700, 50:550] = 0
    return img


@pytest.fixture
def two_region_color(two_region_binary):
    return cv2.cvtColor(two_region_binary, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# 轮廓检测
# ---------------------------------------------------------------------------

class TestDetectRegions:
    def test_finds_two_regions(self, two_region_binary):
        analyzer = LayoutAnalyzer()
        boxes, _, _ = analyzer._detect_regions(two_region_binary)
        assert len(boxes) == 2

    def test_boxes_sorted_by_y(self, two_region_binary):
        analyzer = LayoutAnalyzer()
        boxes, _, _ = analyzer._detect_regions(two_region_binary)
        ys = [b[1] for b in boxes]
        assert ys == sorted(ys)

    def test_area_filter(self):
        """面积太小的区域应被过滤。"""
        h, w = 900, 600
        img = np.ones((h, w), dtype=np.uint8) * 255
        img[50:250, 50:550] = 0
        img[300:310, 50:60] = 0
        analyzer = LayoutAnalyzer()
        boxes, _, _ = analyzer._detect_regions(img)
        assert len(boxes) == 1

    def test_max_two_boxes(self):
        """即使有 3+ 个区域，也只取最大的 2 个。"""
        h, w = 900, 600
        img = np.ones((h, w), dtype=np.uint8) * 255
        img[50:150, 50:550] = 0
        img[200:400, 50:550] = 0
        img[450:700, 50:550] = 0
        analyzer = LayoutAnalyzer()
        boxes, _, _ = analyzer._detect_regions(img)
        assert len(boxes) == 2


# ---------------------------------------------------------------------------
# LayoutAnalyzer.analyze（完整版面分析）
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_required_keys(self, two_region_color, two_region_binary):
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(two_region_color, two_region_binary)
        for key in ['student_id', 'choice', 'judge', 'essay', 'image_size']:
            assert key in result

    def test_page1_assigns_regions(self, two_region_color, two_region_binary):
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(two_region_color, two_region_binary, page=1)
        assert result['student_id'] is not None
        assert result['choice'] is not None
        assert result['student_id'][1] < result['choice'][1]

    def test_page2_assigns_regions(self, two_region_color, two_region_binary):
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(two_region_color, two_region_binary, page=2)
        assert result['judge'] is not None
        assert result['essay'] is not None
        assert result['judge'][1] < result['essay'][1]

    def test_image_size_correct(self, two_region_color, two_region_binary):
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(two_region_color, two_region_binary)
        assert result['image_size'] == (600, 900)

    def test_fallback_on_empty_image(self):
        """空白图像应回退到固定比例。"""
        h, w = 900, 600
        img = np.ones((h, w), dtype=np.uint8) * 255
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(color, img, page=1)
        assert result['student_id'] is not None
        assert result['choice'] is not None
        assert result['student_id'][0] == 0

    def test_debug_image_generated(self, two_region_color, two_region_binary):
        analyzer = LayoutAnalyzer()
        analyzer.analyze(two_region_color, two_region_binary)
        assert analyzer.debug_image is not None
        assert analyzer.morph_image is not None
