"""版面分析多页适配测试。

验证 LayoutAnalyzer.analyze_multipage 的多页区域检测与映射。
运行: pytest tests/test_layout_multipage.py -v
"""

import cv2
import numpy as np
import pytest

from modules import layout as layout_module
from modules.layout import LayoutAnalyzer


# ---------------------------------------------------------------------------
# fixture: 模拟每页有 2 个黑色矩形区域的答题卡
# ---------------------------------------------------------------------------

def _make_page_binary(h=900, w=600, region_specs=None):
    """生成单页二值图：白色背景上按 region_specs 绘制黑色矩形。"""
    img = np.ones((h, w), dtype=np.uint8) * 255
    if region_specs is None:
        # 默认上下两个区域
        region_specs = [(50, 250, 50, 550), (400, 700, 50, 550)]
    for y1, y2, x1, x2 in region_specs:
        img[y1:y2, x1:x2] = 0
    return img


@pytest.fixture
def page1_binary():
    return _make_page_binary(region_specs=[(50, 250, 50, 550), (300, 750, 50, 550)])


@pytest.fixture
def page2_binary():
    return _make_page_binary(region_specs=[(60, 400, 50, 550), (450, 800, 50, 550)])


@pytest.fixture
def page1_color(page1_binary):
    return cv2.cvtColor(page1_binary, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def page2_color(page2_binary):
    return cv2.cvtColor(page2_binary, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Mock _pages 配置
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_pages_config(monkeypatch):
    """在每个测试前注入 _pages 配置，测试后自动恢复。"""
    original_layout = layout_module._LAYOUT.copy()
    layout_module._LAYOUT['_pages'] = [
        ['student_id', 'choice'],
        ['judge', 'essay'],
    ]
    yield
    layout_module._LAYOUT = original_layout


# ---------------------------------------------------------------------------
# analyze_multipage 测试
# ---------------------------------------------------------------------------

class TestAnalyzeMultipage:
    def test_returns_list_of_dicts(self, page1_color, page1_binary,
                                    page2_color, page2_binary):
        analyzer = LayoutAnalyzer()
        results = analyzer.analyze_multipage(
            [page1_color, page2_color],
            [page1_binary, page2_binary]
        )
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, dict)

    def test_page1_maps_student_id_and_choice(self, page1_color, page1_binary,
                                               page2_color, page2_binary):
        analyzer = LayoutAnalyzer()
        results = analyzer.analyze_multipage(
            [page1_color, page2_color],
            [page1_binary, page2_binary]
        )
        r1 = results[0]
        assert r1['student_id'] is not None
        assert r1['choice'] is not None
        assert r1['student_id'][1] < r1['choice'][1]

    def test_page2_maps_judge_and_essay(self, page1_color, page1_binary,
                                         page2_color, page2_binary):
        analyzer = LayoutAnalyzer()
        results = analyzer.analyze_multipage(
            [page1_color, page2_color],
            [page1_binary, page2_binary]
        )
        r2 = results[1]
        assert r2['judge'] is not None
        assert r2['essay'] is not None
        assert r2['judge'][1] < r2['essay'][1]

    def test_boxes_sorted_by_y(self, page1_color, page1_binary,
                                page2_color, page2_binary):
        analyzer = LayoutAnalyzer()
        results = analyzer.analyze_multipage(
            [page1_color, page2_color],
            [page1_binary, page2_binary]
        )
        for r in results:
            boxes = r['boxes']
            ys = [b[1] for b in boxes]
            assert ys == sorted(ys)

    def test_image_size_in_result(self, page1_color, page1_binary,
                                   page2_color, page2_binary):
        analyzer = LayoutAnalyzer()
        results = analyzer.analyze_multipage(
            [page1_color, page2_color],
            [page1_binary, page2_binary]
        )
        assert results[0]['image_size'] == (600, 900)
        assert results[1]['image_size'] == (600, 900)

    def test_fallback_when_regions_missing(self, monkeypatch):
        """当检测不到足够区域时，应回退到固定比例。"""
        h, w = 900, 600
        blank = np.ones((h, w), dtype=np.uint8) * 255
        blank_color = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)

        analyzer = LayoutAnalyzer()
        results = analyzer.analyze_multipage([blank_color], [blank])
        r1 = results[0]
        assert r1['student_id'] is not None
        assert r1['choice'] is not None
        assert r1['student_id'][0] == 0

    def test_fallback_without_pages_config(self, monkeypatch,
                                            page1_color, page1_binary,
                                            page2_color, page2_binary):
        """当 _pages 不存在时，应回退到单页 analyze() 行为。"""
        monkeypatch.delitem(layout_module._LAYOUT, '_pages', raising=False)

        analyzer = LayoutAnalyzer()
        results = analyzer.analyze_multipage(
            [page1_color, page2_color],
            [page1_binary, page2_binary]
        )
        assert len(results) == 2
        assert results[0]['student_id'] is not None
        assert results[0]['choice'] is not None
        assert results[1]['judge'] is not None
        assert results[1]['essay'] is not None

    def test_debug_image_generated(self, page1_color, page1_binary,
                                    page2_color, page2_binary):
        analyzer = LayoutAnalyzer()
        analyzer.analyze_multipage(
            [page1_color, page2_color],
            [page1_binary, page2_binary]
        )
        assert analyzer.debug_image is not None
        assert analyzer.morph_image is not None
