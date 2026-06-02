"""空白答卷基准校准模块测试。

验证 compute_blank_baseline 和 compute_blank_baseline_multipage。
运行: pytest tests/test_blank_calibrator.py -v
"""

import cv2
import numpy as np
import pytest

from modules import blank_calibrator as bc_module
from modules.blank_calibrator import (
    compute_blank_baseline,
    compute_blank_baseline_multipage,
    _cell_zone_gray_stats,
    _detect_zone_bounds,
    _compute_region_baseline,
    _compute_section_baseline,
    save_baseline,
    load_baseline,
)


# ---------------------------------------------------------------------------
# fixture: 模拟答题卡区域（白底 + 黑色气泡行）
# ---------------------------------------------------------------------------

def _make_fake_region(h=600, w=700, rows=5, cols=4, zone_count=4,
                      bubble_fill=0.85):
    """生成一个模拟 choice/judge 区域的灰度图。

    每行每列有 zone_count 个暗色椭圆（模拟气泡），
    其余区域为白色背景带浅灰网格线。
    """
    img = np.ones((h, w), dtype=np.uint8) * 255

    cell_h = h / rows
    cell_w = w / cols
    zone_w = cell_w / zone_count

    for row in range(rows):
        for col in range(cols):
            y0 = int(row * cell_h)
            y1 = int((row + 1) * cell_h)
            x0 = int(col * cell_w)
            x1 = int((col + 1) * cell_w)

            # 画浅灰网格线
            cv2.rectangle(img, (x0, y0), (x1, y1), 220, 1)

            # 在每个 zone 画暗色椭圆模拟气泡
            for z in range(zone_count):
                zx0 = int(x0 + z * zone_w + zone_w * 0.15)
                zx1 = int(x0 + (z + 1) * zone_w - zone_w * 0.15)
                zy0 = int(y0 + cell_h * 0.25)
                zy1 = int(y1 - cell_h * 0.25)
                cv2.ellipse(img,
                            ((zx0 + zx1) // 2, (zy0 + zy1) // 2),
                            ((zx1 - zx0) // 2, (zy1 - zy0) // 2),
                            0, 0, 360,
                            int(255 * (1 - bubble_fill)), -1)

    return img


@pytest.fixture
def fake_choice_region():
    return _make_fake_region(rows=5, cols=4, zone_count=4)


@pytest.fixture
def fake_judge_region():
    return _make_fake_region(rows=3, cols=4, zone_count=2, bubble_fill=0.70)


@pytest.fixture
def fake_page1_color(fake_choice_region):
    """模拟第1页：学号区 + 选择题区（上下排列）。"""
    h, w = 1100, 800
    img = np.ones((h, w), dtype=np.uint8) * 255

    # 学号区（上方黑色矩形框）
    img[50:250, 50:750] = 0
    # 填充内部为白色
    img[55:245, 55:745] = 255

    # 选择题区（下方）
    choice_h, choice_w = fake_choice_region.shape
    x0 = 50
    img[300:300 + choice_h, x0:x0 + choice_w] = fake_choice_region

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def fake_page1_binary(fake_page1_color):
    gray = cv2.cvtColor(fake_page1_color, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


@pytest.fixture
def fake_page2_color(fake_judge_region):
    """模拟第2页：判断题区 + 简答题区（上下排列）。"""
    h, w = 1100, 800
    img = np.ones((h, w), dtype=np.uint8) * 255

    # 判断题区（上方）
    judge_h, judge_w = fake_judge_region.shape
    x0 = 50
    img[50:50 + judge_h, x0:x0 + judge_w] = fake_judge_region

    # 简答题区（下方黑色矩形框）
    img[500:900, 50:750] = 0
    img[505:895, 55:745] = 255

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def fake_page2_binary(fake_page2_color):
    gray = cv2.cvtColor(fake_page2_color, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# 内部 helper 测试
# ---------------------------------------------------------------------------

class TestCellZoneGrayStats:
    def test_returns_list_of_dicts(self):
        cell = np.ones((100, 200), dtype=np.uint8) * 200
        stats = _cell_zone_gray_stats(cell, [(0, 50), (50, 100), (100, 150), (150, 200)])
        assert isinstance(stats, list)
        assert len(stats) == 4
        for s in stats:
            assert "mean" in s
            assert "std" in s


class TestDetectZoneBounds:
    def test_returns_relative_bounds(self):
        cell = np.ones((100, 200), dtype=np.uint8) * 255
        # 在中间画两个暗带
        cell[:, 40:60] = 50
        cell[:, 140:160] = 50
        bounds = _detect_zone_bounds(cell, zone_count=2)
        assert len(bounds) == 2
        for b in bounds:
            assert 0.0 <= b[0] <= 1.0
            assert 0.0 <= b[1] <= 1.0
            assert b[0] < b[1]


class TestComputeRegionBaseline:
    def test_choice_questions_count(self, fake_choice_region):
        gray = fake_choice_region
        fill_start = 0
        questions = _compute_region_baseline(
            gray, fill_start, rows_n=5, cols_n=4,
            question_start=1, question_count=20, zone_count=4)
        assert len(questions) == 20
        for q in range(1, 21):
            assert str(q) in questions
            assert "zones" in questions[str(q)]
            assert "zone_bounds_rel" in questions[str(q)]
            assert len(questions[str(q)]["zones"]) == 4

    def test_judge_questions_count(self, fake_judge_region):
        gray = fake_judge_region
        fill_start = 0
        questions = _compute_region_baseline(
            gray, fill_start, rows_n=3, cols_n=4,
            question_start=21, question_count=10, zone_count=2)
        assert len(questions) == 10
        for q in range(21, 31):
            assert str(q) in questions
            assert len(questions[str(q)]["zones"]) == 2


# ---------------------------------------------------------------------------
# compute_blank_baseline 测试
# ---------------------------------------------------------------------------

class TestComputeBlankBaseline:
    def test_page1_choice(self, tmp_path, fake_page1_color):
        path = str(tmp_path / "page1.png")
        cv2.imwrite(path, fake_page1_color)

        result = compute_blank_baseline(path, page=1)
        assert "choice" in result
        assert len(result["choice"]["questions"]) == 20

    def test_page2_judge(self, tmp_path, fake_page2_color):
        path = str(tmp_path / "page2.png")
        cv2.imwrite(path, fake_page2_color)

        result = compute_blank_baseline(path, page=2)
        assert "judge" in result
        assert len(result["judge"]["questions"]) == 10

    def test_blank_image_fallback(self, tmp_path):
        """空白图像会回退到固定比例区域，仍能返回结果。"""
        h, w = 1100, 800
        blank = np.ones((h, w), dtype=np.uint8) * 255
        path = str(tmp_path / "blank.png")
        cv2.imwrite(path, blank)

        # LayoutAnalyzer 会回退到固定比例，不会抛异常
        result = compute_blank_baseline(path, page=1)
        assert "choice" in result
        assert len(result["choice"]["questions"]) == 20


# ---------------------------------------------------------------------------
# compute_blank_baseline_multipage 测试
# ---------------------------------------------------------------------------

class TestComputeBlankBaselineMultipage:
    def test_legacy_two_page(self, tmp_path, fake_page1_color, fake_page2_color):
        """无 _pages 配置时，回退到旧的两页逻辑。"""
        path1 = str(tmp_path / "page1.png")
        path2 = str(tmp_path / "page2.png")
        cv2.imwrite(path1, fake_page1_color)
        cv2.imwrite(path2, fake_page2_color)

        result = compute_blank_baseline_multipage([path1, path2])
        assert "choice" in result
        assert "judge" in result
        assert len(result["choice"]["questions"]) == 20
        assert len(result["judge"]["questions"]) == 10

    def test_with_pages_config(self, tmp_path, fake_page1_color, fake_page2_color,
                               monkeypatch):
        """有 _pages 配置时，按配置逐页匹配 section。"""
        original_layout = bc_module._LAYOUT.copy() if hasattr(bc_module, '_LAYOUT') else None

        layout_with_pages = {
            "choice": {"rows": 5, "cols": 4, "question_start": 1,
                       "question_count": 20},
            "judge": {"rows": 3, "cols": 4, "question_start": 21,
                      "question_count": 10},
            "_pages": [
                ["student_id", "choice"],
                ["judge", "essay"],
            ],
        }

        path1 = str(tmp_path / "page1.png")
        path2 = str(tmp_path / "page2.png")
        cv2.imwrite(path1, fake_page1_color)
        cv2.imwrite(path2, fake_page2_color)

        result = compute_blank_baseline_multipage([path1, path2], layout=layout_with_pages)
        assert "choice" in result
        assert "judge" in result
        assert len(result["choice"]["questions"]) == 20
        assert len(result["judge"]["questions"]) == 10

    def test_single_page_choice_only(self, tmp_path, fake_page1_color):
        """单页只含 choice。"""
        path1 = str(tmp_path / "page1.png")
        cv2.imwrite(path1, fake_page1_color)

        layout = {
            "choice": {"rows": 5, "cols": 4, "question_start": 1,
                       "question_count": 20},
            "_pages": [["student_id", "choice"]],
        }

        result = compute_blank_baseline_multipage([path1], layout=layout)
        assert "choice" in result
        assert "judge" not in result
        assert len(result["choice"]["questions"]) == 20

    def test_empty_paths_raises(self, tmp_path):
        """空路径列表应抛出 ValueError。"""
        with pytest.raises(ValueError):
            compute_blank_baseline_multipage([])

    def test_blank_image_fallback(self, tmp_path):
        """空白图像会回退到固定比例区域，仍能返回结果。"""
        h, w = 1100, 800
        blank = np.ones((h, w), dtype=np.uint8) * 255
        path = str(tmp_path / "blank.png")
        cv2.imwrite(path, blank)

        result = compute_blank_baseline_multipage([path])
        assert "choice" in result
        assert len(result["choice"]["questions"]) == 20

    def test_question_numbers_preserved(self, tmp_path, fake_page1_color, fake_page2_color):
        """题号应正确保留（choice 1-20, judge 21-30）。"""
        path1 = str(tmp_path / "page1.png")
        path2 = str(tmp_path / "page2.png")
        cv2.imwrite(path1, fake_page1_color)
        cv2.imwrite(path2, fake_page2_color)

        result = compute_blank_baseline_multipage([path1, path2])
        choice_qs = set(int(k) for k in result["choice"]["questions"].keys())
        judge_qs = set(int(k) for k in result["judge"]["questions"].keys())
        assert choice_qs == set(range(1, 21))
        assert judge_qs == set(range(21, 31))


# ---------------------------------------------------------------------------
# save_baseline / load_baseline 测试
# ---------------------------------------------------------------------------

class TestSaveLoadBaseline:
    def test_roundtrip(self, tmp_path):
        baseline = {
            "choice": {
                "questions": {
                    "1": {"zones": [{"mean": 200.0, "std": 10.0}]},
                }
            }
        }
        path = str(tmp_path / "baseline.json")
        save_baseline(baseline, path)
        loaded = load_baseline(path)
        assert loaded == baseline

    def test_load_missing_returns_none(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        assert load_baseline(path) is None
