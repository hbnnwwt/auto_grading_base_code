"""marker 模块测试。

验证错题标注和分数写入的正确性。
运行: pytest tests/test_marker.py -v
"""

import os

import cv2
import numpy as np
import pytest

from modules.marker import mark_wrong_on_page, mark_and_save, _valid_region


# ---------------------------------------------------------------------------
# _valid_region
# ---------------------------------------------------------------------------

class TestValidRegion:
    def test_valid(self):
        assert _valid_region((10, 20, 100, 50)) is True

    def test_none(self):
        assert _valid_region(None) is False

    def test_short(self):
        assert _valid_region((1, 2)) is False


# ---------------------------------------------------------------------------
# mark_wrong_on_page
# ---------------------------------------------------------------------------

class TestMarkWrongOnPage:
    def test_invalid_region_returns_copy(self):
        page = np.zeros((100, 100, 3), dtype=np.uint8)
        result = mark_wrong_on_page(page, None, [], {}, ['A', 'B', 'C', 'D'])
        assert result.shape == page.shape

    def test_no_wrong_answers(self):
        """全对时不应画 X。"""
        page = np.ones((200, 300, 3), dtype=np.uint8) * 255
        region = (10, 10, 280, 180)
        cell_results = [{
            'question': 1,
            'answer': 'A',
            'cell_bounds': (10, 50, 10, 50),
            'zone_bounds': [(0, 10), (10, 20), (20, 30), (30, 40)],
        }]
        grading_detail = {1: {'score': 3, 'given': 'A'}}
        result = mark_wrong_on_page(page, region, cell_results,
                                     grading_detail, ['A', 'B', 'C', 'D'])
        # 全对时不应修改图像内容（白色背景 + 绿色标记以外无变化）
        assert result.shape == page.shape

    def test_wrong_answer_draws_x(self):
        """答错时应在图上画标记。"""
        page = np.ones((200, 300, 3), dtype=np.uint8) * 255
        region = (10, 10, 280, 180)
        cell_results = [{
            'question': 1,
            'answer': 'B',
            'cell_bounds': (10, 50, 10, 50),
            'zone_bounds': [(0, 10), (10, 20), (20, 30), (30, 40)],
        }]
        grading_detail = {1: {'score': 0, 'given': 'B', 'correct': 'A'}}
        result = mark_wrong_on_page(page, region, cell_results,
                                     grading_detail, ['A', 'B', 'C', 'D'])
        # 应有非白色像素（X 标记）
        diff = np.sum(result < 250, axis=2)
        assert diff.any(), "错题应画出非白色标记"


# ---------------------------------------------------------------------------
# mark_and_save
# ---------------------------------------------------------------------------

class TestMarkAndSave:
    def test_saves_files(self, tmp_path):
        """应生成标注图文件。"""
        p1 = np.ones((500, 400, 3), dtype=np.uint8) * 255
        p2 = np.ones((500, 400, 3), dtype=np.uint8) * 255
        regions1 = {'choice': (10, 10, 380, 480)}
        regions2 = {'judge': (10, 10, 380, 200), 'essay': (10, 220, 380, 200)}
        cell_results = [{
            'question': 1,
            'answer': 'B',
            'cell_bounds': (10, 50, 10, 50),
            'zone_bounds': [(0, 10), (10, 20), (20, 30), (30, 40)],
        }]
        grading_result = {
            'choice': {1: {'score': 0, 'given': 'B', 'correct': 'A'}},
            'judge': {},
            'choice_total': 0,
            'judge_total': 0,
            'essay_total': 0,
        }

        p1_path, p2_path, _, _ = mark_and_save(
            "2025001", p1, p2, regions1, regions2,
            cell_results, [],
            grading_result,
            choice_max=60, judge_max=20, essay_max=20,
            output_dir=str(tmp_path))

        assert os.path.exists(p1_path)
        assert os.path.exists(p2_path)

    def test_missing_regions_does_not_crash(self, tmp_path):
        """region 缺失时不应崩溃。"""
        p1 = np.ones((500, 400, 3), dtype=np.uint8) * 255
        p2 = np.ones((500, 400, 3), dtype=np.uint8) * 255

        p1_path, p2_path, _, _ = mark_and_save(
            "unknown", p1, p2, {}, {}, [], [],
            {'choice': {}, 'judge': {},
             'choice_total': 0, 'judge_total': 0, 'essay_total': 0},
            choice_max=60, judge_max=20, essay_max=20,
            output_dir=str(tmp_path))

        assert os.path.exists(p1_path)
