"""pipeline 模块测试。

验证识别管线的各函数对无效输入和有效 region 的处理。
运行: pytest tests/test_pipeline.py -v
"""

import numpy as np
import pytest

from modules.pipeline import (
    _valid_region,
    extract_student_id,
    recognize_choices,
    recognize_judges,
    recognize_essay,
)


# ---------------------------------------------------------------------------
# _valid_region
# ---------------------------------------------------------------------------

class TestValidRegion:
    def test_valid_tuple(self):
        assert _valid_region((10, 20, 100, 50)) is True

    def test_valid_list(self):
        assert _valid_region([10, 20, 100, 50]) is True

    def test_none(self):
        assert _valid_region(None) is False

    def test_empty_dict(self):
        assert _valid_region({}) is False

    def test_short_tuple(self):
        assert _valid_region((10, 20)) is False

    def test_string_in_tuple(self):
        assert _valid_region((10, 20, "a", 50)) is False

    def test_zero_values(self):
        assert _valid_region((0, 0, 0, 0)) is True


# ---------------------------------------------------------------------------
# extract_student_id
# ---------------------------------------------------------------------------

class TestExtractStudentId:
    def test_no_region(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        assert extract_student_id(img, {}) is None

    def test_none_region(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        assert extract_student_id(img, {'student_id': None}) is None

    def test_invalid_region_type(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        assert extract_student_id(img, {'student_id': True}) is None

    def test_short_region(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        assert extract_student_id(img, {'student_id': (10, 20)}) is None


# ---------------------------------------------------------------------------
# recognize_choices
# ---------------------------------------------------------------------------

class TestRecognizeChoices:
    def test_no_region(self):
        img = np.zeros((500, 400), dtype=np.uint8)
        result = recognize_choices(img, {})
        assert result == {}

    def test_none_region(self):
        img = np.zeros((500, 400), dtype=np.uint8)
        result = recognize_choices(img, {'choice': None})
        assert result == {}

    def test_return_details_no_region(self):
        img = np.zeros((500, 400), dtype=np.uint8)
        result = recognize_choices(img, {}, return_details=True)
        assert isinstance(result, tuple)
        assert result[0] == {}
        assert result[1] == []

    def test_valid_region_returns_dict(self):
        img = np.ones((500, 400), dtype=np.uint8) * 255
        result = recognize_choices(img, {'choice': (10, 10, 380, 480)})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# recognize_judges
# ---------------------------------------------------------------------------

class TestRecognizeJudges:
    def test_no_region(self):
        img = np.zeros((300, 400), dtype=np.uint8)
        result = recognize_judges(img, {})
        assert result == {}

    def test_none_region(self):
        img = np.zeros((300, 400), dtype=np.uint8)
        result = recognize_judges(img, {'judge': None})
        assert result == {}

    def test_return_details_no_region(self):
        img = np.zeros((300, 400), dtype=np.uint8)
        result = recognize_judges(img, {}, return_details=True)
        assert isinstance(result, tuple)
        assert result[0] == {}
        assert result[1] == []


# ---------------------------------------------------------------------------
# recognize_essay
# ---------------------------------------------------------------------------

class TestRecognizeEssay:
    def test_no_region(self):
        img = np.zeros((300, 400), dtype=np.uint8)
        result = recognize_essay(img, {})
        assert result == ""

    def test_none_region(self):
        img = np.zeros((300, 400), dtype=np.uint8)
        result = recognize_essay(img, {'essay': None})
        assert result == ""

    def test_invalid_region_returns_empty(self):
        img = np.zeros((300, 400), dtype=np.uint8)
        result = recognize_essay(img, {'essay': "bad"})
        assert result == ""
