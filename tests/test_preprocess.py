"""图像预处理模块测试。

验证 ImagePreprocessor 的各类方法是否正常工作。
运行: pytest tests/test_preprocess.py -v
"""

import cv2
import numpy as np
import pytest

from modules.preprocess import ImagePreprocessor


# ---------------------------------------------------------------------------
# fixture: 构造一张模拟答题卡图像（白底黑框 + 黑色填涂区域）
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_answer_sheet():
    """生成一张 800x1100 的模拟答题卡图像。"""
    img = np.ones((1100, 800), dtype=np.uint8) * 255
    # 画几个黑色矩形模拟填涂气泡
    cv2.rectangle(img, (100, 100), (150, 150), 0, -1)
    cv2.rectangle(img, (200, 100), (250, 150), 0, -1)
    return img


@pytest.fixture
def fake_color_sheet():
    """彩色版模拟答题卡。"""
    gray = np.ones((600, 800), dtype=np.uint8) * 255
    cv2.rectangle(gray, (50, 50), (100, 100), 0, -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# ImagePreprocessor.load
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_valid_image(self, tmp_path, fake_answer_sheet):
        """能正常加载存在的图片文件。"""
        path = str(tmp_path / "sheet.png")
        cv2.imwrite(path, fake_answer_sheet)

        pre = ImagePreprocessor()
        img = pre.load(path)
        assert img is not None
        assert img.shape[:2] == fake_answer_sheet.shape[:2]

    def test_load_nonexistent_raises(self):
        """加载不存在的文件应抛出 FileNotFoundError。"""
        pre = ImagePreprocessor()
        with pytest.raises(FileNotFoundError):
            pre.load("nonexistent_file.png")


# ---------------------------------------------------------------------------
# ImagePreprocessor.resize
# ---------------------------------------------------------------------------

class TestResize:
    def test_no_resize_when_target_none(self, fake_answer_sheet):
        """target_size=None 时不做缩放。"""
        pre = ImagePreprocessor(target_size=None)
        out = pre.resize(fake_answer_sheet)
        assert out.shape == fake_answer_sheet.shape

    def test_resize_to_target(self, fake_answer_sheet):
        """缩放到指定尺寸。"""
        pre = ImagePreprocessor(target_size=(400, 550))
        out = pre.resize(fake_answer_sheet)
        assert out.shape[:2] == (550, 400)


# ---------------------------------------------------------------------------
# ImagePreprocessor.denoise
# ---------------------------------------------------------------------------

class TestDenoise:
    @pytest.mark.parametrize("method", ["gaussian", "median", "bilateral"])
    def test_denoise_returns_grayscale(self, method, fake_color_sheet):
        """各种去噪方法均返回灰度图。"""
        pre = ImagePreprocessor(denoise_method=method)
        out = pre.denoise(fake_color_sheet)
        assert len(out.shape) == 2  # 灰度图

    def test_denoise_grayscale_input(self, fake_answer_sheet):
        """输入灰度图也能正常处理。"""
        pre = ImagePreprocessor(denoise_method="gaussian")
        out = pre.denoise(fake_answer_sheet)
        assert out.shape == fake_answer_sheet.shape


# ---------------------------------------------------------------------------
# ImagePreprocessor.enhance
# ---------------------------------------------------------------------------

class TestEnhance:
    @pytest.mark.parametrize("method", ["clahe", "histeq", "gamma"])
    def test_enhance_returns_uint8(self, method, fake_answer_sheet):
        """各种增强方法返回 uint8 灰度图。"""
        pre = ImagePreprocessor(enhance_method=method)
        out = pre.enhance(fake_answer_sheet)
        assert out.dtype == np.uint8
        assert len(out.shape) == 2


# ---------------------------------------------------------------------------
# ImagePreprocessor.binarize
# ---------------------------------------------------------------------------

class TestBinarize:
    @pytest.mark.parametrize("method", ["otsu", "adaptive", "simple"])
    def test_binarize_output_values(self, method, fake_answer_sheet):
        """二值化输出应只包含 0 和 255。"""
        pre = ImagePreprocessor(binarize_method=method)
        out = pre.binarize(fake_answer_sheet)
        unique = np.unique(out)
        assert all(v in [0, 255] for v in unique)


# ---------------------------------------------------------------------------
# ImagePreprocessor.process（完整流水线）
# ---------------------------------------------------------------------------

class TestProcess:
    def test_pipeline_returns_four_outputs(self, fake_color_sheet):
        """process 应返回 4 个 ndarray: image, gray, enhanced, binary。"""
        pre = ImagePreprocessor()
        image, gray, enhanced, binary = pre.process(fake_color_sheet)
        assert image is not None
        assert gray is not None
        assert enhanced is not None
        assert binary is not None

    def test_binary_is_binary(self, fake_color_sheet):
        """最终二值化图只含 0 和 255。"""
        pre = ImagePreprocessor()
        _, _, _, binary = pre.process(fake_color_sheet)
        unique = np.unique(binary)
        assert all(v in [0, 255] for v in unique)

    def test_pipeline_with_resize(self, fake_color_sheet):
        """process 含 resize 时尺寸正确。"""
        pre = ImagePreprocessor(target_size=(400, 300))
        image, gray, enhanced, binary = pre.process(fake_color_sheet)
        assert image.shape[:2] == (300, 400)
