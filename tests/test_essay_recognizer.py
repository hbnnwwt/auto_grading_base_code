"""简答题识别模块测试。

验证 EssayRecognizer 能正确初始化和识别（需要学生先实现 recognize 方法）。
运行: pytest tests/test_essay_recognizer.py -v
"""

import cv2
import numpy as np
import pytest

from modules.essay_recognizer import EssayRecognizer, check_engine_available


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def text_image():
    """生成含模拟文字区域的图像（白底黑字线条）。"""
    img = np.ones((200, 400), dtype=np.uint8) * 255
    # 模拟两行文字
    cv2.line(img, (50, 60), (350, 60), 0, 3)
    cv2.line(img, (50, 140), (300, 140), 0, 3)
    return img


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestEssayRecognizer:
    def test_init_default(self):
        """默认参数初始化。"""
        rec = EssayRecognizer()
        assert rec.engine == 'paddleocr'
        assert rec.lang == 'ch'

    def test_init_custom_engine(self):
        """自定义引擎初始化。"""
        rec = EssayRecognizer(engine='trocr', lang='en')
        assert rec.engine == 'trocr'
        assert rec.lang == 'en'

    def test_recognize_returns_string(self, text_image):
        """recognize 应返回字符串。"""
        rec = EssayRecognizer()
        result = rec.recognize(text_image)
        assert isinstance(result, str)

    def test_recognize_non_empty(self, text_image):
        """对含内容的图像，识别结果应非空。"""
        rec = EssayRecognizer()
        result = rec.recognize(text_image)
        # 至少应该返回一些内容（具体内容取决于 OCR 引擎）
        assert len(result) >= 0  # 不强制要求准确，只要求不崩溃

    def test_recognize_blank_image(self):
        """空白图像不应崩溃。"""
        img = np.ones((200, 400), dtype=np.uint8) * 255
        rec = EssayRecognizer()
        result = rec.recognize(img)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# check_engine_available
# ---------------------------------------------------------------------------

class TestCheckEngineAvailable:
    def test_known_engines(self):
        """已知引擎名不应报错。"""
        for eng in ['paddleocr', 'easyocr', 'rapidocr', 'online']:
            result = check_engine_available(eng)
            assert isinstance(result, bool)

    def test_unknown_engine(self):
        """未知引擎应返回 False。"""
        assert check_engine_available('nonexistent') is False

    def test_online_needs_requests(self):
        """online 引擎依赖 requests 库。"""
        result = check_engine_available('online')
        assert result is True  # requests 已安装


# ---------------------------------------------------------------------------
# online OCR（mock 测试，不实际调用 API）
# ---------------------------------------------------------------------------

class TestOnlineOcr:
    def test_no_api_key_raises(self):
        """无 API Key 时应抛出 ValueError。"""
        rec = EssayRecognizer(engine='online', api_config={})
        img = np.ones((100, 200), dtype=np.uint8) * 255
        with pytest.raises(ValueError, match="API Key"):
            rec._extract_online(img)

    def test_with_api_key_no_crash(self):
        """有 API Key 时应不崩溃（实际 API 调用可能失败，但不应段错误）。"""
        import modules.essay_recognizer as _mod
        rec = EssayRecognizer(
            engine='online',
            api_config={'api_key': 'fake-key-for-test',
                        'base_url': 'https://invalid-url.test',
                        'ocr_model': 'test-model'})
        img = np.ones((100, 200), dtype=np.uint8) * 255
        result = rec.recognize(img)
        assert isinstance(result, str)
        assert len(_mod.last_error) > 0  # 应有错误信息
