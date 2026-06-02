import base64
import importlib.util

import cv2
import numpy as np

from modules.defaults import DEFAULT_BASE_URL, DEFAULT_OCR_MODEL

SUPPORTED_ENGINES = ['paddleocr', 'easyocr', 'rapidocr', 'online']


def _ensure_paddleocr():
    """懒加载 paddleocr（含 torch DLL 冲突 workaround）。"""
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        pass
    import paddleocr
    return paddleocr


def check_engine_available(engine):
    """检查 OCR 引擎是否已安装（不触发 import，避免 paddle/torch DLL 冲突）。"""
    if engine == 'paddleocr':
        return importlib.util.find_spec('paddleocr') is not None
    elif engine == 'easyocr':
        return importlib.util.find_spec('easyocr') is not None
    elif engine == 'rapidocr':
        return importlib.util.find_spec('rapidocr_onnxruntime') is not None
    elif engine == 'online':
        return importlib.util.find_spec('requests') is not None
    return False


class EssayRecognizer:
    """简答题文字识别模块。

    支持多引擎：paddleocr / easyocr / rapidocr / online(GLM-OCR)。
    """

    _instances = {}

    def __init__(self, engine='paddleocr', lang='ch', api_config=None,
                 cancel_check=None, max_image_side=2048):
        self.engine = engine
        self.lang = lang
        self.api_config = api_config or {}
        self.cancel_check = cancel_check
        self.max_image_side = max_image_side

    def _get_ocr(self):
        key = (self.engine, self.lang)
        if key not in self._instances:
            self._instances[key] = self._create_ocr()
        return self._instances[key]

    def _create_ocr(self):
        if self.engine == 'paddleocr':
            return self._create_paddleocr()
        elif self.engine == 'easyocr':
            return self._create_easyocr()
        elif self.engine == 'rapidocr':
            return self._create_rapidocr()
        raise ValueError(f"Unknown OCR engine: {self.engine}")

    # ── PaddleOCR ──────────────────────────────────────────
    def _create_paddleocr(self):
        paddleocr = _ensure_paddleocr()
        version = int(paddleocr.__version__.split('.')[0])
        if version >= 3:
            return paddleocr.PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=self.lang,
            )
        return paddleocr.PaddleOCR(use_angle_cls=True, lang=self.lang,
                                        show_log=False)

    def _extract_paddleocr(self, ocr, image):
        paddleocr = _ensure_paddleocr()
        version = int(paddleocr.__version__.split('.')[0])
        if version >= 3:
            return self._extract_paddleocr_v3(ocr, image)
        return self._extract_paddleocr_v2(ocr, image)

    def _extract_paddleocr_v3(self, ocr, image):
        """从 PaddleOCR v3+ 结果中提取文本。

        启发式问题：OCR 返回的多边形坐标如何转换成可用于排序的 Y 坐标？
        为什么要按 Y 坐标排序而不是直接使用返回顺序？
        """
        raise NotImplementedError("请实现 PaddleOCR v3 文本提取")

    def _extract_paddleocr_v2(self, ocr, image):
        """从 PaddleOCR v2 结果中提取文本。

        启发式问题：v2 与 v3 的返回数据结构有何不同？
        如何判断识别结果为空？
        """
        raise NotImplementedError("请实现 PaddleOCR v2 文本提取")

    # ── EasyOCR ────────────────────────────────────────────
    def _create_easyocr(self):
        import easyocr
        return easyocr.Reader([self.lang.replace('ch', 'ch_sim')])

    def _extract_easyocr(self, reader, image):
        """从 EasyOCR 结果中提取文本。

        启发式问题：不同 OCR 引擎返回的 bounding box 格式是否一致？
        如何编写与引擎无关的文本提取逻辑？
        """
        raise NotImplementedError("请实现 EasyOCR 文本提取")

    # ── RapidOCR ───────────────────────────────────────────
    def _create_rapidocr(self):
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()

    def _extract_rapidocr(self, engine, image):
        """从 RapidOCR 结果中提取文本。

        启发式问题：ONNXRuntime 引擎的输出格式与 PaddleOCR 有何异同？
        """
        raise NotImplementedError("请实现 RapidOCR 文本提取")

    # ── Online Vision OCR ───────────────────────────────────
    def _extract_online(self, image):
        """调用在线 Vision API 识别图片文字。

        启发式问题：本地引擎和在线 API 在输入输出格式上有何差异？
        将图片转换为 base64 编码需要考虑哪些因素（格式、大小）？
        网络请求失败时应该如何优雅地处理？
        """
        raise NotImplementedError("请实现在线 OCR 调用")

    # ── Public API ─────────────────────────────────────────
    def recognize(self, image):
        """识别手写文字区域，返回文字内容。

        Returns:
            str: 识别出的文字。空字符串表示无文字或识别失败。
                 失败时可通过 last_error 属性获取错误信息。
        """
        global last_error
        last_error = ""
        try:
            if self.engine == 'online':
                return self._extract_online(image)
            ocr = self._get_ocr()
            if self.engine == 'paddleocr':
                return self._extract_paddleocr(ocr, image)
            elif self.engine == 'easyocr':
                return self._extract_easyocr(ocr, image)
            elif self.engine == 'rapidocr':
                return self._extract_rapidocr(ocr, image)
        except (ImportError, OSError, ValueError) as e:
            import traceback
            traceback.print_exc()
            last_error = str(e)
            return ""
        except Exception as e:
            import traceback
            traceback.print_exc()
            last_error = str(e)
            return ""

last_error = ""


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        for eng in SUPPORTED_ENGINES:
            if not check_engine_available(eng):
                print(f"[{eng}] 未安装")
                continue
            if eng == 'online':
                print("[online] 需要 API Key，跳过自动测试")
                continue
            recognizer = EssayRecognizer(engine=eng)
            result = recognizer.recognize(img)
            print(f"[{eng}] ({len(result)}字) {result[:80]}")
