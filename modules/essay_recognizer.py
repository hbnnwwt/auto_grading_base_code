import cv2
import numpy as np

SUPPORTED_ENGINES = ['paddleocr', 'easyocr', 'rapidocr', 'online']

# 记录最后一次识别的错误信息，供调用者查询
last_error = ''


def check_engine_available(engine):
    """检查 OCR 引擎是否已安装。

    Args:
        engine: str, 引擎名称 ('paddleocr'/'easyocr'/'rapidocr'/'online')

    Returns:
        bool: 是否可用
    """
    import importlib.util
    mapping = {
        'paddleocr': 'paddleocr',
        'easyocr': 'easyocr',
        'rapidocr': 'rapidocr_onnxruntime',
        'online': 'requests',
    }
    return importlib.util.find_spec(mapping.get(engine, '')) is not None


class EssayRecognizer:
    """简答题文字识别模块。

    调用者（pipeline.py）使用方式：
        rec = EssayRecognizer(engine='paddleocr')
        text = rec.recognize(roi)

    目标：给定简答题手写区域图像，识别出其中的文字内容。
    至少实现一种 OCR 引擎即可，推荐 PaddleOCR（对中文手写体效果好）。
    """

    def __init__(self, engine='paddleocr', lang='ch', api_config=None,
                 cancel_check=None, max_image_side=2048):
        """
        Args:
            engine: str, OCR 引擎名称，见 SUPPORTED_ENGINES
            lang: str, 语言代码（默认 'ch' 中文）
            api_config: dict|None, 在线引擎的 API 配置
            cancel_check: callable|None, 取消检查函数（返回 True 时中断）
            max_image_side: int, 图片最大边长（超过则缩放）
        """
        self.engine = engine
        self.lang = lang
        self.api_config = api_config or {}
        self.cancel_check = cancel_check
        self.max_image_side = max_image_side

    def recognize(self, image):
        """识别简答题文字。

        Args:
            image: ndarray (H, W, 3), 简答题区域的 BGR 图像

        Returns:
            str: 识别到的文字内容，失败时返回空字符串 ""

        思路提示：
            - 手写文字和选择题的填涂在图像特征上有什么本质区别？
              填涂是二值（黑/白），手写文字是什么？
            - 现有 OCR 引擎（paddleocr/easyocr）能识别手写中文吗？
              它们各自的优势和局限是什么？
            - OCR 前是否需要图像预处理（去噪、二值化）？
              如果不处理，OCR 引擎自己能否应对低质量图像？
            - 如何判断 OCR 识别失败了（而不是识别出了错误文字）？
        """
        raise NotImplementedError("TODO: 请实现简答题 OCR recognize() 方法")
