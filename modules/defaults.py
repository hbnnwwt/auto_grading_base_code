"""集中管理的默认值常量。"""

import os

# API 默认值
DEFAULT_BASE_URL = "https://api-inference.modelscope.cn"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-235B-A22B"
DEFAULT_OCR_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"

# 图像处理常量
MORPH_KERNEL = (3, 3)
FILL_BAND_THRESHOLD = 0.02

# 图像文件后缀
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def path_constants(base_dir):
    """返回项目路径常量字典。"""
    return {
        'answer_key': os.path.join(base_dir, "参考答案.xlsx"),
        'default_folder': os.path.join(base_dir, "data", "answer_sheets"),
        'output_dir': os.path.join(base_dir, "data", "output"),
        'processed_dir': os.path.join(base_dir, "data", "processed"),
        'api_keys': os.path.join(base_dir, "config", "api_keys.json"),
        'model_config': os.path.join(base_dir, "config", "model_config.json"),
        'batch_checkpoint': os.path.join(base_dir, "data", "output", "_batch_checkpoint.json"),
    }
