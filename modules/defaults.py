"""集中管理的默认值常量。"""

import os
import re


def natural_sort_key(path):
    """提取文件名中的数字用于自然排序。

    例如：kaojuan_10.png -> ['kaojuan_', 10, '.png']
          kaojuan_100.png -> ['kaojuan_', 100, '.png']
    确保 3,4,5...9,10,11... 的顺序正确，而非字典序的 10,100,101...11。
    """
    filename = os.path.basename(path)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]

# API 默认值
DEFAULT_BASE_URL = "https://api-inference.modelscope.cn"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-235B-A22B"
DEFAULT_OCR_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"

# 图像处理常量
MORPH_KERNEL = (3, 3)
FILL_BAND_THRESHOLD = 0.02

# 图像文件后缀
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

# 判断题识别常量
JUDGE_MIN_FILL = 0.05           # 填涂检测最低填充率
JUDGE_STAIN_FILL_LOW = 0.04     # 污渍检测-填充率极低（淡铅笔可低至 4~5%）
JUDGE_STAIN_FILL_HIGH = 0.07    # 污渍检测-填充率较低
JUDGE_STAIN_RATIO = 0.80        # 污渍检测-对称率阈值
JUDGE_VALID_RATIO = 0.92        # 有效填涂-不对称率阈值
JUDGE_MULTI_FILL = 0.10         # 多选检测-单侧最低填充率
JUDGE_MULTI_RATIO = 0.85        # 多选检测-对称率阈值
JUDGE_BLOB_AREA_MIN = 100       # 气泡最小面积
JUDGE_BLOB_AREA_MAX = 3000      # 气泡最大面积
JUDGE_BLOB_ASPECT_MIN = 0.4     # 气泡最小宽高比
JUDGE_BLOB_ASPECT_MAX = 2.5     # 气泡最大宽高比
JUDGE_SIDE_MARGIN_RATIO = 0.10  # 侧边距比例
JUDGE_VERT_MARGIN_RATIO = 0.15  # 上下边距比例


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
