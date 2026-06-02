"""启动时校验配置文件完整性。

缺字段给默认值 + 警告，缺文件给明确提示。
避免配置问题在深层模块报 cryptic 错误。
"""

import json
import os

_REQUIRED_LAYOUT_KEYS = {'choice', 'judge', 'layout', 'scoring'}

_MODEL_CONFIG_DEFAULTS = {
    'base_url': 'https://api-inference.modelscope.cn',
    'llm_model': 'Qwen/Qwen3-235B-A22B',
    'ocr_model': 'Qwen/Qwen3-VL-235B-A22B-Instruct',
    'llm_max_tokens': 256,
    'llm_temperature': 0.3,
}


def validate_layout(path):
    """校验 sheet_layout.json。返回 (config_dict, warnings)。"""
    warnings = []
    if not os.path.exists(path):
        return None, [f"配置文件不存在: {path}"]

    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    missing = _REQUIRED_LAYOUT_KEYS - set(cfg.keys())
    if missing:
        warnings.append(f"sheet_layout.json 缺少字段: {missing}")

    return cfg, warnings


def validate_model_config(path):
    """校验 model_config.json，补全缺省字段。返回 (config_dict, warnings)。"""
    warnings = []
    if not os.path.exists(path):
        return dict(_MODEL_CONFIG_DEFAULTS), [f"模型配置不存在，使用默认值: {path}"]

    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    for key, default in _MODEL_CONFIG_DEFAULTS.items():
        if key not in cfg:
            cfg[key] = default
            warnings.append(f"model_config.json 缺 '{key}'，使用默认值: {default}")

    return cfg, warnings


def validate_all(base_dir):
    """校验所有配置。返回 warnings 列表，供调用方展示。"""
    all_warnings = []

    layout_path = os.path.join(base_dir, 'config', 'sheet_layout.json')
    _, w = validate_layout(layout_path)
    all_warnings.extend(w)

    model_path = os.path.join(base_dir, 'config', 'model_config.json')
    _, w = validate_model_config(model_path)
    all_warnings.extend(w)

    api_path = os.path.join(base_dir, 'config', 'api_keys.json')
    if not os.path.exists(api_path):
        all_warnings.append("api_keys.json 不存在（离线模式可忽略）")

    return all_warnings