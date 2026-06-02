"""空白答卷基准校准模块。

通过扫描空白答题卡，计算判断题每个气泡区域的灰度基准值，
用于后续识别时检测淡铅笔填涂痕迹。
"""
import json
import os

import cv2
import numpy as np

from modules.preprocess import ImagePreprocessor
from modules.layout import LayoutAnalyzer


_BLANK_BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "config", "blank_baseline.json")


def _cell_zone_gray_stats(cell_gray, zone_bounds):
    """计算一个 cell 内各 zone 的灰度统计信息。

    Returns:
        list[dict]: 每个 zone 的 {'mean': float, 'std': float}
    """
    stats = []
    for zx0, zx1 in zone_bounds:
        zone = cell_gray[:, zx0:zx1]
        stats.append({
            "mean": float(np.mean(zone)),
            "std": float(np.std(zone)),
        })
    return stats


def _detect_zone_bounds(cell_gray, zone_count, num_w_ratio=0.20):
    """在 cell 中精确检测气泡位置，返回相对比例边界。

    使用灰度垂直投影在预期位置附近搜索局部峰值，
    以峰值间中点作为 zone 分界线。返回值为相对于 cell 宽度的比例。
    """
    h, w = cell_gray.shape
    # 垂直投影：反转灰度，暗处（气泡/印刷）值更高
    proj = 255.0 - np.mean(cell_gray, axis=0)
    window = max(3, int(w * 0.03))
    if window % 2 == 0:
        window += 1
    smooth = np.convolve(proj, np.ones(window) / window, mode='same')

    num_w = int(w * num_w_ratio)
    usable = w - num_w
    expected = [
        num_w + int((i + 0.5) * usable / zone_count)
        for i in range(zone_count)
    ]
    search = max(3, int(usable / zone_count * 0.35))

    peaks = []
    for exp in expected:
        x0 = max(num_w, exp - search)
        x1 = min(w, exp + search)
        if x1 <= x0:
            peaks.append(exp)
            continue
        peak = x0 + int(np.argmax(smooth[x0:x1]))
        peaks.append(peak)

    # 边界为相邻峰值中点，转换为相对比例
    bounds = []
    for i in range(zone_count):
        left = num_w if i == 0 else int((peaks[i - 1] + peaks[i]) / 2)
        right = w if i == zone_count - 1 else int((peaks[i] + peaks[i + 1]) / 2)
        bounds.append((left / w, right / w))
    return bounds


def _compute_region_baseline(gray, fill_start, rows_n, cols_n,
                             question_start, question_count,
                             zone_count, num_w_ratio=0.20):
    """计算某个区域（选择题或判断题）的灰度基准。"""
    img_h, img_w = gray.shape
    gray_fill = gray[fill_start:, :]
    fill_h = img_h - fill_start
    cell_h = fill_h / rows_n
    cell_w = img_w / cols_n

    questions = {}
    for row in range(rows_n):
        for col in range(cols_n):
            idx = row * cols_n + col
            q = question_start + idx
            if idx >= question_count:
                continue
            y0 = int(row * cell_h)
            y1 = int((row + 1) * cell_h)
            x0 = int(col * cell_w)
            x1 = int((col + 1) * cell_w)
            cell = gray_fill[y0:y1, x0:x1]
            cell_w_pixels = x1 - x0

            # 精确检测气泡边界（相对比例）
            zone_bounds_rel = _detect_zone_bounds(cell, zone_count, num_w_ratio)
            zone_bounds = [
                (int(r0 * cell_w_pixels), int(r1 * cell_w_pixels))
                for r0, r1 in zone_bounds_rel
            ]

            stats = _cell_zone_gray_stats(cell, zone_bounds)
            questions[str(q)] = {
                "zones": stats,
                "zone_bounds_rel": zone_bounds_rel,
            }
    return questions


def _compute_section_baseline(corrected, region_box, section_cfg, zone_count,
                              num_w_ratio=0.20):
    """计算单个 section（choice/judge）的灰度基准。

    Args:
        corrected: 预处理后的校正图像（BGR 或灰度）
        region_box: (x, y, w, h) 区域坐标
        section_cfg: dict with rows, cols, question_start, question_count
        zone_count: 每题选项数（choice=4, judge=2）
        num_w_ratio: 题号宽度比例

    Returns:
        dict: {"questions": {"1": {"zones": [...], "zone_bounds_rel": [...]}, ...}}
    """
    from modules.bubble_base import BubbleRecognizerBase
    base = BubbleRecognizerBase()

    x, y, w, h = region_box
    roi = corrected[y:y + h, x:x + w]
    gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    fill_start = base._detect_fill_start(gray)

    questions = _compute_region_baseline(
        gray, fill_start,
        section_cfg.get("rows", 5), section_cfg.get("cols", 4),
        section_cfg.get("question_start", 1),
        section_cfg.get("question_count", 20),
        zone_count=zone_count, num_w_ratio=num_w_ratio)
    return {"questions": questions}


def compute_blank_baseline(image_path, layout=None, page=2):
    """从空白答卷计算灰度基准。

    Args:
        image_path: 空白答卷图片路径
        layout: 答题卡布局配置 dict，None 时使用默认
        page: 1=第1页（含选择题），2=第2页（含判断题）

    Returns:
        dict: {
            "choice": {"questions": {"1": {"zones": [...]}, ...}},
            "judge": {"questions": {"21": {"zones": [...]}, ...}}
        }
    """
    if layout is None:
        layout = {
            "choice": {"rows": 5, "cols": 4, "question_start": 1,
                       "question_count": 20},
            "judge": {"rows": 3, "cols": 4, "question_start": 21,
                      "question_count": 10},
        }

    preprocessor = ImagePreprocessor()
    analyzer = LayoutAnalyzer()

    image = preprocessor.load(image_path)
    corrected, _, _, binary = preprocessor.process(image)
    regions = analyzer.analyze(corrected, binary, page=page)

    from modules.bubble_base import BubbleRecognizerBase
    base = BubbleRecognizerBase()

    result = {}

    # 选择题（page 1）
    if page == 1 and "choice" in regions:
        ch_cfg = layout.get("choice", {})
        x, y, w, h = regions["choice"]
        roi = corrected[y:y + h, x:x + w]
        gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fill_start = base._detect_fill_start(gray)
        result["choice"] = {
            "questions": _compute_region_baseline(
                gray, fill_start,
                ch_cfg.get("rows", 5), ch_cfg.get("cols", 4),
                ch_cfg.get("question_start", 1),
                ch_cfg.get("question_count", 20),
                zone_count=4, num_w_ratio=0.20)
        }

    # 判断题（page 2）
    if page == 2 and "judge" in regions:
        ju_cfg = layout.get("judge", {})
        x, y, w, h = regions["judge"]
        roi = corrected[y:y + h, x:x + w]
        gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fill_start = base._detect_fill_start(gray)
        result["judge"] = {
            "questions": _compute_region_baseline(
                gray, fill_start,
                ju_cfg.get("rows", 3), ju_cfg.get("cols", 4),
                ju_cfg.get("question_start", 21),
                ju_cfg.get("question_count", 10),
                zone_count=2, num_w_ratio=0.20)
        }

    if not result:
        raise ValueError(f"未能在空白答卷中定位识别区域 (page={page})")
    return result


def compute_blank_baseline_multipage(image_paths, layout=None):
    """多页空白答卷灰度基准校准。

    逐页处理空白答题卡，自动检测各页中的 choice/judge 区域，
    聚合所有页的基准数据，按题号汇总为统一字典。

    Args:
        image_paths: 每页图片路径列表（第0项=第1页，以此类推）
        layout: 答题卡布局配置 dict，None 时使用默认。
                若包含 _pages 字段，则按 _pages 配置逐页匹配 section；
                否则回退到旧的两页逻辑（page1=choice, page2=judge）。

    Returns:
        dict: {
            "choice": {"questions": {"1": {"zones": [...]}, ...}},
            "judge": {"questions": {"21": {"zones": [...]}, ...}}
        }
    """
    if layout is None:
        layout = {
            "choice": {"rows": 5, "cols": 4, "question_start": 1,
                       "question_count": 20},
            "judge": {"rows": 3, "cols": 4, "question_start": 21,
                      "question_count": 10},
        }

    preprocessor = ImagePreprocessor()
    analyzer = LayoutAnalyzer()

    # 加载并预处理所有页面
    images = []
    binaries = []
    corrected_images = []
    for path in image_paths:
        image = preprocessor.load(path)
        corrected, _, _, binary = preprocessor.process(image)
        images.append(image)
        binaries.append(binary)
        corrected_images.append(corrected)

    # 检测多页版面
    pages_config = layout.get("_pages")
    if pages_config:
        regions_list = analyzer.analyze_multipage(images, binaries)
    else:
        # 回退：旧的两页逻辑
        regions_list = []
        for idx, (corrected, binary) in enumerate(zip(corrected_images, binaries), start=1):
            regions_list.append(analyzer.analyze(corrected, binary, page=idx))

    result = {"choice": {"questions": {}}, "judge": {"questions": {}}}

    for page_idx, regions in enumerate(regions_list):
        # 确定当前页期望的 section 类型
        if pages_config and page_idx < len(pages_config):
            expected_sections = pages_config[page_idx]
        else:
            # 旧逻辑：page1 -> choice, page2 -> judge
            expected_sections = []
            if page_idx == 0:
                expected_sections = ["choice"]
            elif page_idx == 1:
                expected_sections = ["judge"]

        corrected = corrected_images[page_idx]

        for section in expected_sections:
            if section not in ("choice", "judge"):
                continue
            if regions.get(section) is None:
                continue

            sec_cfg = layout.get(section, {})
            zone_count = 4 if section == "choice" else 2
            baseline = _compute_section_baseline(
                corrected, regions[section], sec_cfg,
                zone_count=zone_count, num_w_ratio=0.20)
            result[section]["questions"].update(baseline["questions"])

    # 清理空 section
    for section in ("choice", "judge"):
        if not result[section]["questions"]:
            del result[section]

    if not result:
        raise ValueError("未能在多页空白答卷中定位任何 choice/judge 识别区域")
    return result


def save_baseline(baseline, path=None):
    """保存基准数据到 JSON 文件。"""
    if path is None:
        path = _BLANK_BASELINE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)


def load_baseline(path=None):
    """加载基准数据，不存在返回 None。"""
    if path is None:
        path = _BLANK_BASELINE_PATH
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_baseline_means(baseline, section):
    """提取某 section (choice/judge) 的 zone 均值字典。

    Returns:
        dict: {question_number: [zone0_mean, zone1_mean, ...]} 或 None
    """
    if baseline is None:
        return None
    sec = baseline.get(section, {})
    questions = sec.get("questions", {})
    result = {}
    for q_str, data in questions.items():
        q = int(q_str)
        means = [z["mean"] for z in data.get("zones", [])]
        result[q] = means
    return result


def get_judge_baseline_dict(baseline):
    return _get_baseline_means(baseline, "judge")


def get_choice_baseline_dict(baseline):
    return _get_baseline_means(baseline, "choice")


def _get_baseline_zone_bounds(baseline, section):
    """提取某 section 的 zone 边界相对比例。

    Returns:
        dict: {question_number: [(left_rel, right_rel), ...]} 或 None
    """
    if baseline is None:
        return None
    sec = baseline.get(section, {})
    questions = sec.get("questions", {})
    result = {}
    for q_str, data in questions.items():
        q = int(q_str)
        bounds = data.get("zone_bounds_rel")
        if bounds:
            result[q] = bounds
    return result


def get_judge_zone_bounds(baseline):
    return _get_baseline_zone_bounds(baseline, "judge")


def get_choice_zone_bounds(baseline):
    return _get_baseline_zone_bounds(baseline, "choice")
