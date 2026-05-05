"""
智能阅卷系统 - CLI 入口

用法:
    python main.py --folder <答题卡图像文件夹> [--answer-key <参考答案.xlsx>] [--output <结果.xlsx>]
    python main.py --image <答题卡图像路径> [--page 1] [--answer-key <参考答案.xlsx>]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from modules.pipeline import (
    preprocess_and_analyze, extract_student_id,
    recognize_choices, recognize_judges, recognize_essay,
    get_essay_questions, LAYOUT,
)
from modules.preprocess import ImagePreprocessor
from modules.layout import LayoutAnalyzer
from modules.grading import GradingService
from modules.llm_essay_grader import LLMEssayGrader, load_config, save_config
from modules.marker import mark_and_save
from modules.defaults import (
    DEFAULT_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_OCR_MODEL,
    IMAGE_EXTS, path_constants,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = path_constants(BASE_DIR)


def collect_image_files(folder):
    """收集文件夹中的图像文件，按文件名排序。"""
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]
    files.sort()
    return files


def _build_essay_grader(args):
    """根据命令行参数构建 LLM 评分器，未启用时返回 None。"""
    if not args.llm:
        return None

    keys = load_config(PATHS['api_keys'])
    mcfg = load_config(PATHS['model_config'])

    api_key = args.api_key or keys.get("api_key", "")
    base_url = args.base_url or mcfg.get("base_url", DEFAULT_BASE_URL)
    model = args.model or mcfg.get("llm_model", DEFAULT_LLM_MODEL)

    if not api_key.strip():
        print("警告: LLM 评分已启用但未配置 API Key，简答题将需人工评分")
        return None

    return LLMEssayGrader(
        api_key=api_key, base_url=base_url, model=model,
        max_tokens=mcfg.get("llm_max_tokens", 256),
        temperature=mcfg.get("llm_temperature", 0.3))


def _build_ocr_config(args):
    """构建 OCR API 配置。"""
    if args.ocr_engine != 'online':
        return None

    keys = load_config(PATHS['api_keys'])
    mcfg = load_config(PATHS['model_config'])
    api_key = keys.get("ocr_api_key", "") or keys.get("api_key", "")
    return {
        'api_key': api_key,
        'base_url': mcfg.get("base_url", DEFAULT_BASE_URL),
        'ocr_model': mcfg.get("ocr_model", DEFAULT_OCR_MODEL),
        'ocr_max_tokens': mcfg.get("ocr_max_tokens", 1024),
        'ocr_prompt': mcfg.get("ocr_prompt",
            "请逐行识别图片中的所有文字内容，只输出文字，不要添加解释。"),
    }


def process_student(page1_path, page2_path, preprocessor, analyzer,
                    threshold=0.06, digit_count=10,
                    ocr_engine='paddleocr', ocr_config=None):
    """处理一个学生的答题卡（第一页 + 第二页）。"""
    image1 = preprocessor.load(page1_path)
    regions1, image1 = preprocess_and_analyze(image1, 1, preprocessor, analyzer)

    student_id = extract_student_id(image1, regions1, digit_count)
    choice_answers, choice_cells = recognize_choices(
        image1, regions1, threshold, return_details=True)

    recognized = {'choice': choice_answers, 'judge': {}, 'essay': {}}
    regions2 = {}
    image2 = None

    if page2_path:
        image2 = preprocessor.load(page2_path)
        regions2, image2 = preprocess_and_analyze(image2, 2, preprocessor, analyzer)
        judge_answers, judge_cells = recognize_judges(
            image2, regions2, threshold, return_details=True)
        recognized['judge'] = judge_answers
        essay_text = recognize_essay(image2, regions2, ocr_engine, ocr_config)
        if essay_text:
            recognized['essay'] = {get_essay_questions()[0]: essay_text}
    else:
        judge_cells = []

    return student_id, recognized, image1, image2, regions1, regions2, choice_cells, judge_cells


def main():
    parser = argparse.ArgumentParser(description='智能阅卷系统')
    parser.add_argument('--image', help='单张答题卡图像路径')
    parser.add_argument('--folder', help='答题卡图像文件夹路径（批量处理）')
    parser.add_argument('--page', type=int, default=1, choices=[1, 2],
                        help='单图模式下的页码，默认1')
    parser.add_argument('--answer-key', default='参考答案.xlsx',
                        help='标准答案 xlsx 文件路径')
    parser.add_argument('--output', default='结果.xlsx',
                        help='输出结果 xlsx 文件路径')
    parser.add_argument('--digit-count', type=int, default=10,
                        help='学号位数，默认10')
    parser.add_argument('--threshold', type=float, default=0.06,
                        help='填涂识别阈值，默认0.06')
    parser.add_argument('--ocr-engine', default='paddleocr',
                        choices=['paddleocr', 'easyocr', 'rapidocr', 'online'],
                        help='简答题 OCR 引擎，默认 paddleocr')
    parser.add_argument('--llm', action='store_true',
                        help='启用 LLM 简答题自动评分')
    parser.add_argument('--api-key', default='',
                        help='LLM / OCR API Key（也可通过 config/api_keys.json 配置）')
    parser.add_argument('--base-url', default='',
                        help='API Base URL（也可通过 config/model_config.json 配置）')
    parser.add_argument('--model', default='',
                        help='LLM 模型名称（也可通过 config/model_config.json 配置）')
    parser.add_argument('--no-mark', action='store_true',
                        help='不生成错题标注图')
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error('请指定 --image 或 --folder')

    answer_key_path = os.path.join(BASE_DIR, args.answer_key) \
        if not os.path.isabs(args.answer_key) else args.answer_key
    if not os.path.exists(answer_key_path):
        print(f"错误: 参考答案文件不存在: {answer_key_path}")
        sys.exit(1)
    service = GradingService.from_xlsx(answer_key_path)

    essay_grader = _build_essay_grader(args)
    if essay_grader:
        service.essay_grader = essay_grader

    ocr_config = _build_ocr_config(args)

    if args.folder:
        batch_process(args, service, ocr_config)
    else:
        single_process(args, service, ocr_config)


def batch_process(args, service, ocr_config):
    """批量处理：文件夹中的图像按奇偶配对。"""
    folder = os.path.join(BASE_DIR, args.folder) \
        if not os.path.isabs(args.folder) else args.folder
    files = collect_image_files(folder)

    if not files:
        print(f"错误: 文件夹中未找到图像文件: {folder}")
        sys.exit(1)

    if len(files) % 2 != 0:
        print(f"警告: 图像文件数为奇数({len(files)})，最后一个文件将被忽略")

    pair_count = len(files) // 2
    all_results = []
    preprocessor = ImagePreprocessor()
    analyzer = LayoutAnalyzer()

    choice_max = len(service.answer_key.get('choice', {})) * service.choice_score
    judge_max = len(service.answer_key.get('judge', {})) * service.judge_score
    essay_max = len(service.answer_key.get('essay', {})) * service.essay_max_score
    processed_dir = PATHS['processed_dir']

    for i in range(pair_count):
        page1, page2 = files[2 * i], files[2 * i + 1]
        print(f"\n{'=' * 50}")
        print(f"[{i + 1}/{pair_count}] {os.path.basename(page1)} + {os.path.basename(page2)}")

        student_id, recognized, p1, p2, r1, r2, ccells, jcells = process_student(
            page1, page2, preprocessor, analyzer,
            threshold=args.threshold, digit_count=args.digit_count,
            ocr_engine=args.ocr_engine, ocr_config=ocr_config)

        print(f"学号: {student_id}")
        print(f"选择题: {recognized['choice']}")
        print(f"判断题: {recognized['judge']}")
        if recognized['essay']:
            print(f"简答题: {recognized['essay']}")

        result = service.grade(recognized)
        print(service.generate_report(result))

        if not args.no_mark and p1 is not None and p2 is not None:
            mark_and_save(student_id, p1, p2, r1, r2,
                          ccells, jcells, result,
                          choice_max=choice_max, judge_max=judge_max,
                          essay_max=essay_max, output_dir=processed_dir)
            print(f"  标注图已保存到: {processed_dir}/")

        all_results.append((student_id, recognized))

    output_path = os.path.join(BASE_DIR, args.output) \
        if not os.path.isabs(args.output) else args.output
    _save_results_xlsx(service, output_path, all_results)
    print(f"\n批量处理完成({pair_count}份)，结果已保存到: {output_path}")


def single_process(args, service, ocr_config):
    """单图处理模式。"""
    image_path = os.path.join(BASE_DIR, args.image) \
        if not os.path.isabs(args.image) else args.image

    preprocessor = ImagePreprocessor()
    analyzer = LayoutAnalyzer()

    image = preprocessor.load(image_path)
    regions, corrected, _, _ = preprocessor.process(image)
    regions = analyzer.analyze(corrected, _, page=args.page)

    print(f"图像: {image_path} (第{args.page}页)")

    student_id = None
    recognized = {'choice': {}, 'judge': {}, 'essay': {}}

    if args.page == 1:
        student_id = extract_student_id(corrected, regions, args.digit_count)
        recognized['choice'] = recognize_choices(corrected, regions, args.threshold)
        print(f"学号: {student_id}")
        print(f"选择题: {recognized['choice']}")
    else:
        recognized['judge'] = recognize_judges(corrected, regions, args.threshold)
        essay_text = recognize_essay(corrected, regions, args.ocr_engine, ocr_config)
        if essay_text:
            recognized['essay'] = {get_essay_questions()[0]: essay_text}
        print(f"判断题: {recognized['judge']}")
        print(f"简答题: {recognized['essay']}")

    result = service.grade(recognized)
    print(service.generate_report(result))


def _save_results_xlsx(service, output_path, all_results):
    """将识别结果保存为 xlsx（从头创建工作簿，不依赖模板文件）。"""
    import openpyxl

    _ch = LAYOUT['choice']
    _ju = LAYOUT['judge']
    _es = LAYOUT.get('essay', {})
    choice_start = _ch['question_start']
    choice_end = choice_start + _ch['question_count'] - 1
    judge_start = _ju['question_start']
    judge_end = judge_start + _ju['question_count'] - 1
    essay_questions = get_essay_questions(service.answer_key)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "评分结果"

    # 表头：学号 + 各题 + 简答题内容 + 简答题反馈
    headers = ["学号"]
    for q in range(choice_start, judge_end + 1):
        headers.append(str(q))
    for eq in essay_questions:
        headers += [f"简答题{eq}内容", f"简答题{eq}反馈"]
    ws.append(headers)

    choice_score = service.choice_score
    judge_score = service.judge_score

    for student_id, rec in all_results:
        row = [student_id or "unknown"]
        for q in range(choice_start, judge_end + 1):
            if q <= choice_end:
                row.append(rec.get('choice', {}).get(q))
            else:
                row.append(rec.get('judge', {}).get(q))
        for eq in essay_questions:
            row.append(rec.get('essay', {}).get(eq, ''))
            row.append('')
        ws.append(row)

        # 分数行
        score_row = [f"{student_id}_score" if student_id else "unknown_score"]
        for q in range(choice_start, choice_end + 1):
            ans = rec.get('choice', {}).get(q)
            correct = service.answer_key.get('choice', {}).get(q)
            score_row.append(choice_score if ans and ans == correct else 0)
        for q in range(judge_start, judge_end + 1):
            ans = rec.get('judge', {}).get(q)
            correct = service.answer_key.get('judge', {}).get(q)
            score_row.append(judge_score if ans and ans == correct else 0)
        score_row += [''] * (len(essay_questions) * 2)
        ws.append(score_row)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    wb.save(output_path)


if __name__ == '__main__':
    main()
