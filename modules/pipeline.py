"""共享识别管线：app.py 和 main.py 共用的图像处理和识别函数。"""

import json
import os

from modules.preprocess import ImagePreprocessor
from modules.layout import LayoutAnalyzer
from modules.choice_recognizer import ChoiceRecognizer
from modules.judge_recognizer import JudgeRecognizer
from modules.essay_recognizer import EssayRecognizer
from modules.student_id_recognizer import StudentIdRecognizer

# 加载答题卡布局配置
_LAYOUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "config", "sheet_layout.json")
if os.path.exists(_LAYOUT_PATH):
    with open(_LAYOUT_PATH, "r", encoding="utf-8") as _f:
        LAYOUT = json.load(_f)
else:
    LAYOUT = {
        "choice": {"rows": 5, "cols": 4, "question_start": 1,
                   "question_count": 20},
        "judge": {"rows": 3, "cols": 4, "question_start": 21,
                  "question_count": 10},
        "layout": {
            "page1_fallback": {"student_id": [0.06, 0.26],
                               "choice": [0.28, 0.80]},
            "page2_fallback": {"judge": [0.06, 0.46],
                               "essay": [0.50, 0.90]},
        },
    }


def _valid_region(val):
    """检查 region 是否为有效的 (x, y, w, h) 序列。"""
    return (isinstance(val, (tuple, list))
            and len(val) == 4
            and all(isinstance(v, (int, float)) for v in val))


def get_essay_questions(answer_key=None):
    """获取简答题题号列表，优先从答案键推断，否则 fallback 到 [31]。"""
    if answer_key is not None and 'essay' in answer_key:
        if answer_key['essay']:
            return sorted(answer_key['essay'].keys())
        return []
    return [31]


def preprocess_and_analyze(image, page, preprocessor=None, analyzer=None):
    """预处理图像并分析版面，返回 (regions, corrected_image)。"""
    if preprocessor is None:
        preprocessor = ImagePreprocessor()
    if analyzer is None:
        analyzer = LayoutAnalyzer()
    try:
        corrected, _, _, binary = preprocessor.process(image)
        regions = analyzer.analyze(corrected, binary, page=page)
    except NotImplementedError:
        # 骨架模式：预处理/版面分析未实现时返回空区域
        h, w = image.shape[:2]
        regions = {
            'student_id': None, 'choice': None,
            'judge': None, 'essay': None,
            'image_size': (w, h)
        }
        corrected = image
    return regions, corrected


def extract_student_id(image, regions, digit_count=10, threshold=0.3):
    """从版面区域中提取学号。"""
    if not _valid_region(regions.get('student_id')):
        return None
    x, y, w, h = regions['student_id']
    roi = image[y:y + h, x:x + w]
    rec = StudentIdRecognizer(digit_count=digit_count, threshold=threshold)
    try:
        return rec.recognize(roi)
    except NotImplementedError:
        return None


def recognize_choices(image, regions, threshold=0.06, return_details=False):
    """识别选择题答案。"""
    _ch = LAYOUT['choice']
    answers = {}
    if not _valid_region(regions.get('choice')):
        return (answers, []) if return_details else answers
    x, y, w, h = regions['choice']
    roi = image[y:y + h, x:x + w]
    try:
        recognizer = ChoiceRecognizer(threshold=threshold)
        result = recognizer.recognize_all_with_viz(
            roi, question_count=_ch['question_count'],
            question_start=_ch['question_start'],
            fixed_grid=(_ch.get('rows', 5), _ch.get('cols', 4)))
    except NotImplementedError:
        result = {'answers': {}, 'cell_results': []}
    if return_details:
        return result['answers'], result.get('cell_results', [])
    return result.get('answers', {})


def recognize_judges(image, regions, threshold=0.06, return_details=False):
    """识别判断题答案。"""
    _ju = LAYOUT['judge']
    answers = {}
    if not _valid_region(regions.get('judge')):
        return (answers, []) if return_details else answers
    x, y, w, h = regions['judge']
    roi = image[y:y + h, x:x + w]
    try:
        recognizer = JudgeRecognizer(threshold=threshold)
        result = recognizer.recognize_all_with_viz(
            roi, question_count=_ju['question_count'],
            question_start=_ju['question_start'],
            rows_n=_ju.get('rows', 3), cols_n=_ju.get('cols', 4))
    except NotImplementedError:
        result = {'answers': {}, 'cell_results': []}
    if return_details:
        return result['answers'], result.get('cell_results', [])
    return result.get('answers', {})


def recognize_essay(image, regions, ocr_engine='paddleocr', api_config=None,
                    cancel_check=None):
    """识别简答题答案。"""
    if not _valid_region(regions.get('essay')):
        return ""
    x, y, w, h = regions['essay']
    roi = image[y:y + h, x:x + w]
    try:
        recognizer = EssayRecognizer(engine=ocr_engine, api_config=api_config,
                                     cancel_check=cancel_check)
        return recognizer.recognize(roi)
    except NotImplementedError:
        return ""


def process_student_pair(page1_path, page2_path=None,
                         preprocessor=None, analyzer=None,
                         digit_count=10, threshold=0.5,
                         essay_questions=None):
    """处理一个学生的答题卡（第一页 + 第二页）。

    Returns:
        tuple: (student_id, recognized_answers)
    """
    if essay_questions is None:
        essay_questions = [31]

    image1_preprocessor = preprocessor or ImagePreprocessor()
    image1_analyzer = analyzer or LayoutAnalyzer()

    image1 = image1_preprocessor.load(page1_path)
    regions1, image1 = preprocess_and_analyze(image1, 1, image1_preprocessor, image1_analyzer)

    student_id = extract_student_id(image1, regions1, digit_count, threshold)
    choice_answers = recognize_choices(image1, regions1, threshold)

    recognized = {'choice': choice_answers, 'judge': {}, 'essay': {}}

    if page2_path:
        image2 = image1_preprocessor.load(page2_path)
        regions2, image2 = preprocess_and_analyze(image2, 2, image1_preprocessor, image1_analyzer)
        recognized['judge'] = recognize_judges(image2, regions2, threshold)
        essay_text = recognize_essay(image2, regions2)
        if essay_text:
            q = essay_questions[0] if essay_questions else 31
            recognized['essay'] = {q: essay_text}

    return student_id, recognized
