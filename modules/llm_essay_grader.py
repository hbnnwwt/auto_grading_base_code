import json
import os
import re

import requests

from modules.defaults import DEFAULT_BASE_URL, DEFAULT_LLM_MODEL
from modules.grading import EssayGraderBase


def load_config(config_path):
    """加载 LLM 配置文件，不存在时返回默认值。"""
    if not os.path.exists(config_path):
        return {
            "api_key": "",
            "base_url": DEFAULT_BASE_URL,
            "model": DEFAULT_LLM_MODEL,
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config, config_path):
    """保存 LLM 配置到 JSON 文件。"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


_DEFAULT_PROMPT = (
    "你是一个阅卷助手。请根据参考答案评估学生答案。\n\n"
    "注意事项：\n"
    "- 学生答案是通过 OCR 从手写文字识别的，可能存在识别错误（形近字、同音字）\n"
    "- 请根据语义判断，容忍合理的 OCR 识别偏差\n"
    "- 如果答案明显不确定或 OCR 识别质量差，请在反馈中说明\n\n"
    "参考答案：{reference}\n"
    "学生答案：{student_answer}\n"
    "满分：{max_score}分\n\n"
    "请严格按以下格式返回（不要输出其他内容）：\n"
    "得分：X\n"
    "反馈：一句话评语"
)

_PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "config", "llm_grading_prompt.txt")


def _load_prompt_template():
    if os.path.exists(_PROMPT_PATH):
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return _DEFAULT_PROMPT


class LLMEssayGrader(EssayGraderBase):
    """通过 LLM API 对简答题进行语义评分。

    调用 ModelScope OpenAI 兼容格式的 API，将参考答案和学生答案
    发给 LLM 判断得分。容忍 OCR 识别偏差，按语义匹配。
    """

    _prompt_template = None

    def __init__(self, api_key, base_url, model,
                 max_tokens=256, temperature=0.3):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_config(cls, config_path):
        """从配置文件构造 LLMEssayGrader。"""
        config = load_config(config_path)
        return cls(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url", DEFAULT_BASE_URL),
            model=config.get("model", DEFAULT_LLM_MODEL),
        )

    def _build_prompt(self, question, reference, student_answer, max_score):
        if LLMEssayGrader._prompt_template is None:
            LLMEssayGrader._prompt_template = _load_prompt_template()
        return LLMEssayGrader._prompt_template.format(
            reference=reference, student_answer=student_answer,
            max_score=max_score)

    def _call_api(self, messages):
        """调用 ModelScope OpenAI 兼容格式 API。"""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("message", {}).get("content", "")
        return ""

    def _parse_response(self, text, max_score):
        """从 LLM 返回文本中提取分数和反馈。"""
        score_match = re.search(r"得分[：:]\s*(\d+(?:\.\d+)?)", text)
        feedback_match = re.search(r"反馈[：:]\s*(.+)", text)

        if not score_match:
            return 0, "LLM 返回格式异常，需人工复核"

        score = min(int(score_match.group(1)), max_score)
        feedback = feedback_match.group(1).strip() if feedback_match else "LLM 评分完成"
        return score, feedback

    def score(self, question, reference, student_answer, max_score):
        if not student_answer or not student_answer.strip():
            return 0, max_score, "未作答"

        prompt = self._build_prompt(question, reference, student_answer,
                                    max_score)
        messages = [{"role": "user", "content": prompt}]

        max_retries = 2
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                text = self._call_api(messages)
                score, feedback = self._parse_response(text, max_score)
                return score, max_score, feedback
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    import time
                    time.sleep(2 * (attempt + 1))
        return 0, max_score, f"LLM 调用失败（重试{max_retries}次）: {last_err}"
