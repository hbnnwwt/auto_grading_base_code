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
            "llm_model": DEFAULT_LLM_MODEL,
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config, config_path):
    """保存 LLM 配置到 JSON 文件（合并写，不覆盖已有键）。"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    existing = load_config(config_path)
    existing.update(config)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


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
        # 支持 str | list[str]，统一为列表
        self.api_keys = [api_key] if isinstance(api_key, str) else list(api_key or [])
        self.base_urls = [base_url] if isinstance(base_url, str) else list(base_url or [])
        self.models = [model] if isinstance(model, str) else list(model or [])
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_config(cls, config_path):
        """从配置文件构造 LLMEssayGrader。"""
        config = load_config(config_path)
        return cls(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url", DEFAULT_BASE_URL),
            model=config.get("llm_model", DEFAULT_LLM_MODEL),
        )

    def _build_prompt(self, question, reference, student_answer, max_score):
        if LLMEssayGrader._prompt_template is None:
            LLMEssayGrader._prompt_template = _load_prompt_template()
        return LLMEssayGrader._prompt_template.format(
            reference=reference, student_answer=student_answer,
            max_score=max_score)

    def _call_api(self, messages, cfg):
        """调用 ModelScope OpenAI 兼容格式 API。

        Args:
            messages: 消息列表
            cfg: dict，包含 api_key, base_url, model
        """
        url = f"{cfg['base_url'].rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        data = {
            "model": cfg["model"],
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            err_str = str(e)
            if ("10053" in err_str or "10054" in err_str
                    or "Connection aborted" in err_str):
                raise RuntimeError(
                    "网络连接被本地软件终止（错误 10053/10054）。"
                    "可能原因：Windows 防火墙/杀毒软件拦截、代理/VPN 冲突、"
                    "或网络不稳定。请检查："
                    "1) 防火墙是否放行 Python；"
                    "2) 是否开启了代理/VPN；"
                    "3) 网络连接是否正常。"
                ) from e
            raise RuntimeError(
                f"网络连接失败: {e}。"
                "请检查网络连接和 API 地址是否正确。"
            ) from e
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "请求超时（120秒）。请检查网络连接或稍后重试。"
            )

        result = resp.json()

        # 优先检查 API 返回的错误信息（某些代理服务会返回 200 但带 error 字段）
        error_info = result.get("error")
        if error_info:
            err_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
            raise RuntimeError(f"LLM API 错误: {err_msg}，原始响应: {json.dumps(result, ensure_ascii=False)[:500]}")

        choices = result.get("choices") or []
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if content:
                return content
            raise RuntimeError(f"LLM 返回空 content，原始响应: {json.dumps(result, ensure_ascii=False)[:500]}")

        # choices 为 null 或空数组时，根据 usage 给出更具体的诊断
        usage = result.get("usage", {})
        if usage.get("prompt_tokens", 0) == 0 and usage.get("completion_tokens", 0) == 0:
            model_id = cfg.get("model", "未知")
            raise RuntimeError(
                f"LLM 返回无 choices（请求未实际执行）。可能原因：模型名称 '{model_id}' 不被支持、"
                f"API Key 无效或服务端内部错误。原始响应: {json.dumps(result, ensure_ascii=False)[:500]}"
            )
        raise RuntimeError(f"LLM 返回无 choices，原始响应: {json.dumps(result, ensure_ascii=False)[:500]}")

    def _parse_response(self, text, max_score):
        """从 LLM 返回文本中提取分数和反馈。

        支持多种格式：
        - 得分：5 / 得分: 5 / 得分 5
        - Score: 5 / score: 5
        - 5分 / 5/20
        - {"score": 5, "feedback": "..."}
        """
        if not text or not text.strip():
            return 0, "LLM 返回为空"

        t = text.strip()

        # 1. 尝试解析 JSON 格式
        try:
            data = json.loads(t)
            if isinstance(data, dict):
                score = data.get("score") or data.get("得分")
                feedback = data.get("feedback") or data.get("反馈") or data.get("comment") or ""
                if score is not None:
                    return min(int(float(score)), max_score), str(feedback).strip() or "LLM 评分完成"
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 2. 正则匹配中文格式
        score_match = re.search(r"得分[：:\s]+(\d+(?:\.\d+)?)", t)
        feedback_match = re.search(r"反馈[：:\s]+(.+?)(?:\n|$)", t)

        # 3. 正则匹配英文格式
        if not score_match:
            score_match = re.search(r"[Ss]core[：:\s]+(\d+(?:\.\d+)?)", t)
        if not feedback_match:
            feedback_match = re.search(r"[Ff]eedback[：:\s]+(.+?)(?:\n|$)", t)
            if not feedback_match:
                feedback_match = re.search(r"[Cc]omment[：:\s]+(.+?)(?:\n|$)", t)

        # 4. 尝试提取 "X分" 或 "X/Y" 格式
        if not score_match:
            score_match = re.search(r"(\d+(?:\.\d+)?)\s*分", t)
        if not score_match:
            score_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*\d+", t)

        # 5. 最后的兜底：提取文本中第一个合理的数字（0 ~ max_score 范围内）
        if not score_match:
            all_nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", t)
            for num_str in all_nums:
                num = float(num_str)
                if 0 <= num <= max_score:
                    score_match = re.search(re.escape(num_str), t)
                    break

        if not score_match:
            # 实在找不到分数，把原始返回作为 feedback
            return 0, f"LLM 返回无法解析，原始内容：{t[:200]}"

        score = min(int(float(score_match.group(1))), max_score)

        if feedback_match:
            feedback = feedback_match.group(1).strip()
        else:
            # 尝试把除了得分行之外的其余内容作为反馈
            lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
            feedback_lines = []
            for ln in lines:
                if not re.search(r"得分[：:\s]+\d+", ln) and not re.search(r"[Ss]core[：:\s]+\d+", ln):
                    feedback_lines.append(ln)
            feedback = " ".join(feedback_lines) if feedback_lines else "LLM 评分完成"

        return score, feedback

    def score(self, question, reference, student_answer, max_score):
        if not student_answer or not str(student_answer).strip():
            return 0, max_score, "未作答"

        prompt = self._build_prompt(question, reference, student_answer,
                                    max_score)
        messages = [{"role": "user", "content": prompt}]

        # 构建所有 (key, url, model) 配置组合
        configs = []
        for key in (self.api_keys or []):
            for url in (self.base_urls or []):
                for model in (self.models or []):
                    configs.append({"api_key": key, "base_url": url, "model": model})

        if not configs:
            return 0, max_score, "LLM 未配置：缺少 api_key / base_url / model"

        last_err = None
        for i, cfg in enumerate(configs):
            try:
                text = self._call_api(messages, cfg)
                score, feedback = self._parse_response(text, max_score)
                return score, max_score, feedback
            except Exception as e:
                last_err = e
                # 检测 429 限流
                is_429 = False
                if hasattr(e, "response") and e.response is not None:
                    is_429 = e.response.status_code == 429
                elif "429" in str(e):
                    is_429 = True

                # 429 且有下一个配置时，退避后切换
                if is_429 and i < len(configs) - 1:
                    import time
                    time.sleep(min(2 * (i + 1), 30))  # 2s, 4s, 6s... 上限 30s
                # 非 429 或最后一个配置：继续尝试下一个（如果有）或最终失败

        return 0, max_score, f"LLM 调用失败: {last_err}"
