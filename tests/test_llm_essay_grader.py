import pytest

from modules.llm_essay_grader import LLMEssayGrader, load_config, save_config


class TestParseResponse:
    def test_valid_response(self):
        grader = LLMEssayGrader(api_key="test", base_url="http://test",
                                model="test-model")
        text = "得分：15\n反馈：基本正确，但表述不够完整"
        score, feedback = grader._parse_response(text, max_score=20)
        assert score == 15
        assert "基本正确" in feedback

    def test_full_score(self):
        grader = LLMEssayGrader(api_key="test", base_url="http://test",
                                model="test-model")
        text = "得分：20\n反馈：完全正确"
        score, feedback = grader._parse_response(text, max_score=20)
        assert score == 20
        assert "完全正确" in feedback

    def test_zero_score(self):
        grader = LLMEssayGrader(api_key="test", base_url="http://test",
                                model="test-model")
        text = "得分：0\n反馈：答案错误"
        score, feedback = grader._parse_response(text, max_score=20)
        assert score == 0

    def test_score_exceeds_max_clamped(self):
        grader = LLMEssayGrader(api_key="test", base_url="http://test",
                                model="test-model")
        text = "得分：30\n反馈：超出满分"
        score, feedback = grader._parse_response(text, max_score=20)
        assert score == 20

    def test_malformed_response_returns_zero(self):
        grader = LLMEssayGrader(api_key="test", base_url="http://test",
                                model="test-model")
        text = "这个问题不太确定怎么评分"
        score, feedback = grader._parse_response(text, max_score=20)
        assert score == 0
        assert "人工" in feedback or "异常" in feedback

    def test_empty_response(self):
        grader = LLMEssayGrader(api_key="test", base_url="http://test",
                                model="test-model")
        score, feedback = grader._parse_response("", max_score=20)
        assert score == 0

    def test_response_with_extra_text(self):
        grader = LLMEssayGrader(api_key="test", base_url="http://test",
                                model="test-model")
        text = "好的，让我来评估。\n\n得分：10\n反馈：部分正确\n\n希望这有帮助。"
        score, feedback = grader._parse_response(text, max_score=20)
        assert score == 10


class TestInit:
    def test_stores_params(self):
        grader = LLMEssayGrader(api_key="sk-123", base_url="http://api.test",
                                model="deepseek-v4")
        assert grader.api_key == "sk-123"
        assert grader.base_url == "http://api.test"
        assert grader.model == "deepseek-v4"


class TestConfigLoadSave:
    def test_save_and_load(self, tmp_path):
        config = {
            "api_key": "sk-test-123",
            "base_url": "https://api-inference.modelscope.cn",
            "model": "deepseek-ai/DeepSeek-V4-Pro-Base",
        }
        config_path = str(tmp_path / "llm_config.json")
        save_config(config, config_path)
        loaded = load_config(config_path)
        assert loaded["api_key"] == "sk-test-123"
        assert loaded["base_url"] == "https://api-inference.modelscope.cn"
        assert loaded["model"] == "deepseek-ai/DeepSeek-V4-Pro-Base"

    def test_load_missing_file_returns_defaults(self, tmp_path):
        config_path = str(tmp_path / "nonexistent.json")
        loaded = load_config(config_path)
        assert loaded["api_key"] == ""
        assert "base_url" in loaded

    def test_from_config_creates_grader(self, tmp_path):
        config = {
            "api_key": "sk-abc",
            "base_url": "http://localhost:8080",
            "model": "test-model",
        }
        config_path = str(tmp_path / "llm_config.json")
        save_config(config, config_path)
        grader = LLMEssayGrader.from_config(config_path)
        assert grader.api_key == "sk-abc"
        assert grader.model == "test-model"
