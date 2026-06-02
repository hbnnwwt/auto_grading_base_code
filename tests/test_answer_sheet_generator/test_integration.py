"""Integration tests for the full answer-sheet generation pipeline.

End-to-end tests covering:
1. Default template (20 choice + 10 judge + 1 essay)
2. A4 large exam (100 choice questions)
3. Config save/load roundtrip
"""

from __future__ import annotations

import pytest

from answer_sheet_generator.config_exporter import export_sheet_layout
from answer_sheet_generator.html_renderer import generate
from answer_sheet_generator.schema import (
    AnswerSheetConfig,
    MetaConfig,
    PageConfig,
    SectionConfig,
    StudentIdConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_default_config() -> AnswerSheetConfig:
    """Create the default template: 20 choice + 10 judge + 1 essay."""
    meta = MetaConfig(title="标准化考试答题卡", paper_size="A4")
    student_id = StudentIdConfig(digit_count=10)
    pages = [
        PageConfig(
            sections=[
                SectionConfig(
                    type="choice",
                    question_start=1,
                    question_count=20,
                    options=["A", "B", "C", "D"],
                    score=2.0,
                ),
                SectionConfig(
                    type="judge",
                    question_start=21,
                    question_count=10,
                    options=["T", "F"],
                    score=1.0,
                ),
                SectionConfig(
                    type="essay",
                    question_start=31,
                    question_count=1,
                    lines_per_question=5,
                    score=10.0,
                ),
            ]
        )
    ]
    return AnswerSheetConfig(meta=meta, student_id=student_id, pages=pages)


def _make_a4_large_exam_config() -> AnswerSheetConfig:
    """Create an A4 config with 100 choice questions."""
    meta = MetaConfig(title="A4 大型考试答题卡", paper_size="A4")
    student_id = StudentIdConfig(digit_count=12)
    pages = [
        PageConfig(
            sections=[
                SectionConfig(
                    type="choice",
                    question_start=1,
                    question_count=100,
                    options=["A", "B", "C", "D"],
                    score=1.0,
                ),
            ]
        )
    ]
    return AnswerSheetConfig(meta=meta, student_id=student_id, pages=pages)


# ---------------------------------------------------------------------------
# Test 1: Full pipeline — default template
# ---------------------------------------------------------------------------

class TestFullPipelineDefaultTemplate:
    def test_html_contains_section_titles(self) -> None:
        """HTML output must contain all three section titles."""
        cfg = _make_default_config()
        html = generate(cfg)

        assert "选择题" in html
        assert "判断题" in html
        assert "简答题" in html

    def test_html_contains_question_numbers(self) -> None:
        """HTML must contain the first and last question numbers of each section."""
        cfg = _make_default_config()
        html = generate(cfg)

        # Choice: 1~20
        assert "1." in html
        assert "20." in html
        # Judge: 21~30
        assert "21." in html
        assert "30." in html
        # Essay: 31
        assert "31." in html

    def test_layout_has_correct_question_counts(self) -> None:
        """Exported layout must reflect correct question counts."""
        cfg = _make_default_config()
        layout = export_sheet_layout(cfg)

        assert layout["choice"]["question_count"] == 20
        assert layout["judge"]["question_count"] == 10
        # Essay is not part of old-format choice/judge; verify via _pages
        pages = layout["_pages"]
        assert len(pages) == 1
        sections = pages[0]["sections"]
        assert sections[0]["question_count"] == 20
        assert sections[1]["question_count"] == 10
        assert sections[2]["question_count"] == 1
        assert sections[2]["type"] == "essay"

    def test_layout_student_id(self) -> None:
        """Layout must contain correct student_id digit count."""
        cfg = _make_default_config()
        layout = export_sheet_layout(cfg)

        assert layout["student_id"]["digit_count"] == 10


# ---------------------------------------------------------------------------
# Test 2: Full pipeline — A4 large exam
# ---------------------------------------------------------------------------

class TestFullPipelineA4LargeExam:
    def test_html_mentions_a4(self) -> None:
        """HTML output must reference A4 page size in CSS."""
        cfg = _make_a4_large_exam_config()
        html = generate(cfg)

        # A4 page size in CSS: 210mm 297mm
        assert "210mm 297mm" in html

    def test_layout_choice_count_is_100(self) -> None:
        """Exported layout must report 100 choice questions."""
        cfg = _make_a4_large_exam_config()
        layout = export_sheet_layout(cfg)

        assert layout["choice"]["question_count"] == 100
        assert layout["choice"]["cols"] == 5  # A4 uses 5 columns

    def test_html_contains_all_question_numbers(self) -> None:
        """HTML must contain first and last question numbers."""
        cfg = _make_a4_large_exam_config()
        html = generate(cfg)

        assert "1." in html
        assert "100." in html


# ---------------------------------------------------------------------------
# Test 3: Config roundtrip
# ---------------------------------------------------------------------------

class TestConfigRoundtrip:
    def test_save_load_preserves_data(self, tmp_path) -> None:
        """Saving and loading a config must preserve all fields."""
        cfg = _make_default_config()
        original_dict = cfg.to_dict()

        path = str(tmp_path / "config.json")
        cfg.save(path)

        loaded = AnswerSheetConfig.load(path)
        loaded_dict = loaded.to_dict()

        assert loaded_dict == original_dict

    def test_save_load_a4_large_config(self, tmp_path) -> None:
        """Roundtrip must work for A4 large configs as well."""
        cfg = _make_a4_large_exam_config()
        original_dict = cfg.to_dict()

        path = str(tmp_path / "a4_large_config.json")
        cfg.save(path)

        loaded = AnswerSheetConfig.load(path)
        loaded_dict = loaded.to_dict()

        assert loaded_dict == original_dict

    def test_roundtrip_preserves_meta(self, tmp_path) -> None:
        """Meta fields (title, paper_size) survive roundtrip."""
        cfg = _make_default_config()
        path = str(tmp_path / "meta_test.json")
        cfg.save(path)

        loaded = AnswerSheetConfig.load(path)
        assert loaded.meta.title == "标准化考试答题卡"
        assert loaded.meta.paper_size == "A4"

    def test_roundtrip_preserves_student_id(self, tmp_path) -> None:
        """Student ID digit count survives roundtrip."""
        cfg = _make_a4_large_exam_config()
        path = str(tmp_path / "sid_test.json")
        cfg.save(path)

        loaded = AnswerSheetConfig.load(path)
        assert loaded.student_id.digit_count == 12
