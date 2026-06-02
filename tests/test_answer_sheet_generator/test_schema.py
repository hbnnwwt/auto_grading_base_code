"""Tests for answer_sheet_generator/schema.py."""

import json
import os
import tempfile

import pytest

from answer_sheet_generator.schema import (
    AnswerSheetConfig,
    MetaConfig,
    PageConfig,
    SectionConfig,
    StudentIdConfig,
)


class TestMetaConfig:
    def test_defaults(self):
        m = MetaConfig()
        assert m.title == "标准化考试答题卡"
        assert m.paper_size == "A4"

    def test_invalid_paper_size(self):
        with pytest.raises(ValueError, match="paper_size"):
            MetaConfig(paper_size="A5")


class TestStudentIdConfig:
    def test_valid(self):
        s = StudentIdConfig(digit_count=8)
        assert s.digit_count == 8

    def test_too_small(self):
        with pytest.raises(ValueError, match="digit_count"):
            StudentIdConfig(digit_count=5)

    def test_too_large(self):
        with pytest.raises(ValueError, match="digit_count"):
            StudentIdConfig(digit_count=15)


class TestSectionConfig:
    def test_choice_valid(self):
        sec = SectionConfig(
            type="choice",
            question_start=1,
            question_count=5,
            options=["A", "B", "C", "D"],
            score=2.0,
        )
        assert sec.get_score_for_question(1) == 2.0
        assert sec.get_score_for_question(5) == 2.0

    def test_essay_with_scores(self):
        sec = SectionConfig(
            type="essay",
            question_start=1,
            question_count=3,
            scores=[5.0, 10.0, 15.0],
            lines_per_question=5,
        )
        assert sec.get_score_for_question(2) == 10.0

    def test_missing_score(self):
        with pytest.raises(ValueError, match="score"):
            SectionConfig(
                type="choice",
                question_start=1,
                question_count=3,
                options=["A", "B"],
            )

    def test_both_score_and_scores(self):
        with pytest.raises(ValueError, match="score"):
            SectionConfig(
                type="choice",
                question_start=1,
                question_count=3,
                options=["A", "B"],
                score=1.0,
                scores=[1.0, 2.0, 3.0],
            )

    def test_scores_length_mismatch(self):
        with pytest.raises(ValueError, match="scores"):
            SectionConfig(
                type="choice",
                question_start=1,
                question_count=3,
                options=["A", "B"],
                scores=[1.0, 2.0],
            )

    def test_judge_must_be_tf(self):
        with pytest.raises(ValueError, match="judge"):
            SectionConfig(
                type="judge",
                question_start=1,
                question_count=2,
                options=["对", "错"],
                score=1.0,
            )

    def test_choice_options_too_few(self):
        with pytest.raises(ValueError, match="options"):
            SectionConfig(
                type="choice",
                question_start=1,
                question_count=2,
                options=["A"],
                score=1.0,
            )

    def test_essay_missing_lines(self):
        with pytest.raises(ValueError, match="lines_per_question"):
            SectionConfig(
                type="essay",
                question_start=1,
                question_count=2,
                score=5.0,
            )

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="type"):
            SectionConfig(
                type="fill_blank",
                question_start=1,
                question_count=2,
                score=1.0,
            )


class TestAnswerSheetConfig:
    def _make_valid(self):
        return AnswerSheetConfig(
            meta=MetaConfig(),
            student_id=StudentIdConfig(),
            pages=[
                PageConfig(
                    sections=[
                        SectionConfig(
                            type="choice",
                            question_start=1,
                            question_count=5,
                            options=["A", "B", "C", "D"],
                            score=2.0,
                        ),
                        SectionConfig(
                            type="judge",
                            question_start=6,
                            question_count=5,
                            options=["T", "F"],
                            score=1.0,
                        ),
                    ]
                )
            ],
        )

    def test_valid(self):
        cfg = self._make_valid()
        assert len(cfg.pages) == 1
        assert cfg.pages[0].sections[0].type == "choice"

    def test_empty_pages(self):
        with pytest.raises(ValueError, match="pages"):
            AnswerSheetConfig(
                meta=MetaConfig(),
                student_id=StudentIdConfig(),
                pages=[],
            )

    def test_overlap(self):
        with pytest.raises(ValueError, match="重叠"):
            AnswerSheetConfig(
                meta=MetaConfig(),
                student_id=StudentIdConfig(),
                pages=[
                    PageConfig(
                        sections=[
                            SectionConfig(
                                type="choice",
                                question_start=1,
                                question_count=5,
                                options=["A", "B", "C", "D"],
                                score=2.0,
                            ),
                            SectionConfig(
                                type="judge",
                                question_start=4,
                                question_count=5,
                                options=["T", "F"],
                                score=1.0,
                            ),
                        ]
                    )
                ],
            )

    def test_gap(self):
        with pytest.raises(ValueError, match="连续"):
            AnswerSheetConfig(
                meta=MetaConfig(),
                student_id=StudentIdConfig(),
                pages=[
                    PageConfig(
                        sections=[
                            SectionConfig(
                                type="choice",
                                question_start=1,
                                question_count=3,
                                options=["A", "B", "C", "D"],
                                score=2.0,
                            ),
                            SectionConfig(
                                type="judge",
                                question_start=5,
                                question_count=2,
                                options=["T", "F"],
                                score=1.0,
                            ),
                        ]
                    )
                ],
            )

    def test_from_dict_roundtrip(self):
        cfg = self._make_valid()
        d = cfg.to_dict()
        cfg2 = AnswerSheetConfig.from_dict(d)
        assert cfg2.meta.title == cfg.meta.title
        assert cfg2.pages[0].sections[0].question_count == 5

    def test_load_save(self):
        cfg = self._make_valid()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            cfg.save(path)
            cfg2 = AnswerSheetConfig.load(path)
            assert cfg2.to_dict() == cfg.to_dict()
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["meta"]["title"] == "标准化考试答题卡"
