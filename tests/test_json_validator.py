import pytest
from src.json_validator import validate_output, detect_score_anomalies


def _make_valid_system_output():
    return {
        "metadata": {
            "course_type": "keystone",
            "class": "B2",
            "title": "My Day",
            "title_corrected": "My Day",
            "topic": "daily life",
            "topic_corrected": "daily life",
            "writing_type": "paragraph",
        },
        "original_writing": "I go to school every [G]day.",
        "corrected_writing": "I go to school every day.",
        "scoring_analysis": {
            "grammar": {
                "sentence_accuracy": {"score": 8, "max_score": 10, "comment": ""},
                "verb_tense_consistency": {"score": 8, "max_score": 10, "comment": ""},
                "article_preposition": {"score": 7, "max_score": 10, "comment": ""},
                "subtotal": {"score": 23, "max_score": 30},
            },
            "vocabulary": {
                "word_variety": {"score": 7, "max_score": 10, "comment": ""},
                "appropriateness": {"score": 8, "max_score": 10, "comment": ""},
                "expression_naturalness": {"score": 7, "max_score": 10, "comment": ""},
                "subtotal": {"score": 22, "max_score": 30},
            },
            "writing_flow": {
                "structure_organization": {"score": 12, "max_score": 15, "comment": ""},
                "sentence_variety": {"score": 8, "max_score": 10, "comment": ""},
                "coherence_transitions": {"score": 12, "max_score": 15, "comment": ""},
                "subtotal": {"score": 32, "max_score": 40},
            },
        },
        "writing_performance": {
            "grammar": {"score": 76.7, "percentage": "76.7%"},
            "vocabulary": {"score": 73.3, "percentage": "73.3%"},
            "writing_flow": {"score": 80.0, "percentage": "80.0%"},
            "total_score": {"score": 77.0, "percentage": "77.0%"},
        },
        "error_explanations": [
            {"error": "day", "explanation_en": "Grammar error", "explanation": "문법 오류"}
        ],
        "error_analysis": {
            "grammar": 1, "vocabulary": 0, "word_order": 0,
            "punctuation": 0, "spelling": 0, "coherence": 0, "total_errors": 1,
        },
        "overall_comments": "Good writing.",
    }


def test_validate_valid_output():
    result = validate_output(_make_valid_system_output())
    assert result["valid"] is True
    assert result["required_key_completeness"] == 1.0
    assert result["missing_keys"] == []
    assert result["score_math_valid"] is True


def test_validate_missing_required_key():
    data = _make_valid_system_output()
    del data["error_analysis"]
    result = validate_output(data)
    assert result["valid"] is False
    assert "error_analysis" in result["missing_keys"]


def test_validate_none_input():
    result = validate_output(None)
    assert result["valid"] is False
    assert result["required_key_completeness"] == 0.0


def test_schema_type_detection_system():
    result = validate_output(_make_valid_system_output())
    assert result["schema_type"] == "system_prompt_schema"


def test_schema_type_detection_output():
    data = _make_valid_system_output()
    data["scoring_analysis"]["grammar"]["subject_verb_agreement"] = {"score": 8, "max_score": 10, "comment": ""}
    result = validate_output(data)
    assert result["schema_type"] == "output_prompt_schema"


def test_detect_score_anomaly_exceeds_max():
    data = _make_valid_system_output()
    data["scoring_analysis"]["grammar"]["sentence_accuracy"]["score"] = 15
    anomalies = detect_score_anomalies(data)
    paths = [a["path"] for a in anomalies]
    assert "scoring_analysis.grammar.sentence_accuracy" in paths


def test_detect_score_anomaly_subtotal_mismatch():
    data = _make_valid_system_output()
    data["scoring_analysis"]["grammar"]["subtotal"]["score"] = 99
    anomalies = detect_score_anomalies(data)
    paths = [a["path"] for a in anomalies]
    assert any("grammar" in p and "subtotal" in p for p in paths)


def test_no_anomaly_for_valid_output():
    anomalies = detect_score_anomalies(_make_valid_system_output())
    assert anomalies == []
