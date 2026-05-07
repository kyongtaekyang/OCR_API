import pytest
from src.comparator import compare


def _make_expected():
    return {
        "metadata": {"course_type": "keystone", "class": "B2", "writing_type": "paragraph"},
        "original_writing": "I [G]go to school every day.",
        "corrected_writing": "I go to school every day.",
        "scoring_analysis": {
            "grammar": {"subtotal": {"score": 20, "max_score": 30}},
            "vocabulary": {"subtotal": {"score": 20, "max_score": 30}},
            "writing_flow": {"subtotal": {"score": 30, "max_score": 40}},
        },
        "writing_performance": {"total_score": {"score": 70}},
        "error_explanations": [{"error": "go", "explanation_en": "tense error"}],
        "error_analysis": {
            "grammar": 1, "vocabulary": 0, "word_order": 0,
            "punctuation": 0, "spelling": 0, "coherence": 0, "total_errors": 1,
        },
        "overall_comments": "Good.",
    }


def _make_parse_success(output: dict) -> dict:
    return {"parse_success": True, "parsed_json": output, "error": None}


def _make_parse_fail() -> dict:
    return {"parse_success": False, "parsed_json": None, "error": "JSONDecodeError"}


def test_compare_identical_outputs():
    expected = _make_expected()
    result = compare(expected, _make_parse_success(expected))
    assert result["json_parse_success"] is True
    assert result["original_writing_similarity"] == pytest.approx(1.0)
    assert result["corrected_writing_similarity"] == pytest.approx(1.0)
    assert result["error_tag_f1"] == pytest.approx(1.0)
    assert result["overall_accuracy_score"] > 0.5


def test_compare_parse_failure():
    expected = _make_expected()
    result = compare(expected, _make_parse_fail())
    assert result["json_parse_success"] is False
    assert result["overall_accuracy_score"] == pytest.approx(0.0)


def test_compare_with_none_expected():
    result = compare(None, _make_parse_success(_make_expected()))
    assert result["json_parse_success"] is True
    assert result["overall_accuracy_score"] == pytest.approx(0.0)


def test_compare_error_analysis_difference():
    expected = _make_expected()
    model_out = dict(expected)
    model_out["error_analysis"] = {
        "grammar": 3, "vocabulary": 0, "word_order": 0,
        "punctuation": 0, "spelling": 0, "coherence": 0, "total_errors": 3,
    }
    result = compare(expected, _make_parse_success(model_out))
    assert result["total_errors_difference"] == 2


def test_compare_error_explanations_count_difference():
    expected = _make_expected()
    model_out = dict(expected)
    model_out["error_explanations"] = [
        {"error": "go", "explanation_en": "e1"},
        {"error": "day", "explanation_en": "e2"},
        {"error": "to", "explanation_en": "e3"},
    ]
    result = compare(expected, _make_parse_success(model_out))
    assert result["error_explanations_count_difference"] == 2
