import pytest
from src.metrics import (
    text_similarity,
    strip_error_tags,
    normalize_text,
    extract_error_tag_counts,
    extract_error_tag_counts_from_output,
    calculate_tag_precision_recall_f1,
    calculate_overall_accuracy_score,
)


def test_text_similarity_identical():
    assert text_similarity("hello world", "hello world") == pytest.approx(1.0)


def test_text_similarity_empty():
    assert text_similarity("", "") == pytest.approx(1.0)


def test_text_similarity_one_empty():
    assert text_similarity("hello", "") == pytest.approx(0.0)


def test_text_similarity_partial():
    score = text_similarity("hello world", "hello")
    assert 0.0 < score < 1.0


def test_text_similarity_ignores_tags():
    a = "I [G]go to school"
    b = "I go to school"
    assert text_similarity(a, b) == pytest.approx(1.0)


def test_strip_error_tags():
    text = "[G]hello [V]world [S]test"
    assert strip_error_tags(text) == "hello world test"


def test_strip_error_tags_multi():
    assert strip_error_tags("[G][V]word") == "word"


def test_normalize_text():
    result = normalize_text("  Hello  World  ")
    assert result == "hello world"


def test_extract_error_tag_counts():
    text = "[G]I [V]go [G]to [S]school [P]."
    counts = extract_error_tag_counts(text)
    assert counts["grammar"] == 2
    assert counts["vocabulary"] == 1
    assert counts["spelling"] == 1
    assert counts["punctuation"] == 1
    assert counts["word_order"] == 0


def test_extract_error_tag_counts_from_output():
    output = {"original_writing": "[G]bad [V]word [S]speling"}
    counts = extract_error_tag_counts_from_output(output)
    assert counts["grammar"] == 1
    assert counts["vocabulary"] == 1
    assert counts["spelling"] == 1


def test_precision_recall_f1_perfect():
    expected = {"grammar": 2, "vocabulary": 1, "word_order": 0,
                "punctuation": 0, "spelling": 0, "coherence": 0}
    actual = {"grammar": 2, "vocabulary": 1, "word_order": 0,
              "punctuation": 0, "spelling": 0, "coherence": 0}
    result = calculate_tag_precision_recall_f1(expected, actual)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_precision_recall_f1_zero_actual():
    expected = {"grammar": 2}
    actual = {"grammar": 0}
    result = calculate_tag_precision_recall_f1(expected, actual)
    assert result["f1"] == pytest.approx(0.0)


def test_overall_accuracy_score_perfect():
    comparison = {
        "schema_compliance_score": 1.0,
        "original_writing_similarity": 1.0,
        "corrected_writing_similarity": 1.0,
        "error_tag_f1": 1.0,
        "total_score_difference": 0.0,
        "error_analysis_category_difference": {},
    }
    score = calculate_overall_accuracy_score(comparison)
    assert score == pytest.approx(1.0)


def test_overall_accuracy_score_zero():
    comparison = {
        "schema_compliance_score": 0.0,
        "original_writing_similarity": 0.0,
        "corrected_writing_similarity": 0.0,
        "error_tag_f1": 0.0,
        "total_score_difference": 100.0,
        "error_analysis_category_difference": {"total_errors": 20},
    }
    score = calculate_overall_accuracy_score(comparison)
    assert score < 0.1


def test_total_score_difference():
    from src.metrics import calculate_score_differences
    expected = {
        "scoring_analysis": {
            "grammar": {"subtotal": {"score": 20, "max_score": 30}},
        },
        "writing_performance": {"total_score": {"score": 70}},
    }
    actual = {
        "scoring_analysis": {
            "grammar": {"subtotal": {"score": 18, "max_score": 30}},
        },
        "writing_performance": {"total_score": {"score": 65}},
    }
    diffs = calculate_score_differences(expected, actual)
    assert diffs["grammar_subtotal_difference"] == pytest.approx(2.0)
    assert diffs["total_score_difference"] == pytest.approx(5.0)
