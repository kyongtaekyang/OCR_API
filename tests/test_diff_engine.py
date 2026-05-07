"""Tests for diff_engine: field-level diff between expected and model output."""
import pytest
from src.diff_engine import (
    compute_subcategory_diffs,
    compute_per_tag_f1,
    compute_error_explanation_diffs,
    compute_metadata_diffs,
    compute_writing_performance_diffs,
    compute_error_analysis_diff,
    compute_field_diffs,
    compute_text_char_diffs,
    compute_internal_consistency,
    compute_diff,
)


def _make_expected():
    return {
        "metadata": {"course_type": "keystone", "class": "B2",
                     "title": "My Day", "title_corrected": "My Day",
                     "topic": "daily life", "topic_corrected": "daily life",
                     "writing_type": "paragraph"},
        "original_writing": "I [G]go to school every [S]day.",
        "corrected_writing": "I go to school every day.",
        "scoring_analysis": {
            "grammar": {
                "sentence_accuracy": {"score": 8, "max_score": 10},
                "verb_tense_consistency": {"score": 8, "max_score": 10},
                "article_preposition": {"score": 7, "max_score": 10},
                "subtotal": {"score": 23, "max_score": 30},
            },
            "vocabulary": {
                "word_variety": {"score": 7, "max_score": 10},
                "appropriateness": {"score": 8, "max_score": 10},
                "expression_naturalness": {"score": 7, "max_score": 10},
                "subtotal": {"score": 22, "max_score": 30},
            },
            "writing_flow": {
                "structure_organization": {"score": 12, "max_score": 15},
                "sentence_variety": {"score": 8, "max_score": 10},
                "coherence_transitions": {"score": 12, "max_score": 15},
                "subtotal": {"score": 32, "max_score": 40},
            },
        },
        "writing_performance": {
            "grammar": {"score": 76.7},
            "vocabulary": {"score": 73.3},
            "writing_flow": {"score": 80.0},
            "total_score": {"score": 77.0},
        },
        "error_explanations": [
            {"error": "[G]go", "explanation_en": "Tense error"},
            {"error": "[S]day", "explanation_en": "Spelling error"},
        ],
        "error_analysis": {
            "grammar": 1, "vocabulary": 0, "word_order": 0,
            "punctuation": 0, "spelling": 1, "coherence": 0, "total_errors": 2,
        },
        "overall_comments": "Good effort overall.",
    }


# ── compute_subcategory_diffs ─────────────────────────────────────────────────

def test_subcategory_diffs_identical():
    expected = _make_expected()
    result = compute_subcategory_diffs(expected, expected)
    for cat, entries in result.items():
        for e in entries:
            assert e["status"] == "match", f"{cat}.{e['subcategory']} should be match"
            assert e["score_delta"] == 0


def test_subcategory_diffs_off_by_small():
    expected = _make_expected()
    actual = _make_expected()
    actual["scoring_analysis"]["grammar"]["sentence_accuracy"]["score"] = 6  # diff = 2
    result = compute_subcategory_diffs(expected, actual)
    item = next(e for e in result["grammar"] if e["subcategory"] == "sentence_accuracy")
    assert item["status"] == "off_by_small"
    assert item["score_delta"] == pytest.approx(2.0)


def test_subcategory_diffs_off_by_large():
    expected = _make_expected()
    actual = _make_expected()
    actual["scoring_analysis"]["grammar"]["sentence_accuracy"]["score"] = 1  # diff = 7
    result = compute_subcategory_diffs(expected, actual)
    item = next(e for e in result["grammar"] if e["subcategory"] == "sentence_accuracy")
    assert item["status"] == "off_by_large"
    assert item["score_delta"] == pytest.approx(7.0)


def test_subcategory_diffs_missing_category():
    expected = _make_expected()
    actual = _make_expected()
    del actual["scoring_analysis"]["grammar"]
    result = compute_subcategory_diffs(expected, actual)
    for e in result.get("grammar", []):
        assert e["status"] == "missing"


# ── compute_per_tag_f1 ────────────────────────────────────────────────────────

def test_per_tag_f1_perfect():
    expected = _make_expected()
    result = compute_per_tag_f1(expected, expected)
    g = next(t for t in result if t["tag_name"] == "grammar")
    s = next(t for t in result if t["tag_name"] == "spelling")
    assert g["f1"] == pytest.approx(1.0)
    assert s["f1"] == pytest.approx(1.0)
    assert g["status"] == "perfect"


def test_per_tag_f1_over_tagged():
    expected = _make_expected()
    actual = _make_expected()
    actual["original_writing"] = "I [G]go [G]to school every [S]day."
    result = compute_per_tag_f1(expected, actual)
    g = next(t for t in result if t["tag_name"] == "grammar")
    assert g["fp"] == 1
    assert g["status"] == "over_tagged"


def test_per_tag_f1_under_tagged():
    expected = _make_expected()
    actual = _make_expected()
    actual["original_writing"] = "I go to school every [S]day."
    result = compute_per_tag_f1(expected, actual)
    g = next(t for t in result if t["tag_name"] == "grammar")
    assert g["fn"] == 1
    assert g["status"] in ("under_tagged", "missed")


def test_per_tag_f1_missed():
    expected = _make_expected()
    actual = _make_expected()
    actual["original_writing"] = "I go to school every day."  # no tags
    actual["error_explanations"] = []
    actual["error_analysis"] = {
        "grammar": 0, "vocabulary": 0, "word_order": 0,
        "punctuation": 0, "spelling": 0, "coherence": 0, "total_errors": 0,
    }
    result = compute_per_tag_f1(expected, actual)
    g = next(t for t in result if t["tag_name"] == "grammar")
    s = next(t for t in result if t["tag_name"] == "spelling")
    assert g["f1"] == pytest.approx(0.0)
    assert g["status"] == "missed"
    assert s["status"] == "missed"


def test_per_tag_f1_falls_back_to_explanation_tags():
    expected = _make_expected()
    actual = _make_expected()
    actual["original_writing"] = "I go to school every day."
    actual["error_explanations"] = [
        {"error": "go", "explanation": "[G]"},
        {"error": "day", "explanation": "[S]"},
    ]
    result = compute_per_tag_f1(expected, actual)
    g = next(t for t in result if t["tag_name"] == "grammar")
    s = next(t for t in result if t["tag_name"] == "spelling")
    assert g["f1"] == pytest.approx(1.0)
    assert s["f1"] == pytest.approx(1.0)


def test_per_tag_f1_na_for_absent_category():
    expected = _make_expected()
    actual = _make_expected()
    result = compute_per_tag_f1(expected, actual)
    v = next(t for t in result if t["tag_name"] == "vocabulary")
    assert v["status"] == "n/a"
    assert v["expected_count"] == 0
    assert v["actual_count"] == 0


# ── compute_error_explanation_diffs ───────────────────────────────────────────

def test_explanation_diffs_matched():
    expected = _make_expected()
    result = compute_error_explanation_diffs(expected, expected)
    for d in result:
        assert d["status"] == "matched"


def test_explanation_diffs_extra():
    expected = _make_expected()
    actual = _make_expected()
    actual["error_explanations"].append({"error": "[P].", "explanation_en": "Extra"})
    result = compute_error_explanation_diffs(expected, actual)
    assert result[-1]["status"] == "extra"


def test_explanation_diffs_missed():
    expected = _make_expected()
    actual = _make_expected()
    actual["error_explanations"] = actual["error_explanations"][:1]
    result = compute_error_explanation_diffs(expected, actual)
    assert result[-1]["status"] == "missed"


# ── compute_metadata_diffs ───────────────────────────────────────────────────

def test_metadata_all_match():
    expected = _make_expected()
    result = compute_metadata_diffs(expected, expected)
    assert all(m["match"] for m in result)


def test_metadata_case_insensitive():
    expected = _make_expected()
    actual = _make_expected()
    actual["metadata"]["course_type"] = "KEYSTONE"
    result = compute_metadata_diffs(expected, actual)
    f = next(m for m in result if m["field"] == "course_type")
    assert f["match"] is True


def test_metadata_partial_mismatch():
    expected = _make_expected()
    actual = _make_expected()
    actual["metadata"]["class"] = "C1"
    result = compute_metadata_diffs(expected, actual)
    mismatches = [m for m in result if not m["match"]]
    assert len(mismatches) == 1
    assert mismatches[0]["field"] == "class"


# ── compute_writing_performance_diffs ────────────────────────────────────────

def test_wp_diffs_exact():
    expected = _make_expected()
    result = compute_writing_performance_diffs(expected, expected)
    assert all(w["status"] == "exact" for w in result)


def test_wp_diffs_within_tolerance():
    expected = _make_expected()
    actual = _make_expected()
    actual["writing_performance"]["grammar"]["score"] = 75.2  # diff = 1.5
    result = compute_writing_performance_diffs(expected, actual)
    g = next(w for w in result if w["domain"] == "grammar")
    assert g["status"] == "within_3"


def test_wp_diffs_off():
    expected = _make_expected()
    actual = _make_expected()
    actual["writing_performance"]["total_score"]["score"] = 60.0  # diff = 17
    result = compute_writing_performance_diffs(expected, actual)
    ts = next(w for w in result if w["domain"] == "total_score")
    assert ts["status"] == "off"
    assert ts["delta"] == pytest.approx(17.0)


# ── compute_error_analysis_diff ──────────────────────────────────────────────

def test_error_analysis_diff_exact():
    expected = _make_expected()
    result = compute_error_analysis_diff(expected, expected)
    assert all(v["match"] for v in result.values())
    assert all(v["diff"] == 0 for v in result.values())


def test_error_analysis_diff_mismatch():
    expected = _make_expected()
    actual = _make_expected()
    actual["error_analysis"]["grammar"] = 3
    actual["error_analysis"]["total_errors"] = 4
    result = compute_error_analysis_diff(expected, actual)
    assert result["grammar"]["diff"] == 2
    assert result["grammar"]["match"] is False


# ── compute_field_diffs ───────────────────────────────────────────────────────

def test_field_diffs_flatten_expected_and_actual_values():
    expected = _make_expected()
    actual = _make_expected()
    actual["metadata"]["class"] = "C1"
    actual["writing_performance"]["total_score"]["score"] = 70.0
    result = compute_field_diffs(expected, actual)
    cls = next(d for d in result if d["path"] == "metadata.class")
    total = next(d for d in result if d["path"] == "writing_performance.total_score.score")
    assert cls["expected_value"] == "B2"
    assert cls["actual_value"] == "C1"
    assert cls["status"] == "different"
    assert total["status"] == "off_by_large"


# ── compute_text_char_diffs ──────────────────────────────────────────────────

def test_text_char_diffs_identical():
    expected = _make_expected()
    result = compute_text_char_diffs(expected, expected)
    for t in result:
        assert t["insert_chars"] == 0
        assert t["delete_chars"] == 0
        assert t["replace_chars"] == 0
        assert t["char_accuracy"] == pytest.approx(1.0)
        assert t["similarity"] == pytest.approx(1.0)


def test_text_char_diffs_partial():
    expected = _make_expected()
    actual = _make_expected()
    actual["original_writing"] = "I go to class."
    result = compute_text_char_diffs(expected, actual)
    orig = next(t for t in result if t["field"] == "original_writing")
    assert orig["similarity"] < 1.0
    total = orig["equal_chars"] + orig["insert_chars"] + orig["delete_chars"] + orig["replace_chars"]
    assert total > 0


# ── compute_internal_consistency ─────────────────────────────────────────────

def _valid_schema():
    from src.json_validator import validate_output
    data = _make_expected()
    return validate_output(data)


def test_internal_consistency_fully_consistent():
    data = _make_expected()
    schema_val = _valid_schema()
    result = compute_internal_consistency(data, schema_val)
    # Tags: 2 (G + S), explanations: 2, error_analysis.total: 2
    assert result["tag_count_in_original"] == 2
    assert result["error_explanations_count"] == 2
    assert result["error_analysis_total"] == 2
    assert result["tag_exp_analysis_consistent"] is True
    assert result["consistency_score"] > 0.5


def test_internal_consistency_tag_count_mismatch():
    data = _make_expected()
    data["error_explanations"] = data["error_explanations"][:1]  # only 1
    schema_val = _valid_schema()
    result = compute_internal_consistency(data, schema_val)
    assert result["tag_exp_analysis_consistent"] is False
    assert result["consistency_score"] < 1.0


# ── compute_diff (integration) ───────────────────────────────────────────────

def test_compute_diff_none_actual():
    expected = _make_expected()
    result = compute_diff(expected, None, {})
    assert "subcategory_diffs" in result
    assert "per_tag_f1" in result
    assert "internal_consistency" in result
    assert result["summary_flags"]["any_subcategory_mismatch"] is False


def test_compute_diff_none_expected():
    actual = _make_expected()
    from src.json_validator import validate_output
    schema_val = validate_output(actual)
    result = compute_diff(None, actual, schema_val)
    assert "internal_consistency" in result
    assert result["internal_consistency"]["consistency_score"] > 0.0


def test_compute_diff_summary_flags_no_mismatch():
    expected = _make_expected()
    from src.json_validator import validate_output
    schema_val = validate_output(expected)
    result = compute_diff(expected, expected, schema_val)
    assert result["summary_flags"]["any_subcategory_mismatch"] is False
    assert result["summary_flags"]["metadata_all_match"] is True
    assert result["summary_flags"]["writing_performance_all_exact"] is True


def test_compute_diff_summary_flags_with_mismatch():
    expected = _make_expected()
    actual = _make_expected()
    actual["scoring_analysis"]["grammar"]["sentence_accuracy"]["score"] = 1
    from src.json_validator import validate_output
    schema_val = validate_output(actual)
    result = compute_diff(expected, actual, schema_val)
    assert result["summary_flags"]["any_subcategory_mismatch"] is True
