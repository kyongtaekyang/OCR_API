from src.json_validator import validate_output
from src.diff_engine import compute_diff
from src.metrics import (
    text_similarity,
    extract_error_tag_counts_from_output,
    calculate_tag_precision_recall_f1,
    calculate_score_differences,
    calculate_error_analysis_difference,
    metadata_match_score,
    calculate_overall_accuracy_score,
)


def compare(expected: dict | None, model_output_parse_result: dict) -> dict:
    parse_success = model_output_parse_result.get("parse_success", False)
    parsed = model_output_parse_result.get("parsed_json") if parse_success else None

    validation = validate_output(parsed)

    base = {
        "json_parse_success": parse_success,
        "schema_type": validation["schema_type"],
        "schema_compliance_score": validation["schema_compliance_score"],
        "required_key_completeness": validation["required_key_completeness"],
        "score_math_valid": validation["score_math_valid"],
        "score_anomalies": validation["score_anomalies"],
        "metadata_match_score": 0.0,
        "original_writing_similarity": 0.0,
        "corrected_writing_similarity": 0.0,
        "error_tag_precision": 0.0,
        "error_tag_recall": 0.0,
        "error_tag_f1": 0.0,
        "error_analysis_category_difference": {},
        "total_errors_difference": 0,
        "error_explanations_count_difference": 0,
        "grammar_subtotal_difference": None,
        "vocabulary_subtotal_difference": None,
        "writing_flow_subtotal_difference": None,
        "total_score_difference": None,
        "overall_accuracy_score": 0.0,
    }

    if parsed is None or expected is None:
        return base

    base["metadata_match_score"] = metadata_match_score(expected, parsed)

    base["original_writing_similarity"] = text_similarity(
        expected.get("original_writing", ""),
        parsed.get("original_writing", ""),
    )
    base["corrected_writing_similarity"] = text_similarity(
        expected.get("corrected_writing", ""),
        parsed.get("corrected_writing", ""),
    )

    exp_tags = extract_error_tag_counts_from_output(expected)
    act_tags = extract_error_tag_counts_from_output(parsed)
    prf = calculate_tag_precision_recall_f1(exp_tags, act_tags)
    base["error_tag_precision"] = prf["precision"]
    base["error_tag_recall"] = prf["recall"]
    base["error_tag_f1"] = prf["f1"]

    score_diffs = calculate_score_differences(expected, parsed)
    base["grammar_subtotal_difference"] = score_diffs.get("grammar_subtotal_difference")
    base["vocabulary_subtotal_difference"] = score_diffs.get("vocabulary_subtotal_difference")
    base["writing_flow_subtotal_difference"] = score_diffs.get("writing_flow_subtotal_difference")
    base["total_score_difference"] = score_diffs.get("total_score_difference")

    ea_diff = calculate_error_analysis_difference(expected, parsed)
    base["error_analysis_category_difference"] = ea_diff
    base["total_errors_difference"] = ea_diff.get("total_errors", 0)

    exp_exp = expected.get("error_explanations", [])
    act_exp = parsed.get("error_explanations", [])
    base["error_explanations_count_difference"] = abs(
        len(exp_exp if isinstance(exp_exp, list) else [])
        - len(act_exp if isinstance(act_exp, list) else [])
    )

    base["overall_accuracy_score"] = calculate_overall_accuracy_score(base)
    base["detail_diff"] = compute_diff(expected, parsed, validation)
    return base
