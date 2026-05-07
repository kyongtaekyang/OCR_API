"""Practical JSON schema validation for benchmark outputs."""

SYSTEM_PROMPT_REQUIRED_KEYS = [
    "metadata",
    "original_writing",
    "corrected_writing",
    "scoring_analysis",
    "writing_performance",
    "error_explanations",
    "error_analysis",
    "overall_comments",
]

SYSTEM_PROMPT_METADATA_KEYS = [
    "course_type", "class", "title", "title_corrected",
    "topic", "topic_corrected", "writing_type",
]

SYSTEM_PROMPT_SCORING_PATHS = [
    ("grammar", "sentence_accuracy"),
    ("grammar", "verb_tense_consistency"),
    ("grammar", "article_preposition"),
    ("grammar", "subtotal"),
    ("vocabulary", "word_variety"),
    ("vocabulary", "appropriateness"),
    ("vocabulary", "expression_naturalness"),
    ("vocabulary", "subtotal"),
    ("writing_flow", "structure_organization"),
    ("writing_flow", "sentence_variety"),
    ("writing_flow", "coherence_transitions"),
    ("writing_flow", "subtotal"),
]

OUTPUT_PROMPT_SCORING_PATHS = [
    ("grammar", "sentence_accuracy"),
    ("grammar", "verb_tense_consistency"),
    ("grammar", "subject_verb_agreement"),
    ("grammar", "articles_prepositions"),
    ("grammar", "word_order"),
    ("grammar", "punctuation_capitalization"),
    ("grammar", "subtotal"),
    ("vocabulary", "range_of_vocabulary"),
    ("vocabulary", "word_choice_appropriateness"),
    ("vocabulary", "subtotal"),
    ("writing_flow", "organization_coherence"),
    ("writing_flow", "paragraph_unity"),
    ("writing_flow", "subtotal"),
]

ERROR_ANALYSIS_KEYS = [
    "grammar", "vocabulary", "word_order",
    "punctuation", "spelling", "coherence", "total_errors",
]


def _detect_schema_type(parsed: dict) -> str:
    sa = parsed.get("scoring_analysis", {})
    grammar = sa.get("grammar", {})
    if "subject_verb_agreement" in grammar or "articles_prepositions" in grammar:
        return "output_prompt_schema"
    if "sentence_accuracy" in grammar or "article_preposition" in grammar:
        return "system_prompt_schema"
    return "unknown"


def _check_scoring_paths(scoring: dict, paths: list[tuple]) -> tuple[int, int]:
    found = 0
    for category, key in paths:
        if key in scoring.get(category, {}):
            found += 1
    return found, len(paths)


def _get_score_value(item) -> float | None:
    if isinstance(item, dict):
        v = item.get("score")
    elif isinstance(item, (int, float)):
        v = item
    else:
        return None
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _get_max_score_value(item) -> float | None:
    if isinstance(item, dict):
        v = item.get("max_score")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return None


def detect_score_anomalies(parsed: dict) -> list[dict]:
    anomalies = []
    sa = parsed.get("scoring_analysis", {})

    for category, items in sa.items():
        if not isinstance(items, dict):
            continue
        sub_scores_sum = 0.0
        sub_count = 0
        subtotal_score = None
        subtotal_max = None

        for key, val in items.items():
            if key == "subtotal":
                subtotal_score = _get_score_value(val)
                subtotal_max = _get_max_score_value(val)
                continue
            score = _get_score_value(val)
            max_sc = _get_max_score_value(val)
            if score is not None:
                sub_scores_sum += score
                sub_count += 1
            if score is not None and max_sc is not None and score > max_sc:
                anomalies.append({
                    "path": f"scoring_analysis.{category}.{key}",
                    "issue": "score exceeds max_score",
                    "score": score,
                    "max_score": max_sc,
                })

        if subtotal_score is not None and subtotal_max is not None:
            if subtotal_score > subtotal_max:
                anomalies.append({
                    "path": f"scoring_analysis.{category}.subtotal",
                    "issue": "score exceeds max_score",
                    "score": subtotal_score,
                    "max_score": subtotal_max,
                })
            if sub_count > 0 and abs(subtotal_score - sub_scores_sum) > 0.01:
                anomalies.append({
                    "path": f"scoring_analysis.{category}.subtotal",
                    "issue": "subtotal does not equal sum of sub-scores",
                    "score": subtotal_score,
                    "expected": round(sub_scores_sum, 4),
                })

    return anomalies


def _check_score_math(parsed: dict) -> bool:
    return len(detect_score_anomalies(parsed)) == 0


def validate_output(parsed: dict | None) -> dict:
    if parsed is None:
        return {
            "valid": False,
            "schema_type": "unknown",
            "required_key_completeness": 0.0,
            "missing_keys": SYSTEM_PROMPT_REQUIRED_KEYS[:],
            "schema_compliance_score": 0.0,
            "score_math_valid": False,
            "score_anomalies": [],
            "issues": ["parsed_json is None"],
        }

    issues = []
    missing_keys = [k for k in SYSTEM_PROMPT_REQUIRED_KEYS if k not in parsed]
    required_key_completeness = 1.0 - len(missing_keys) / len(SYSTEM_PROMPT_REQUIRED_KEYS)

    schema_type = _detect_schema_type(parsed)
    scoring_paths = (
        OUTPUT_PROMPT_SCORING_PATHS
        if schema_type == "output_prompt_schema"
        else SYSTEM_PROMPT_SCORING_PATHS
    )
    sa = parsed.get("scoring_analysis", {})
    found, total = _check_scoring_paths(sa, scoring_paths)
    scoring_compliance = found / total if total else 0.0

    meta = parsed.get("metadata", {})
    meta_keys_present = sum(1 for k in SYSTEM_PROMPT_METADATA_KEYS if k in meta)
    meta_compliance = meta_keys_present / len(SYSTEM_PROMPT_METADATA_KEYS)

    ea = parsed.get("error_analysis", {})
    ea_keys_present = sum(1 for k in ERROR_ANALYSIS_KEYS if k in ea)
    ea_compliance = ea_keys_present / len(ERROR_ANALYSIS_KEYS)

    error_exp = parsed.get("error_explanations", [])
    exp_is_list = isinstance(error_exp, list)
    if not exp_is_list:
        issues.append("error_explanations is not a list")

    anomalies = detect_score_anomalies(parsed)
    score_math_valid = len(anomalies) == 0

    schema_compliance_score = round(
        (required_key_completeness * 0.4)
        + (scoring_compliance * 0.3)
        + (meta_compliance * 0.15)
        + (ea_compliance * 0.15),
        4,
    )

    for k in missing_keys:
        issues.append(f"missing required key: {k}")

    return {
        "valid": len(missing_keys) == 0 and score_math_valid,
        "schema_type": schema_type,
        "required_key_completeness": round(required_key_completeness, 4),
        "missing_keys": missing_keys,
        "schema_compliance_score": schema_compliance_score,
        "score_math_valid": score_math_valid,
        "score_anomalies": anomalies,
        "issues": issues,
    }
