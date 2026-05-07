import re
from difflib import SequenceMatcher

ERROR_TAG_PATTERN = re.compile(r"\[([GVOPSC])\]", re.IGNORECASE)
TAG_MAP = {"G": "grammar", "V": "vocabulary", "O": "word_order",
           "P": "punctuation", "S": "spelling", "C": "coherence"}


def normalize_text(text: str) -> str:
    text = re.sub(r"\[([GVOPSC])\]", "", text, flags=re.IGNORECASE)
    return " ".join(text.lower().split())


def strip_error_tags(text: str) -> str:
    return re.sub(r"\[([GVOPSC])\]", "", text, flags=re.IGNORECASE)


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    na, nb = normalize_text(a), normalize_text(b)
    return SequenceMatcher(None, na, nb).ratio()


def extract_error_tag_counts(text: str) -> dict[str, int]:
    counts = {v: 0 for v in TAG_MAP.values()}
    for m in ERROR_TAG_PATTERN.finditer(text or ""):
        key = TAG_MAP.get(m.group(1).upper())
        if key:
            counts[key] += 1
    return counts


def extract_error_tag_counts_from_output(output_json: dict) -> dict[str, int]:
    original = output_json.get("original_writing", "") or ""
    counts = extract_error_tag_counts(original)
    if sum(counts.values()) > 0:
        return counts

    # Some models omit tags in original_writing but include them in explanation
    # fields. Use those as a fallback so comparison does not collapse to zero.
    for item in output_json.get("error_explanations", []) or []:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get(k, "") or "") for k in ("error", "explanation", "explanation_en"))
        item_counts = extract_error_tag_counts(text)
        for key, value in item_counts.items():
            counts[key] += value
    if sum(counts.values()) > 0:
        return counts

    analysis = output_json.get("error_analysis", {}) or {}
    for key in counts:
        try:
            counts[key] = int(analysis.get(key, 0) or 0)
        except (TypeError, ValueError):
            counts[key] = 0
    return counts


def calculate_tag_precision_recall_f1(
    expected_counts: dict[str, int],
    actual_counts: dict[str, int],
) -> dict:
    all_keys = set(expected_counts) | set(actual_counts)
    tp = sum(min(expected_counts.get(k, 0), actual_counts.get(k, 0)) for k in all_keys)
    total_expected = sum(expected_counts.values())
    total_actual = sum(actual_counts.values())

    precision = tp / total_actual if total_actual else 0.0
    recall = tp / total_expected if total_expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _get_subtotal(scoring: dict, category: str) -> float | None:
    cat = scoring.get(category, {})
    sub = cat.get("subtotal")
    if isinstance(sub, dict):
        return sub.get("score")
    if isinstance(sub, (int, float)):
        return float(sub)
    return None


def _get_total_score(wp: dict) -> float | None:
    ts = wp.get("total_score")
    if isinstance(ts, dict):
        return ts.get("score")
    if isinstance(ts, (int, float)):
        return float(ts)
    return None


def calculate_score_differences(expected: dict, actual: dict) -> dict:
    def sa(d):
        return d.get("scoring_analysis", {})

    def wp(d):
        return d.get("writing_performance", {})

    result = {}
    for cat in ("grammar", "vocabulary", "writing_flow"):
        exp_val = _get_subtotal(sa(expected), cat)
        act_val = _get_subtotal(sa(actual), cat)
        if exp_val is not None and act_val is not None:
            result[f"{cat}_subtotal_difference"] = round(abs(exp_val - act_val), 4)
        else:
            result[f"{cat}_subtotal_difference"] = None

    exp_ts = _get_total_score(wp(expected))
    act_ts = _get_total_score(wp(actual))
    result["total_score_difference"] = (
        round(abs(exp_ts - act_ts), 4) if exp_ts is not None and act_ts is not None else None
    )
    return result


def calculate_error_analysis_difference(expected: dict, actual: dict) -> dict:
    exp_ea = expected.get("error_analysis", {})
    act_ea = actual.get("error_analysis", {})
    keys = ["grammar", "vocabulary", "word_order", "punctuation", "spelling", "coherence", "total_errors"]
    diff = {}
    for k in keys:
        ev = exp_ea.get(k, 0) or 0
        av = act_ea.get(k, 0) or 0
        diff[k] = abs(ev - av)
    return diff


def score_similarity_from_difference(diff: float | None, scale: float = 100.0) -> float:
    if diff is None:
        return 0.0
    return max(0.0, 1.0 - diff / scale)


def error_analysis_similarity(category_diff: dict) -> float:
    total_diff = sum(category_diff.values())
    return max(0.0, 1.0 - total_diff / max(1, total_diff + 5))


def metadata_match_score(expected: dict, actual: dict) -> float:
    exp_meta = expected.get("metadata", {})
    act_meta = actual.get("metadata", {})
    keys = ["course_type", "class", "writing_type"]
    matches = sum(
        1 for k in keys
        if str(exp_meta.get(k, "")).lower() == str(act_meta.get(k, "")).lower()
    )
    return round(matches / len(keys), 4)


def calculate_overall_accuracy_score(comparison: dict) -> float:
    schema_compliance = comparison.get("schema_compliance_score", 0.0) or 0.0
    orig_sim = comparison.get("original_writing_similarity", 0.0) or 0.0
    corr_sim = comparison.get("corrected_writing_similarity", 0.0) or 0.0
    tag_f1 = comparison.get("error_tag_f1", 0.0) or 0.0
    score_sim = score_similarity_from_difference(comparison.get("total_score_difference"))
    ea_diff = comparison.get("error_analysis_category_difference", {})
    ea_sim = error_analysis_similarity(ea_diff)

    overall = (
        schema_compliance * 0.20
        + orig_sim * 0.20
        + corr_sim * 0.20
        + tag_f1 * 0.20
        + score_sim * 0.10
        + ea_sim * 0.10
    )
    return round(overall, 4)
