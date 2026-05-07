"""Field-level diff engine: compares model output against expected (정답 분석지)."""

import re
from difflib import SequenceMatcher

_TAG_RE = re.compile(r"\[([GVOPSC])\]", re.IGNORECASE)
_TAG_MAP = {
    "G": "grammar", "V": "vocabulary", "O": "word_order",
    "P": "punctuation", "S": "spelling", "C": "coherence",
}
_TAG_LABELS = {
    "grammar": "[G] Grammar", "vocabulary": "[V] Vocabulary",
    "word_order": "[O] Word Order", "punctuation": "[P] Punctuation",
    "spelling": "[S] Spelling", "coherence": "[C] Coherence",
}


def _sim(a: str, b: str) -> float:
    a = " ".join(_TAG_RE.sub("", str(a)).lower().split())
    b = " ".join(_TAG_RE.sub("", str(b)).lower().split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio(), 4)


def _score_val(v) -> float | None:
    if isinstance(v, dict):
        return v.get("score")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _max_val(v) -> float | None:
    if isinstance(v, dict):
        return v.get("max_score")
    return None


def compute_subcategory_diffs(expected: dict, actual: dict) -> dict:
    """Per-subcategory score comparison across all scoring_analysis categories."""
    exp_sa = expected.get("scoring_analysis", {}) or {}
    act_sa = actual.get("scoring_analysis", {}) or {}
    result = {}
    for cat in sorted(set(list(exp_sa) + list(act_sa))):
        exp_cat = exp_sa.get(cat, {}) or {}
        act_cat = act_sa.get(cat, {}) or {}
        entries = []
        for key in sorted(set(list(exp_cat) + list(act_cat))):
            ev = _score_val(exp_cat.get(key))
            av = _score_val(act_cat.get(key))
            max_sc = _max_val(exp_cat.get(key)) or _max_val(act_cat.get(key))
            delta = round(abs(ev - av), 4) if ev is not None and av is not None else None
            if ev is None or av is None:
                status = "missing"
            elif delta == 0:
                status = "match"
            elif delta <= 2:
                status = "off_by_small"
            else:
                status = "off_by_large"
            entries.append({
                "subcategory": key,
                "expected_score": ev,
                "actual_score": av,
                "max_score": max_sc,
                "score_delta": delta,
                "exact_match": (ev == av),
                "status": status,
            })
        result[cat] = entries
    return result


def compute_per_tag_f1(expected: dict, actual: dict) -> list:
    """Per error-type (G/V/O/P/S/C) precision/recall/F1."""
    from src.metrics import extract_error_tag_counts_from_output
    exp_counts = extract_error_tag_counts_from_output(expected)
    act_counts = extract_error_tag_counts_from_output(actual)
    result = []
    for cat_name, label in _TAG_LABELS.items():
        ec = exp_counts.get(cat_name, 0)
        ac = act_counts.get(cat_name, 0)
        tp = min(ec, ac)
        fp = max(0, ac - ec)
        fn = max(0, ec - ac)
        precision = round(tp / ac, 4) if ac else 0.0
        recall = round(tp / ec, 4) if ec else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        if ec == 0 and ac == 0:
            status = "n/a"
        elif f1 >= 1.0:
            status = "perfect"
        elif ec == 0 and ac > 0:
            status = "over_tagged"
        elif fp > 0 and fn == 0:
            status = "over_tagged"
        elif ec > 0 and ac == 0:
            status = "missed"
        elif fn > 0 and fp == 0:
            status = "under_tagged"
        else:
            status = "partial"
        result.append({
            "tag": cat_name[0].upper(),
            "tag_name": cat_name,
            "label": label,
            "expected_count": ec,
            "actual_count": ac,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "status": status,
        })
    return result


def compute_error_explanation_diffs(expected: dict, actual: dict) -> list:
    """Align error_explanations lists by index and compare tags."""
    _TAG_EXTRACT = re.compile(r"\[([GVOPSC])\]", re.IGNORECASE)
    exp_list = expected.get("error_explanations", []) or []
    act_list = actual.get("error_explanations", []) or []
    max_len = max(len(exp_list), len(act_list))
    result = []
    for i in range(max_len):
        e_item = exp_list[i] if i < len(exp_list) else None
        a_item = act_list[i] if i < len(act_list) else None

        def tag_from(item):
            if not item:
                return None
            text = " ".join(str(item.get(k, "") or "") for k in ("error", "explanation", "explanation_en"))
            m = _TAG_EXTRACT.search(text)
            return m.group(1).upper() if m else None

        e_tag = tag_from(e_item)
        a_tag = tag_from(a_item)
        tag_match = (e_tag == a_tag) if e_tag and a_tag else False

        if e_item is None:
            status = "extra"
        elif a_item is None:
            status = "missed"
        elif tag_match:
            status = "matched"
        else:
            status = "tag_mismatch"

        result.append({
            "index": i,
            "expected_error": e_item.get("error") if e_item else None,
            "actual_error": a_item.get("error") if a_item else None,
            "expected_tag": e_tag,
            "actual_tag": a_tag,
            "tag_match": tag_match,
            "status": status,
        })
    return result


def _flatten_leaf_fields(value, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_leaf_fields(child, path))
        return result
    if isinstance(value, list):
        result = {}
        for idx, child in enumerate(value):
            path = f"{prefix}[{idx}]"
            result.update(_flatten_leaf_fields(child, path))
        if not value and prefix:
            result[prefix] = []
        return result
    return {prefix: value}


def _value_similarity(expected_value, actual_value) -> float:
    if expected_value == actual_value:
        return 1.0
    if expected_value is None or actual_value is None:
        return 0.0
    if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
        scale = max(abs(float(expected_value)), abs(float(actual_value)), 1.0)
        return round(max(0.0, 1.0 - abs(float(expected_value) - float(actual_value)) / scale), 4)
    return _sim(str(expected_value), str(actual_value))


def compute_field_diffs(expected: dict, actual: dict, limit: int = 300) -> list:
    """Flatten expected/model JSON and compare every leaf field by path."""
    exp_fields = _flatten_leaf_fields(expected)
    act_fields = _flatten_leaf_fields(actual)
    rows = []
    for path in sorted(set(exp_fields) | set(act_fields)):
        ev = exp_fields.get(path)
        av = act_fields.get(path)
        present_expected = path in exp_fields
        present_actual = path in act_fields
        similarity = _value_similarity(ev, av) if present_expected and present_actual else 0.0
        if not present_expected:
            status = "extra"
        elif not present_actual:
            status = "missing"
        elif ev == av:
            status = "match"
        elif isinstance(ev, (int, float)) and isinstance(av, (int, float)):
            status = "off_by_small" if abs(float(ev) - float(av)) <= 2 else "off_by_large"
        elif similarity >= 0.9:
            status = "near_match"
        else:
            status = "different"
        rows.append({
            "path": path,
            "expected_value": ev,
            "actual_value": av,
            "expected_present": present_expected,
            "actual_present": present_actual,
            "match": status == "match",
            "similarity": similarity,
            "status": status,
        })
    return rows[:limit]


def compute_metadata_diffs(expected: dict, actual: dict) -> list:
    """Field-level metadata comparison."""
    exp_m = expected.get("metadata", {}) or {}
    act_m = actual.get("metadata", {}) or {}
    fields = ["course_type", "class", "title", "title_corrected",
              "topic", "topic_corrected", "writing_type"]
    return [
        {
            "field": f,
            "expected_value": str(exp_m.get(f, "")).strip(),
            "actual_value": str(act_m.get(f, "")).strip(),
            "match": str(exp_m.get(f, "")).strip().lower() == str(act_m.get(f, "")).strip().lower(),
        }
        for f in fields
    ]


def compute_writing_performance_diffs(expected: dict, actual: dict) -> list:
    """Writing performance domain score comparison."""
    exp_wp = expected.get("writing_performance", {}) or {}
    act_wp = actual.get("writing_performance", {}) or {}
    result = []
    for f in ["grammar", "vocabulary", "writing_flow", "total_score"]:
        ev = _score_val(exp_wp.get(f))
        av = _score_val(act_wp.get(f))
        delta = round(abs(ev - av), 4) if ev is not None and av is not None else None
        if delta is None:
            status = "missing"
        elif delta == 0:
            status = "exact"
        elif delta <= 1:
            status = "within_1"
        elif delta <= 3:
            status = "within_3"
        else:
            status = "off"
        result.append({"domain": f, "expected_score": ev, "actual_score": av,
                        "delta": delta, "status": status})
    return result


def compute_error_analysis_diff(expected: dict, actual: dict) -> dict:
    """Compare error_analysis count fields (expected vs actual per category)."""
    keys = ["grammar", "vocabulary", "word_order", "punctuation",
            "spelling", "coherence", "total_errors"]
    exp_ea = expected.get("error_analysis", {}) or {}
    act_ea = actual.get("error_analysis", {}) or {}
    return {
        k: {
            "expected": exp_ea.get(k, 0) or 0,
            "actual": act_ea.get(k, 0) or 0,
            "diff": abs((exp_ea.get(k, 0) or 0) - (act_ea.get(k, 0) or 0)),
            "match": (exp_ea.get(k, 0) or 0) == (act_ea.get(k, 0) or 0),
        }
        for k in keys
    }


def compute_text_char_diffs(expected: dict, actual: dict) -> list:
    """Character-level diff stats on original_writing and corrected_writing."""
    result = []
    for f in ["original_writing", "corrected_writing"]:
        exp_t = expected.get(f, "") or ""
        act_t = actual.get(f, "") or ""
        ins = dl = rep = eq = 0
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, exp_t, act_t).get_opcodes():
            if tag == "equal":
                eq += i2 - i1
            elif tag == "insert":
                ins += j2 - j1
            elif tag == "delete":
                dl += i2 - i1
            elif tag == "replace":
                rep += max(i2 - i1, j2 - j1)
        total = max(len(exp_t), len(act_t)) or 1
        result.append({
            "field": f,
            "expected_len": len(exp_t),
            "actual_len": len(act_t),
            "insert_chars": ins,
            "delete_chars": dl,
            "replace_chars": rep,
            "equal_chars": eq,
            "char_accuracy": round(eq / total, 4),
            "similarity": _sim(exp_t, act_t),
        })
    return result


def compute_internal_consistency(actual: dict, schema_validation: dict) -> dict:
    """Inferred self-consistency metrics — no expected output needed."""
    original = actual.get("original_writing", "") or ""
    tag_count = len(_TAG_RE.findall(original))

    exp_list = actual.get("error_explanations", []) or []
    explanations_count = len(exp_list) if isinstance(exp_list, list) else 0

    ea = actual.get("error_analysis", {}) or {}
    ea_total = ea.get("total_errors", 0) or 0

    tags_consistent = (tag_count == explanations_count == ea_total)
    subtotal_math = schema_validation.get("score_math_valid", False)

    # Validate: total_score == sum of all subcategory subtotals
    sa = actual.get("scoring_analysis", {}) or {}
    g_sub = _score_val((sa.get("grammar") or {}).get("subtotal"))
    v_sub = _score_val((sa.get("vocabulary") or {}).get("subtotal"))
    f_sub = _score_val((sa.get("writing_flow") or {}).get("subtotal"))
    wp = actual.get("writing_performance", {}) or {}
    ts = _score_val(wp.get("total_score"))
    total_formula_ok = True
    if g_sub is not None and v_sub is not None and f_sub is not None and ts is not None:
        if abs((g_sub + v_sub + f_sub) - ts) > 1.0:
            total_formula_ok = False

    # Validate: writing_performance domain percentages match subtotal/max*100
    perf_math_ok = True
    for domain, sa_key in [("grammar", "grammar"), ("vocabulary", "vocabulary"), ("writing_flow", "writing_flow")]:
        cat = sa.get(sa_key, {}) or {}
        sub = cat.get("subtotal")
        sub_score = _score_val(sub)
        sub_max = _max_val(sub)
        wp_score = _score_val(wp.get(domain))
        if sub_score is not None and sub_max and wp_score is not None:
            expected_pct = round(sub_score / sub_max * 100, 1)
            if abs(expected_pct - wp_score) > 0.6:
                perf_math_ok = False

    # Silent fix: significant text changes in corrected but zero tags in original
    corrected = actual.get("corrected_writing", "") or ""
    orig_clean = _TAG_RE.sub("", original).split()
    corr_words = corrected.split()
    silent_fix = False
    if orig_clean and corr_words and tag_count == 0:
        changed_ops = sum(
            1 for tag, *_ in SequenceMatcher(None, orig_clean, corr_words).get_opcodes()
            if tag in ("replace", "delete", "insert")
        )
        if changed_ops > 0:
            silent_fix = True

    checks_passed = sum([tags_consistent, subtotal_math, perf_math_ok, total_formula_ok, not silent_fix])
    return {
        "tag_count_in_original": tag_count,
        "error_explanations_count": explanations_count,
        "error_analysis_total": ea_total,
        "tag_exp_analysis_consistent": tags_consistent,
        "subtotal_math_valid": subtotal_math,
        "writing_performance_math_valid": perf_math_ok,
        "total_score_formula_valid": total_formula_ok,
        "silent_fix_detected": silent_fix,
        "consistency_score": round(checks_passed / 5, 4),
    }


_EMPTY_DIFF: dict = {
    "subcategory_diffs": {},
    "per_tag_f1": [],
    "error_explanation_diffs": [],
    "metadata_diffs": [],
    "writing_performance_diffs": [],
    "error_analysis_diff": {},
    "text_char_diffs": [],
    "field_diffs": [],
    "internal_consistency": {
        "tag_count_in_original": 0, "error_explanations_count": 0,
        "error_analysis_total": 0, "tag_exp_analysis_consistent": False,
        "subtotal_math_valid": False, "writing_performance_math_valid": False,
        "total_score_formula_valid": False, "silent_fix_detected": False,
        "consistency_score": 0.0,
    },
    "summary_flags": {
        "any_subcategory_mismatch": False, "any_tag_f1_below_threshold": False,
        "metadata_all_match": False, "writing_performance_all_exact": False,
        "internally_consistent": False,
    },
}


def compute_diff(
    expected: dict | None,
    actual: dict | None,
    schema_validation: dict,
) -> dict:
    """Master entry point. Returns the complete detail_diff dict."""
    import copy

    if actual is None:
        return copy.deepcopy(_EMPTY_DIFF)

    consistency = compute_internal_consistency(actual, schema_validation)

    if expected is None:
        result = copy.deepcopy(_EMPTY_DIFF)
        result["internal_consistency"] = consistency
        result["summary_flags"]["internally_consistent"] = consistency["consistency_score"] >= 0.8
        return result

    sub_diffs = compute_subcategory_diffs(expected, actual)
    tag_f1 = compute_per_tag_f1(expected, actual)
    exp_diffs = compute_error_explanation_diffs(expected, actual)
    meta_diffs = compute_metadata_diffs(expected, actual)
    wp_diffs = compute_writing_performance_diffs(expected, actual)
    ea_diff = compute_error_analysis_diff(expected, actual)
    text_diffs = compute_text_char_diffs(expected, actual)
    field_diffs = compute_field_diffs(expected, actual)

    any_mismatch = any(
        e["status"] != "match"
        for entries in sub_diffs.values()
        for e in entries
    )
    any_f1_low = any(t["f1"] < 0.5 and t["expected_count"] > 0 for t in tag_f1)
    meta_all = all(m["match"] for m in meta_diffs)
    wp_all_exact = all(w["status"] == "exact" for w in wp_diffs)

    return {
        "subcategory_diffs": sub_diffs,
        "per_tag_f1": tag_f1,
        "error_explanation_diffs": exp_diffs,
        "metadata_diffs": meta_diffs,
        "writing_performance_diffs": wp_diffs,
        "error_analysis_diff": ea_diff,
        "text_char_diffs": text_diffs,
        "field_diffs": field_diffs,
        "internal_consistency": consistency,
        "summary_flags": {
            "any_subcategory_mismatch": any_mismatch,
            "any_tag_f1_below_threshold": any_f1_low,
            "metadata_all_match": meta_all,
            "writing_performance_all_exact": wp_all_exact,
            "internally_consistent": consistency["consistency_score"] >= 0.8,
        },
    }
