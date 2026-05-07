import json
from datetime import datetime
from pathlib import Path

from src.comparator import compare
from src.config_loader import get_models_by_names
from src.aws_model_storage import build_model_artifact_manifest
from src.inference_client_factory import create_inference_client
from src.json_validator import validate_output
from src.ocr_processor import OCRProcessor, release_doctr_predictor
from src.output_parser import parse_model_output
from src.prompt_builder import build_compact_prompt_from_ocr_json, build_prompt_from_ocr_json
from src.report_generator import (
    build_summary,
    save_combined_prompt_to_report,
    save_comparison,
    save_ocr_result,
    save_parsed_output,
    save_raw_output,
    save_report,
    save_summary,
    save_validation,
)
from src.utils import PROJECT_ROOT, ensure_dir, sanitize_run_name, setup_logging, timestamp


def _load_json_file(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _default_prompt_path(filename: str) -> Path:
    return PROJECT_ROOT / "data" / "prompts" / filename


def _format_template(value):
    if isinstance(value, dict):
        return {k: _format_template(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_format_template(value[0])] if value else []
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return 0
    if value is None:
        return None
    return ""


def _default_for_expected(value):
    if isinstance(value, dict):
        return {k: _default_for_expected(v) for k, v in value.items()}
    if isinstance(value, list):
        return []
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return 0
    if value is None:
        return None
    return ""


def _align_to_expected_format(actual, expected):
    """Return actual values constrained to the exact JSON shape of expected."""
    if expected is None:
        return actual
    if isinstance(expected, dict):
        actual_dict = actual if isinstance(actual, dict) else {}
        return {
            key: _align_to_expected_format(actual_dict.get(key), expected_value)
            for key, expected_value in expected.items()
        }
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return []
        if not expected:
            return actual
        template = expected[0]
        return [_align_to_expected_format(item, template) for item in actual]
    return actual if actual is not None else _default_for_expected(expected)


def _append_expected_format(prompt: str, expected: dict | None) -> str:
    if not expected:
        return prompt
    template = json.dumps(_format_template(expected), ensure_ascii=False, indent=2)
    return (
        prompt
        + "\n\n[EXPECTED ANALYSIS JSON FORMAT - STRUCTURE ONLY]\n"
        + template
        + "\n\n[FORMAT INSTRUCTION]\n"
        + "The final answer must use the same top-level keys and nested JSON shape as the structure above. "
        + "Do not copy answer values from the format reference; fill all values from the OCR text and prompts.\n"
    )


def _save_model_comparison_report(run_name: str, model_name: str, comparison: dict) -> str | None:
    detail = comparison.get("detail_diff") or {}
    if not detail:
        return None
    report_dir = ensure_dir(PROJECT_ROOT / "results" / "reports" / run_name)
    path = report_dir / f"{model_name}_analysis_comparison.md"
    lines = [
        f"# {model_name} Analysis JSON Comparison",
        "",
        "## Summary",
        "",
        f"- JSON parse success: {comparison.get('json_parse_success')}",
        f"- Schema compliance: {comparison.get('schema_compliance_score')}",
        f"- Overall accuracy: {comparison.get('overall_accuracy_score')}",
        f"- Original writing similarity: {comparison.get('original_writing_similarity')}",
        f"- Corrected writing similarity: {comparison.get('corrected_writing_similarity')}",
        f"- Error tag F1: {comparison.get('error_tag_f1')}",
        f"- Total score difference: {comparison.get('total_score_difference')}",
        "",
        "## Field Differences",
        "",
        "| JSON Path | Expected | Actual | Similarity | Status |",
        "|---|---|---|---:|---|",
    ]
    field_diffs = [row for row in detail.get("field_diffs", []) if row.get("status") != "match"]
    for row in field_diffs[:80]:
        exp = str(row.get("expected_value", "")).replace("|", "\\|").replace("\n", " ")
        act = str(row.get("actual_value", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{row.get('path')}` | {exp[:160]} | {act[:160]} | "
            f"{row.get('similarity')} | {row.get('status')} |"
        )
    if not field_diffs:
        lines.append("| All fields | Match | Match | 1.0 | match |")
    lines += [
        "",
        "## Score Differences",
        "",
        "| Domain | Expected | Actual | Delta | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in detail.get("writing_performance_diffs", []):
        lines.append(
            f"| {row.get('domain')} | {row.get('expected_score')} | {row.get('actual_score')} | "
            f"{row.get('delta')} | {row.get('status')} |"
        )
    lines += [
        "",
        "## Error Count Differences",
        "",
        "| Category | Expected | Actual | Diff | Match |",
        "|---|---:|---:|---:|---|",
    ]
    for key, row in (detail.get("error_analysis_diff") or {}).items():
        lines.append(
            f"| {key} | {row.get('expected')} | {row.get('actual')} | "
            f"{row.get('diff')} | {row.get('match')} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path.relative_to(PROJECT_ROOT))


def run_writing_analysis(
    *,
    run_name: str | None = None,
    image_path: str | None = None,
    ocr_json_path: str | None = None,
    system_prompt_path: str | None = None,
    output_prompt_path: str | None = None,
    expected_output_path: str | None = None,
    model_names: list[str] | None = None,
    inference_mode: str = "remote",
) -> dict:
    run_name = sanitize_run_name(run_name or f"api_run_{timestamp()}")
    logger = setup_logging(run_name)
    started_at = datetime.now().isoformat()

    sys_path = Path(system_prompt_path) if system_prompt_path else _default_prompt_path("system_prompt.txt")
    out_path = Path(output_prompt_path) if output_prompt_path else _default_prompt_path("output_prompt.txt")
    system_prompt = sys_path.read_text(encoding="utf-8")
    output_prompt = out_path.read_text(encoding="utf-8")

    processor = OCRProcessor()
    input_type = "ocr_json" if ocr_json_path else "image"
    if ocr_json_path:
        ocr_result = processor.load_ocr_json(ocr_json_path)
        image_base64 = None
    elif image_path:
        ocr_result = processor.process_image(image_path)
        image_base64 = None
    else:
        raise ValueError("image_path or ocr_json_path is required")

    ocr_json = ocr_result.model_dump(by_alias=True)
    if image_path and not (ocr_json.get("handwritten_text") or "").strip():
        warnings = "; ".join(ocr_json.get("warnings") or [])
        raise ValueError(f"OCR failed before model analysis. {warnings}".strip())
    ocr_artifact = save_ocr_result(ocr_json, run_name)
    if image_path:
        release_doctr_predictor()

    expected = _load_json_file(expected_output_path)
    combined_prompt = _append_expected_format(
        build_prompt_from_ocr_json(system_prompt, output_prompt, ocr_json),
        expected,
    )
    combined_prompt_path = save_combined_prompt_to_report(combined_prompt, run_name)
    compact_prompt = _append_expected_format(build_compact_prompt_from_ocr_json(ocr_json), expected)

    model_configs = get_models_by_names(model_names)
    model_results = []
    comparisons = []
    api_models = []
    artifact_paths = {
        "ocr_json": str(ocr_artifact.relative_to(PROJECT_ROOT)),
        "summary": str((PROJECT_ROOT / "results" / "reports" / run_name / "benchmark_summary.json").relative_to(PROJECT_ROOT)),
        "report": str((PROJECT_ROOT / "results" / "reports" / run_name / "benchmark_report.md").relative_to(PROJECT_ROOT)),
    }

    for model_config in model_configs:
        model_name = model_config["name"]
        logger.info("Calling model %s via %s", model_name, inference_mode)
        client = create_inference_client(model_config, inference_mode)
        model_prompt = compact_prompt if model_config.get("prompt_mode") == "compact" else combined_prompt
        result = client.generate(model_config, model_prompt, ocr_json, image_base64=image_base64)
        result["model_name"] = model_name
        result.setdefault("model", model_name)
        result.setdefault("provider", model_config.get("provider", "ollama_local"))
        model_results.append(result)
        save_raw_output(result.get("raw_text", ""), run_name, model_name)

        parse_result = parse_model_output(result.get("raw_text", ""))
        parsed_json = parse_result.get("parsed_json")
        if parse_result.get("parse_success") and expected is not None:
            parsed_json = _align_to_expected_format(parsed_json, expected)
            parse_result["parsed_json"] = parsed_json
        save_parsed_output(parsed_json, run_name, model_name)

        validation = validate_output(parsed_json)
        save_validation(validation, run_name, model_name)

        comparison = compare(expected, parse_result)
        comparison_report_path = None
        if expected is not None:
            save_comparison(comparison, run_name, model_name)
            comparison_report_path = _save_model_comparison_report(run_name, model_name, comparison)
        comparisons.append(comparison)

        api_models.append({
            "model_name": model_name,
            "provider": result.get("provider"),
            "model_storage": build_model_artifact_manifest(model_config),
            "success": result.get("success", False),
            "duration_seconds": result.get("duration_seconds"),
            "parse_success": parse_result.get("parse_success", False),
            "schema_compliance_score": validation.get("schema_compliance_score", 0.0),
            "analysis": parsed_json,
            "validation": validation,
            "comparison": comparison if expected is not None else None,
            "comparison_report": comparison_report_path,
            "error": result.get("error") or parse_result.get("error"),
        })

    ended_at = datetime.now().isoformat()
    summary = build_summary(
        run_name=run_name,
        image_path=image_path or "",
        system_prompt_file=str(sys_path),
        output_prompt_file=str(out_path),
        combined_prompt_file=str(combined_prompt_path),
        expected_output_file=expected_output_path or "",
        started_at=started_at,
        ended_at=ended_at,
        model_results=model_results,
        comparisons=comparisons,
    )
    if expected is not None:
        summary["expected_output"] = expected
    save_summary(summary, run_name)
    save_report(summary, run_name)

    return {
        "run_name": run_name,
        "inference_mode": inference_mode,
        "input_type": input_type,
        "ocr_json": ocr_json,
        "models": api_models,
        "winner": summary.get("winner", {}),
        "artifacts": artifact_paths,
        "benchmark_summary": summary,
    }
