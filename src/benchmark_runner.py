import json
from datetime import datetime
from pathlib import Path
from src.config_loader import get_enabled_models
from src.image_loader import load_image_as_base64
from src.prompt_builder import build_combined_prompt, build_compact_eval_prompt, save_combined_prompt
from src.ollama_client import generate_with_image
from src.output_parser import parse_model_output
from src.comparator import compare
from src.report_generator import (
    save_raw_output,
    save_parsed_output,
    save_comparison,
    save_combined_prompt_to_report,
    build_summary,
    save_summary,
    save_report,
)
from src.html_reporter import save_html_report
from src.utils import PROJECT_ROOT, setup_logging, sanitize_run_name, ensure_dir


def _load_expected(expected_path: str) -> dict | None:
    p = Path(expected_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_benchmark(
    image_path: str,
    system_prompt_path: str,
    output_prompt_path: str,
    expected_path: str,
    run_name: str,
    prompt_path: str | None = None,
    progress_callback=None,
    model_progress_callback=None,
) -> dict:
    run_name = sanitize_run_name(run_name)
    logger = setup_logging(run_name)
    started_at = datetime.now().isoformat()
    logger.info(f"Benchmark started: {run_name}")

    if prompt_path:
        combined_prompt = Path(prompt_path).read_text(encoding="utf-8")
        combined_file = prompt_path
    else:
        combined_prompt = build_combined_prompt(system_prompt_path, output_prompt_path)
        combined_out = str(PROJECT_ROOT / "data" / "prompts" / "combined_eval_prompt.txt")
        save_combined_prompt(combined_prompt, combined_out)
        combined_file = combined_out

    save_combined_prompt_to_report(combined_prompt, run_name)

    logger.info("Loading image as base64")
    image_b64 = load_image_as_base64(image_path)

    expected = _load_expected(expected_path)
    if expected is None:
        logger.warning(f"Expected output not found or invalid: {expected_path}")

    models = get_enabled_models()
    model_results = []
    comparisons = []
    parsed_jsons = []

    for model_idx, model_cfg in enumerate(models):
        model_id = model_cfg["ollama_model"]
        model_name = model_cfg["name"]
        if progress_callback:
            progress_callback(model_idx, len(models), model_name)
        prompt_mode = model_cfg.get("prompt_mode", "full")
        model_prompt = build_compact_eval_prompt() if prompt_mode == "compact" else combined_prompt
        if prompt_mode == "compact":
            compact_prompt_path = (
                PROJECT_ROOT / "results" / "reports" / run_name / f"{model_name}_compact_eval_prompt.txt"
            )
            ensure_dir(compact_prompt_path.parent)
            compact_prompt_path.write_text(model_prompt, encoding="utf-8")
        logger.info(f"Calling model: {model_id} prompt_mode={prompt_mode}")

        result = generate_with_image(
            model_id,
            model_prompt,
            image_b64,
            options=model_cfg.get("options"),
            timeout_seconds=model_cfg.get("timeout_seconds"),
            max_retries=model_cfg.get("max_retries"),
            response_format=model_cfg.get("format"),
            chunk_callback=model_progress_callback,
        )
        result["model_name"] = model_name
        result["prompt_mode"] = prompt_mode
        result["operational_profile"] = model_cfg.get("operational_profile", "")
        result["deployment_recommendation"] = model_cfg.get("deployment_recommendation", "")
        model_results.append(result)

        save_raw_output(result.get("raw_text", ""), run_name, model_name)

        parse_result = parse_model_output(result.get("raw_text", ""))
        parsed_json = parse_result.get("parsed_json")
        save_parsed_output(parsed_json, run_name, model_name)
        parsed_jsons.append(parsed_json)

        comp = compare(expected, parse_result)
        save_comparison(comp, run_name, model_name)
        comparisons.append(comp)

        logger.info(
            f"{model_id} done - success={result['success']} "
            f"duration={result['duration_seconds']}s "
            f"timeout_seconds={result.get('timeout_seconds')} "
            f"retry_count={result.get('retry_count')} "
            f"json_parse={comp['json_parse_success']} "
            f"accuracy={comp['overall_accuracy_score']} "
            f"error={result.get('error')}"
        )

    ended_at = datetime.now().isoformat()

    summary = build_summary(
        run_name=run_name,
        image_path=image_path,
        system_prompt_file=system_prompt_path,
        output_prompt_file=output_prompt_path,
        combined_prompt_file=combined_file,
        expected_output_file=expected_path,
        started_at=started_at,
        ended_at=ended_at,
        model_results=model_results,
        comparisons=comparisons,
    )
    for model_summary, parsed_json in zip(summary.get("models", []), parsed_jsons):
        model_summary["parsed_output"] = parsed_json

    save_summary(summary, run_name)
    save_report(summary, run_name)
    html_path = save_html_report(summary, comparisons, run_name, expected, parsed_jsons)
    summary["html_report_path"] = str(html_path)

    logger.info(f"Benchmark complete. Winner: {summary['winner']['best_overall_model']}")
    logger.info(f"HTML report: {html_path}")
    return summary
