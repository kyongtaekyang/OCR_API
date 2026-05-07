import json
from pathlib import Path
from datetime import datetime
from src.utils import PROJECT_ROOT, ensure_dir, safe_path, sanitize_run_name


def _run_dir(subdir: str, run_name: str) -> Path:
    base = PROJECT_ROOT / "results" / subdir
    return ensure_dir(safe_path(base, sanitize_run_name(run_name)))


def save_raw_output(raw_text: str, run_name: str, model_name: str) -> Path:
    path = _run_dir("raw_outputs", run_name) / f"{model_name}_raw.txt"
    path.write_text(raw_text or "", encoding="utf-8")
    return path


def save_parsed_output(parsed_json: dict | None, run_name: str, model_name: str) -> Path:
    path = _run_dir("parsed_outputs", run_name) / f"{model_name}_parsed.json"
    path.write_text(
        json.dumps(parsed_json, ensure_ascii=False, indent=2) if parsed_json else "null",
        encoding="utf-8",
    )
    return path


def save_comparison(comparison: dict, run_name: str, model_name: str) -> Path:
    path = _run_dir("comparisons", run_name) / f"{model_name}_comparison.json"
    path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_ocr_result(ocr_json: dict, run_name: str) -> Path:
    path = _run_dir("ocr", run_name) / "ocr_result.json"
    path.write_text(json.dumps(ocr_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_validation(validation: dict, run_name: str, model_name: str) -> Path:
    path = _run_dir("validations", run_name) / f"{model_name}_validation.json"
    path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_combined_prompt_to_report(prompt: str, run_name: str) -> Path:
    path = _run_dir("reports", run_name) / "combined_eval_prompt.txt"
    path.write_text(prompt, encoding="utf-8")
    return path


def _api_readiness(model_result: dict, comparison: dict) -> dict:
    success = model_result.get("success", False)
    duration = model_result.get("duration_seconds", 9999)
    timeout = model_result.get("timeout", False)
    wall_timeout_exceeded = model_result.get("wall_timeout_exceeded", False)
    parse_ok = comparison.get("json_parse_success", False)
    schema_score = comparison.get("schema_compliance_score", 0.0) or 0.0
    key_completeness = comparison.get("required_key_completeness", 0.0) or 0.0
    score_math = comparison.get("score_math_valid", False)
    raw_text = model_result.get("raw_text", "")

    json_validity = 1.0 if parse_ok else 0.0
    response_time = max(0.0, 1.0 - duration / 300.0) if success and not timeout else 0.0
    consistency = key_completeness
    error_handling = 0.0 if (timeout or not success) else (0.5 if wall_timeout_exceeded else 1.0)
    output_schema = schema_score
    score_math_score = 1.0 if score_math else 0.5

    strict_penalty = 0.0
    raw = raw_text.strip() if raw_text else ""
    if raw and not (raw.startswith("{") and raw.endswith("}")):
        strict_penalty = 0.1

    total = (
        json_validity * 0.25
        + response_time * 0.20
        + consistency * 0.15
        + error_handling * 0.15
        + output_schema * 0.15
        + score_math_score * 0.10
        - strict_penalty
    )
    total = max(0.0, min(1.0, round(total, 4)))

    return {
        "json_validity_score": round(json_validity, 4),
        "response_time_score": round(response_time, 4),
        "consistency_score": round(consistency, 4),
        "error_handling_score": round(error_handling, 4),
        "output_schema_compliance_score": round(output_schema, 4),
        "score_math_validity_score": round(score_math_score, 4),
        "total_api_readiness_score": total,
    }


def _aws_suitability(model_name: str, api_readiness: dict, model_result: dict) -> dict:
    duration = model_result.get("duration_seconds", 9999)
    total_api = api_readiness.get("total_api_readiness_score", 0.0)

    gpu_comment = (
        "Large vision-language model requires GPU inference. "
        "Recommend GPU-enabled instance (g4dn/g5/p3) or EC2 with NVIDIA driver."
    )
    container_ready = (
        "Docker containerization is feasible. "
        "Mount Ollama model cache as volume; handle cold-start latency."
    )
    lambda_unsuitable = (
        "Lambda is unsuitable for model inference due to memory limits, "
        "execution timeout (15 min max), and lack of GPU support. "
        "Lambda can serve as orchestration layer only."
    )
    ecs_suitability = (
        "ECS on EC2 GPU instances or EC2 GPU directly is recommended for model serving. "
        "EKS with GPU node groups is suitable for larger deployments."
    )
    s3_suitability = (
        "S3 is well-suited for storing input images, prompt files, and benchmark results. "
        "Use S3 URI references rather than inline base64 for production workloads."
    )
    cw_suitability = (
        "CloudWatch Logs for run logs; S3 for benchmark result JSONs and Markdown reports. "
        "CloudWatch Metrics for response time and error rate monitoring."
    )

    if duration > 120:
        complexity = "High - response time exceeds 2 minutes; requires async processing and queue-based architecture."
    elif duration > 60:
        complexity = "Medium-High - response time over 1 minute; consider async API pattern."
    else:
        complexity = "Medium - response time acceptable for synchronous API with adequate timeout settings."

    if total_api >= 0.8 and duration < 120:
        recommendation = (
            f"{model_name} shows strong API readiness. "
            "Recommended for ECS GPU deployment with S3 I/O and CloudWatch monitoring."
        )
    elif total_api >= 0.6:
        recommendation = (
            f"{model_name} is a viable candidate with moderate API readiness. "
            "Improve JSON consistency before production deployment."
        )
    else:
        recommendation = (
            f"{model_name} requires significant improvement in JSON output stability "
            "before AWS production deployment."
        )

    return {
        "expected_gpu_requirement_comment": gpu_comment,
        "containerization_readiness": container_ready,
        "api_gateway_lambda_unsuitable_reason": lambda_unsuitable,
        "ecs_or_ec2_suitability": ecs_suitability,
        "s3_input_output_suitability": s3_suitability,
        "cloudwatch_logging_suitability": cw_suitability,
        "estimated_operational_complexity": complexity,
        "recommendation": recommendation,
    }


def _select_winner(models_summary: list[dict]) -> dict:
    successful = [m for m in models_summary if m.get("success")]

    def best_by(key, reverse=True):
        candidates = [m for m in successful if m.get(key) is not None]
        if not candidates:
            return "N/A"
        return max(candidates, key=lambda x: x[key])["model_name"] if reverse else \
               min(candidates, key=lambda x: x[key])["model_name"]

    best_accuracy = best_by("overall_accuracy_score", reverse=True)
    fastest = best_by("duration_seconds", reverse=False)
    best_api = best_by("api_readiness_score", reverse=True)

    def aws_score(m):
        return m.get("api_readiness_score", 0) * 0.5 + (
            1 - min(m.get("duration_seconds", 9999), 300) / 300
        ) * 0.3 + (1.0 if m.get("success") else 0.0) * 0.2

    best_aws = max(successful, key=aws_score)["model_name"] if successful else "N/A"

    def overall_score(m):
        return (
            (m.get("overall_accuracy_score") or 0) * 0.50
            + (1.0 if m.get("success") else 0.0) * 0.20
            + (1 - min(m.get("duration_seconds", 9999), 300) / 300) * 0.15
            + (m.get("api_readiness_score") or 0) * 0.10
            + (aws_score(m)) * 0.05
        )

    best_overall = max(successful, key=overall_score)["model_name"] if successful else "N/A"

    return {
        "best_accuracy_model": best_accuracy,
        "fastest_model": fastest,
        "best_api_candidate": best_api,
        "best_aws_candidate": best_aws,
        "best_overall_model": best_overall,
    }


def build_summary(
    run_name: str,
    image_path: str,
    system_prompt_file: str,
    output_prompt_file: str,
    combined_prompt_file: str,
    expected_output_file: str,
    started_at: str,
    ended_at: str,
    model_results: list[dict],
    comparisons: list[dict],
) -> dict:
    models_summary = []
    for mr, comp in zip(model_results, comparisons):
        api_r = _api_readiness(mr, comp)
        aws_s = _aws_suitability(mr["model"], api_r, mr)
        entry = {
            "model_name": mr["model"].split(":")[0],
            "ollama_model": mr["model"],
            "success": mr.get("success", False),
            "duration_seconds": mr.get("duration_seconds"),
            "retry_count": mr.get("retry_count", 0),
            "timeout": mr.get("timeout", False),
            "error": mr.get("error"),
            "response_size_chars": mr.get("response_size_chars", 0),
            "timeout_seconds": mr.get("timeout_seconds"),
            "max_retries": mr.get("max_retries"),
            "expected_wall_seconds": mr.get("expected_wall_seconds"),
            "wall_timeout_exceeded": mr.get("wall_timeout_exceeded", False),
            "failure_category": mr.get("failure_category"),
            "request_options": mr.get("options", {}),
            "response_format": mr.get("response_format"),
            "prompt_mode": mr.get("prompt_mode", "full"),
            "operational_profile": mr.get("operational_profile", ""),
            "deployment_recommendation": mr.get("deployment_recommendation", ""),
            "json_parse_success": comp.get("json_parse_success", False),
            "schema_type": comp.get("schema_type", "unknown"),
            "schema_compliance_score": comp.get("schema_compliance_score", 0),
            "required_key_completeness": comp.get("required_key_completeness", 0),
            "score_math_valid": comp.get("score_math_valid", False),
            "score_anomalies": comp.get("score_anomalies", []),
            "original_writing_similarity": comp.get("original_writing_similarity", 0),
            "corrected_writing_similarity": comp.get("corrected_writing_similarity", 0),
            "error_tag_f1": comp.get("error_tag_f1", 0),
            "total_score_difference": comp.get("total_score_difference"),
            "overall_accuracy_score": comp.get("overall_accuracy_score", 0),
            "api_readiness_score": api_r["total_api_readiness_score"],
            "api_readiness_detail": api_r,
            "aws_suitability": aws_s,
            "aws_suitability_summary": aws_s["recommendation"],
            "detail_diff": comp.get("detail_diff"),
        }
        models_summary.append(entry)

    winner = _select_winner(models_summary)

    return {
        "run_name": run_name,
        "input_image": image_path,
        "system_prompt_file": system_prompt_file,
        "output_prompt_file": output_prompt_file,
        "combined_prompt_file": combined_prompt_file,
        "expected_output_file": expected_output_file,
        "started_at": started_at,
        "ended_at": ended_at,
        "models": models_summary,
        "winner": winner,
    }


def save_summary(summary: dict, run_name: str) -> Path:
    path = _run_dir("reports", run_name) / "benchmark_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _fmt_score(val) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def generate_markdown_report(summary: dict) -> str:
    lines = [
        f"# Benchmark Report: {summary.get('run_name', '')}",
        "",
        f"**Started:** {summary.get('started_at', '')}  ",
        f"**Ended:** {summary.get('ended_at', '')}",
        "",
        f"**Image:** `{summary.get('input_image', '')}`  ",
        f"**Expected Output:** `{summary.get('expected_output_file', '')}`",
        "",
        "---",
        "",
        "## Model Results",
        "",
    ]

    for m in summary.get("models", []):
        lines += [
            f"### {m['ollama_model']}",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Success | {m['success']} |",
            f"| Duration (s) | {_fmt_score(m.get('duration_seconds'))} |",
            f"| Runtime Timeout (s) | {_fmt_score(m.get('timeout_seconds'))} |",
            f"| Expected Wall Seconds | {_fmt_score(m.get('expected_wall_seconds'))} |",
            f"| Wall Timeout Exceeded | {m.get('wall_timeout_exceeded', False)} |",
            f"| Failure Category | {m.get('failure_category') or ''} |",
            f"| Retry Count | {m.get('retry_count', 0)} |",
            f"| Max Retries | {m.get('max_retries', 0)} |",
            f"| Timeout | {m.get('timeout', False)} |",
            f"| Error | {m.get('error') or ''} |",
            f"| Response Size Chars | {m.get('response_size_chars', 0)} |",
            f"| Prompt Mode | {m.get('prompt_mode', 'full')} |",
            f"| Operational Profile | {m.get('operational_profile', '')} |",
            f"| JSON Parse Success | {m.get('json_parse_success')} |",
            f"| Schema Type | {m.get('schema_type')} |",
            f"| Schema Compliance | {_fmt_score(m.get('schema_compliance_score'))} |",
            f"| Required Key Completeness | {_fmt_score(m.get('required_key_completeness'))} |",
            f"| Score Math Valid | {m.get('score_math_valid')} |",
            f"| Original Writing Similarity | {_fmt_score(m.get('original_writing_similarity'))} |",
            f"| Corrected Writing Similarity | {_fmt_score(m.get('corrected_writing_similarity'))} |",
            f"| Error Tag F1 | {_fmt_score(m.get('error_tag_f1'))} |",
            f"| Total Score Difference | {_fmt_score(m.get('total_score_difference'))} |",
            f"| Overall Accuracy Score | {_fmt_score(m.get('overall_accuracy_score'))} |",
            f"| API Readiness Score | {_fmt_score(m.get('api_readiness_score'))} |",
            "",
        ]

        anomalies = m.get("score_anomalies", [])
        if anomalies:
            lines.append("**Score Anomalies:**")
            for a in anomalies:
                lines.append(f"- `{a.get('path')}`: {a.get('issue')} (score={a.get('score')}, max={a.get('max_score')})")
            lines.append("")

        aws = m.get("aws_suitability", {})
        lines += [
            "**AWS Suitability:**",
            f"- {aws.get('recommendation', '')}",
            f"- Complexity: {aws.get('estimated_operational_complexity', '')}",
            f"- Deployment note: {m.get('deployment_recommendation', '')}",
            "",
        ]

    winner = summary.get("winner", {})
    lines += [
        "---",
        "",
        "## Winner Summary",
        "",
        f"| Category | Winner |",
        f"|---|---|",
        f"| Best Accuracy | {winner.get('best_accuracy_model')} |",
        f"| Fastest | {winner.get('fastest_model')} |",
        f"| Best API Candidate | {winner.get('best_api_candidate')} |",
        f"| Best AWS Candidate | {winner.get('best_aws_candidate')} |",
        f"| Best Overall | {winner.get('best_overall_model')} |",
        "",
    ]

    return "\n".join(lines)


def save_report(summary: dict, run_name: str) -> Path:
    md = generate_markdown_report(summary)
    path = _run_dir("reports", run_name) / "benchmark_report.md"
    path.write_text(md, encoding="utf-8")
    return path
