import sys
import json
import os
from datetime import datetime
from pathlib import Path
from src.utils import PROJECT_ROOT, ensure_dir
from src.ollama_client import check_server, list_models
from src.config_loader import get_enabled_models, load_benchmark_config


def run_health_check() -> dict:
    checks = []
    passed = 0

    def add(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        status = "PASS" if ok else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})
        if ok:
            passed += 1

    add("python_version", sys.version_info >= (3, 10),
        f"Python {sys.version.split()[0]}")

    add("project_root_exists", PROJECT_ROOT.exists(), str(PROJECT_ROOT))

    required_dirs = [
        "config", "data/input_images", "data/prompts",
        "data/expected_outputs", "results/raw_outputs",
        "results/parsed_outputs", "results/comparisons",
        "results/reports", "results/logs", "results/ocr",
        "results/validations",
    ]
    for d in required_dirs:
        add(f"dir_{d.replace('/', '_')}", (PROJECT_ROOT / d).exists(), d)

    for cfg_file in ["config/models.json", "config/benchmark_config.json"]:
        add(f"config_{cfg_file.split('/')[1]}", (PROJECT_ROOT / cfg_file).exists(), cfg_file)

    for pf in ["data/prompts/system_prompt.txt", "data/prompts/output_prompt.txt"]:
        add(f"prompt_{Path(pf).stem}", (PROJECT_ROOT / pf).exists(), pf)

    results_dir = PROJECT_ROOT / "results"
    try:
        test_file = results_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        add("results_write_permission", True, str(results_dir))
    except Exception as e:
        add("results_write_permission", False, str(e))

    cfg = load_benchmark_config()
    ollama_url = cfg.get("ollama_base_url", "http://localhost:11434")
    tags_endpoint = cfg.get("tags_endpoint", "/api/tags")

    ollama_ok = check_server()
    add("ollama_server_reachable", ollama_ok, ollama_url)
    add("ollama_tags_endpoint", ollama_ok, ollama_url + tags_endpoint)

    available_models: list[str] = []
    if ollama_ok:
        available_models = list_models()

    for model_cfg in get_enabled_models():
        ollama_model = model_cfg.get("ollama_model")
        endpoint_env = model_cfg.get("endpoint_env")
        provider = model_cfg.get("provider", "ollama_local")
        if ollama_model:
            found = any(
                m == ollama_model or m.startswith(ollama_model.split(":")[0])
                for m in available_models
            )
            add(f"model_{model_cfg['name']}_local", found, ollama_model)
        if provider == "aws_remote" and endpoint_env:
            configured = bool(os.getenv(endpoint_env))
            add(f"model_{model_cfg['name']}_remote_endpoint_env", configured, endpoint_env)
        storage = model_cfg.get("model_storage") or {}
        bucket_env = storage.get("bucket_env")
        if provider == "aws_remote" and bucket_env:
            add(
                f"model_{model_cfg['name']}_artifact_bucket_env",
                bool(os.getenv(bucket_env)),
                bucket_env,
            )

    add("sample_input_dir_exists",
        (PROJECT_ROOT / "data" / "sample").exists(),
        "data/sample")

    total = len(checks)
    result = {
        "timestamp": datetime.now().isoformat(),
        "overall": "PASS" if passed == total else ("PARTIAL" if passed > 0 else "FAIL"),
        "passed": passed,
        "total": total,
        "checks": checks,
    }

    log_dir = ensure_dir(PROJECT_ROOT / "results" / "logs")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"health_check_{ts}.json"
    log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def print_health_check(result: dict) -> None:
    print(f"\n=== Health Check [{result['overall']}] ===")
    print(f"Passed: {result['passed']}/{result['total']}")
    print()
    for c in result["checks"]:
        icon = "PASS" if c["status"] == "PASS" else "FAIL"
        detail = f"  ({c['detail']})" if c.get("detail") else ""
        print(f"  {icon} {c['check']}{detail}")
    print()
