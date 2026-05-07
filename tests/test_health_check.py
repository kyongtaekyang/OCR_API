"""Mock-based health check tests."""
import pytest
from unittest.mock import patch
from src import health_check


def test_health_check_structure():
    with patch("src.health_check.check_server", return_value=False), \
         patch("src.health_check.list_models", return_value=[]):
        result = health_check.run_health_check()

    assert "overall" in result
    assert "passed" in result
    assert "total" in result
    assert "checks" in result
    assert isinstance(result["checks"], list)


def test_health_check_check_names():
    with patch("src.health_check.check_server", return_value=False), \
         patch("src.health_check.list_models", return_value=[]):
        result = health_check.run_health_check()

    names = {c["check"] for c in result["checks"]}
    assert "python_version" in names
    assert "project_root_exists" in names
    assert "ollama_server_reachable" in names


def test_health_check_ollama_pass():
    with patch("src.health_check.check_server", return_value=True), \
         patch("src.health_check.list_models", return_value=["qwen2.5vl:latest", "gemma4:latest"]):
        result = health_check.run_health_check()

    checks = {c["check"]: c["status"] for c in result["checks"]}
    assert checks.get("ollama_server_reachable") == "PASS"
    assert checks.get("model_qwen2_local") == "PASS"
    assert checks.get("model_gemma_local") == "PASS"


def test_health_check_ollama_fail():
    with patch("src.health_check.check_server", return_value=False), \
         patch("src.health_check.list_models", return_value=[]):
        result = health_check.run_health_check()

    checks = {c["check"]: c["status"] for c in result["checks"]}
    assert checks.get("ollama_server_reachable") == "FAIL"


def test_health_check_saves_log(tmp_path):
    with patch("src.health_check.check_server", return_value=False), \
         patch("src.health_check.list_models", return_value=[]), \
         patch("src.health_check.PROJECT_ROOT", tmp_path):
        result = health_check.run_health_check()

    log_dir = tmp_path / "results" / "logs"
    logs = list(log_dir.glob("health_check_*.json"))
    assert len(logs) == 1
