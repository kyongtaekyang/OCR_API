import json
from pathlib import Path
from dotenv import load_dotenv
from src.utils import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def load_models_config() -> dict:
    path = PROJECT_ROOT / "config" / "models.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Models config not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in models config ({path}): {e}")


def load_benchmark_config() -> dict:
    path = PROJECT_ROOT / "config" / "benchmark_config.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmark config not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in benchmark config ({path}): {e}")


def get_enabled_models() -> list[dict]:
    cfg = load_models_config()
    return [m for m in cfg.get("models", []) if m.get("enabled", True)]


def get_models_by_names(names: list[str] | None = None) -> list[dict]:
    models = get_enabled_models()
    if not names:
        return models
    wanted = {name.strip() for name in names if name.strip()}
    return [m for m in models if m.get("name") in wanted or m.get("ollama_model") in wanted]
