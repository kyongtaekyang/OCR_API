import json
import time
import requests
from src.config_loader import load_benchmark_config


def _cfg() -> dict:
    return load_benchmark_config()


def check_server() -> bool:
    cfg = _cfg()
    url = cfg["ollama_base_url"] + cfg["tags_endpoint"]
    try:
        r = requests.get(url, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    cfg = _cfg()
    url = cfg["ollama_base_url"] + cfg["tags_endpoint"]
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def is_model_available(model_name: str) -> bool:
    models = list_models()
    return any(m == model_name or m.startswith(model_name.split(":")[0]) for m in models)


def generate_with_image(
    model: str,
    prompt: str,
    image_base64: str | None,
    options: dict | None = None,
    timeout_seconds: int | None = None,
    max_retries: int | None = None,
    response_format: str | dict | None = None,
    chunk_callback=None,
) -> dict:
    cfg = _cfg()
    url = cfg["ollama_base_url"] + cfg["generate_endpoint"]
    timeout = timeout_seconds if timeout_seconds is not None else cfg.get("timeout_seconds", 300)
    retries = max_retries if max_retries is not None else cfg.get("max_retries", 1)
    temperature = cfg.get("temperature", 0)
    max_chars = cfg.get("max_response_chars", 200000)
    merged_options = {"temperature": temperature, **(options or {})}
    expected_wall_seconds = timeout * (retries + 1)
    use_stream = chunk_callback is not None

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": use_stream,
        "options": merged_options,
    }
    if image_base64:
        payload["images"] = [image_base64]
    if response_format:
        payload["format"] = response_format
    keep_alive = merged_options.pop("keep_alive", None)
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    retry_count = 0
    last_error = None
    timed_out = False
    start = time.time()

    for attempt in range(retries + 1):
        if attempt > 0:
            retry_count += 1
        try:
            if use_stream:
                raw_text = ""
                with requests.post(url, json=payload, timeout=timeout, stream=True) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except Exception:
                            continue
                        raw_text += chunk.get("response", "")
                        if chunk_callback:
                            chunk_callback(len(raw_text))
                        if chunk.get("done", False):
                            break
            else:
                r = requests.post(url, json=payload, timeout=timeout)
                r.raise_for_status()
                raw_text = r.json().get("response", "")

            if len(raw_text) > max_chars:
                raw_text = raw_text[:max_chars]
            duration = time.time() - start
            wall_timeout_exceeded = duration > expected_wall_seconds + 5
            return {
                "model": model,
                "success": True,
                "raw_text": raw_text,
                "duration_seconds": round(duration, 3),
                "retry_count": retry_count,
                "timeout": False,
                "error": None,
                "response_size_chars": len(raw_text),
                "timeout_seconds": timeout,
                "max_retries": retries,
                "options": merged_options,
                "response_format": response_format,
                "expected_wall_seconds": expected_wall_seconds,
                "wall_timeout_exceeded": wall_timeout_exceeded,
                "failure_category": "slow_success" if wall_timeout_exceeded else None,
            }
        except requests.exceptions.Timeout:
            timed_out = True
            last_error = "timeout"
        except requests.exceptions.HTTPError as e:
            response = getattr(e, "response", None)
            body = getattr(response, "text", "") if response is not None else ""
            last_error = f"{e.__class__.__name__}: {body or str(e)}"
        except Exception as e:
            last_error = f"{e.__class__.__name__}: {e}"

    duration = time.time() - start
    wall_timeout_exceeded = duration > expected_wall_seconds + 5
    return {
        "model": model,
        "success": False,
        "raw_text": "",
        "duration_seconds": round(duration, 3),
        "retry_count": retry_count,
        "timeout": timed_out,
        "error": last_error,
        "response_size_chars": 0,
        "timeout_seconds": timeout,
        "max_retries": retries,
        "options": merged_options,
        "response_format": response_format,
        "expected_wall_seconds": expected_wall_seconds,
        "wall_timeout_exceeded": wall_timeout_exceeded,
        "failure_category": "timeout" if timed_out else "request_error",
    }
