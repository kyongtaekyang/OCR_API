import json
import os
import time
from typing import Any

import httpx


def mask_secret(value: str | None, visible: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= visible:
        return "*" * len(value)
    return value[:visible] + "..." + "*" * 4


def mask_endpoint(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = httpx.URL(value)
        return f"{parsed.scheme}://{parsed.host}/..."
    except Exception:
        return mask_secret(value)


class RemoteInferenceClient:
    def __init__(self, model_config: dict[str, Any]):
        self.model_config = model_config
        self.model_name = model_config["name"]
        self.provider = model_config.get("provider", "aws_remote")
        self.endpoint_env = model_config.get("endpoint_env", "")
        self.api_key_env = model_config.get("api_key_env", "")
        self.endpoint = os.getenv(self.endpoint_env, "")
        self.api_key = os.getenv(self.api_key_env, "")
        self.timeout_seconds = int(
            model_config.get("timeout_seconds")
            or os.getenv("REQUEST_TIMEOUT_SECONDS", "300")
        )
        self.max_retries = int(
            model_config.get("max_retries")
            if model_config.get("max_retries") is not None
            else os.getenv("MAX_RETRIES", "1")
        )

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def endpoint_configured(self) -> bool:
        return bool(self.endpoint)

    def check_endpoint(self) -> dict[str, Any]:
        if not self.endpoint:
            return {
                "configured": False,
                "reachable": False,
                "endpoint": "",
                "error": f"missing environment variable: {self.endpoint_env}",
            }
        try:
            with httpx.Client(timeout=min(self.timeout_seconds, 10)) as client:
                response = client.get(self.endpoint, headers=self._auth_headers())
            return {
                "configured": True,
                "reachable": response.status_code < 500,
                "endpoint": mask_endpoint(self.endpoint),
                "status_code": response.status_code,
                "error": None if response.status_code < 500 else response.text[:500],
            }
        except Exception as exc:
            return {
                "configured": True,
                "reachable": False,
                "endpoint": mask_endpoint(self.endpoint),
                "error": f"{exc.__class__.__name__}: {exc}",
            }

    def generate(
        self,
        model_config: dict[str, Any],
        prompt: str,
        ocr_json: dict,
        image_base64: str | None = None,
    ) -> dict[str, Any]:
        start = time.time()
        if not self.endpoint:
            return self._failure(start, 0, False, f"missing endpoint env: {self.endpoint_env}", "configuration_error")

        payload = self._build_payload(model_config, prompt, ocr_json, image_base64)
        raw_text = ""
        retry_count = 0
        timed_out = False
        last_error = None
        status_code = None
        failure_category = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                retry_count += 1
            try:
                data, status_code = self._post_with_retry(payload)
                raw_text = self._extract_raw_text(data)
                duration = round(time.time() - start, 3)
                return {
                    "model": self.model_name,
                    "provider": self.provider,
                    "success": True,
                    "raw_text": raw_text,
                    "duration_seconds": duration,
                    "retry_count": retry_count,
                    "timeout": False,
                    "error": None,
                    "response_size_chars": len(raw_text),
                    "timeout_seconds": self.timeout_seconds,
                    "max_retries": self.max_retries,
                    "status_code": status_code,
                    "failure_category": None,
                }
            except httpx.TimeoutException:
                timed_out = True
                last_error = "timeout"
                failure_category = "timeout"
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                status_group = "4xx" if 400 <= status_code < 500 else "5xx"
                last_error = f"{status_group} status_code={status_code}"
                failure_category = f"http_{status_group}"
                if 400 <= status_code < 500:
                    break
            except json.JSONDecodeError as exc:
                last_error = f"JSONDecodeError: {exc}"
                failure_category = "json_decode_error"
                break
            except httpx.HTTPError as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
                failure_category = "network_error"
            except Exception as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
                failure_category = "request_error"

        return self._failure(start, retry_count, timed_out, last_error, failure_category, status_code)

    def _build_payload(
        self,
        model_config: dict[str, Any],
        prompt: str,
        ocr_json: dict,
        image_base64: str | None = None,
    ) -> dict[str, Any]:
        input_payload = {"prompt": prompt, "ocr_json": ocr_json}
        if image_base64:
            input_payload["image_base64"] = image_base64
        model_storage = self._build_model_storage(model_config)
        return {
            "model": model_config.get("name", self.model_name),
            "source_local_model": model_config.get("ollama_model"),
            "model_storage": model_storage,
            "input": input_payload,
            "options": {
                "temperature": 0,
                "response_format": "json",
            },
        }

    def _build_model_storage(self, model_config: dict[str, Any]) -> dict[str, Any]:
        storage = model_config.get("model_storage") or {}
        bucket = os.getenv(storage.get("bucket_env", ""), "")
        key = storage.get("key", "")
        return {
            "storage_provider": storage.get("storage_provider", "s3"),
            "s3_uri": f"s3://{bucket}/{key}" if bucket and key else "",
            "s3_key": key,
            "region": os.getenv(storage.get("region_env", "AWS_REGION"), ""),
            "runtime": storage.get("runtime", "ollama"),
            "source_local_model": storage.get("source_local_model") or model_config.get("ollama_model"),
        }

    def _post_with_retry(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.endpoint, headers=self._auth_headers(), json=payload)
            response.raise_for_status()
            return response.json(), response.status_code

    def _extract_raw_text(self, data: dict[str, Any]) -> str:
        if "response" in data:
            value = data["response"]
        elif "analysis" in data:
            value = data["analysis"]
        elif "raw_text" in data:
            value = data["raw_text"]
        else:
            value = data
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _failure(
        self,
        start: float,
        retry_count: int,
        timed_out: bool,
        error: str | None,
        failure_category: str | None,
        status_code: int | None = None,
    ) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "provider": self.provider,
            "success": False,
            "raw_text": "",
            "duration_seconds": round(time.time() - start, 3),
            "retry_count": retry_count,
            "timeout": timed_out,
            "error": error,
            "response_size_chars": 0,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "status_code": status_code,
            "failure_category": failure_category,
        }
