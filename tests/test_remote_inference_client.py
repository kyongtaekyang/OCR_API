import json

import httpx
import pytest

from src.remote_inference_client import RemoteInferenceClient, mask_secret


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.test/invoke")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("error", request=request, response=response)


class FakeClient:
    responses = []
    calls = []

    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(monkeypatch, responses, max_retries=1):
    FakeClient.responses = list(responses)
    FakeClient.calls = []
    monkeypatch.setattr("src.remote_inference_client.httpx.Client", FakeClient)
    monkeypatch.setenv("QWEN2_INFERENCE_ENDPOINT", "https://example.test/invoke")
    monkeypatch.setenv("QWEN2_API_KEY", "secret-token")
    monkeypatch.setenv("MODEL_ARTIFACT_BUCKET", "model-bucket")
    monkeypatch.setenv("AWS_REGION", "ap-northeast-2")
    return RemoteInferenceClient({
        "name": "qwen2",
        "provider": "aws_remote",
        "ollama_model": "qwen2.5vl:latest",
        "endpoint_env": "QWEN2_INFERENCE_ENDPOINT",
        "api_key_env": "QWEN2_API_KEY",
        "model_storage": {
            "storage_provider": "s3",
            "bucket_env": "MODEL_ARTIFACT_BUCKET",
            "key": "ollama/qwen2.5vl/latest/",
            "region_env": "AWS_REGION",
            "source_local_model": "qwen2.5vl:latest",
            "runtime": "ollama",
        },
        "timeout_seconds": 30,
        "max_retries": max_retries,
    })


def test_response_field_success(monkeypatch):
    client = _client(monkeypatch, [FakeResponse({"response": '{"ok": true}'})])
    result = client.generate(client.model_config, "prompt", {"x": 1})
    assert result["success"] is True
    assert result["raw_text"] == '{"ok": true}'


def test_analysis_field_success(monkeypatch):
    client = _client(monkeypatch, [FakeResponse({"analysis": {"ok": True}})])
    result = client.generate(client.model_config, "prompt", {})
    assert result["raw_text"] == '{"ok": true}'


def test_raw_text_field_success(monkeypatch):
    client = _client(monkeypatch, [FakeResponse({"raw_text": '{"ok": true}'})])
    result = client.generate(client.model_config, "prompt", {})
    assert result["raw_text"] == '{"ok": true}'


def test_timeout_handling(monkeypatch):
    client = _client(monkeypatch, [httpx.TimeoutException("timeout")], max_retries=0)
    result = client.generate(client.model_config, "prompt", {})
    assert result["success"] is False
    assert result["timeout"] is True
    assert result["error"] == "timeout"


def test_500_error_handling(monkeypatch):
    client = _client(monkeypatch, [FakeResponse({"error": "server"}, status_code=500)], max_retries=0)
    result = client.generate(client.model_config, "prompt", {})
    assert result["success"] is False
    assert result["failure_category"] == "http_5xx"
    assert "5xx" in result["error"]


def test_retry_count(monkeypatch):
    client = _client(
        monkeypatch,
        [httpx.ConnectError("boom"), FakeResponse({"response": "{}"})],
        max_retries=1,
    )
    result = client.generate(client.model_config, "prompt", {})
    assert result["success"] is True
    assert result["retry_count"] == 1


def test_api_key_masking():
    assert mask_secret("super-secret-token") == "supe...****"


def test_payload_includes_s3_model_artifact(monkeypatch):
    client = _client(monkeypatch, [FakeResponse({"response": "{}"})])
    result = client.generate(client.model_config, "prompt", {"x": 1})
    assert result["success"] is True
    payload = FakeClient.calls[0]["json"]
    assert payload["source_local_model"] == "qwen2.5vl:latest"
    assert payload["model_storage"]["s3_uri"] == "s3://model-bucket/ollama/qwen2.5vl/latest/"
    assert payload["model_storage"]["runtime"] == "ollama"
