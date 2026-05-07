"""Mock tests for ollama_client — no real Ollama connection required."""
import json
import pytest
from unittest.mock import patch, MagicMock
from src import ollama_client


def _mock_response(json_data: dict, status: int = 200):
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = json_data
    mock.text = json.dumps(json_data)
    mock.raise_for_status = MagicMock()
    return mock


def test_check_server_success():
    with patch("src.ollama_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response({}, 200)
        assert ollama_client.check_server() is True


def test_check_server_failure():
    with patch("src.ollama_client.requests.get") as mock_get:
        mock_get.side_effect = Exception("connection refused")
        assert ollama_client.check_server() is False


def test_list_models():
    with patch("src.ollama_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response({
            "models": [{"name": "qwen2.5vl:latest"}, {"name": "gemma4:latest"}]
        })
        models = ollama_client.list_models()
        assert "qwen2.5vl:latest" in models
        assert "gemma4:latest" in models


def test_is_model_available_true():
    with patch("src.ollama_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response({
            "models": [{"name": "qwen2.5vl:latest"}]
        })
        assert ollama_client.is_model_available("qwen2.5vl:latest") is True


def test_is_model_available_false():
    with patch("src.ollama_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response({"models": []})
        assert ollama_client.is_model_available("nonexistent:latest") is False


def test_generate_with_image_success():
    with patch("src.ollama_client.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"response": '{"key": "value"}'})
        result = ollama_client.generate_with_image(
            "qwen2.5vl:latest",
            "prompt text",
            "base64imagedata",
            options={"num_predict": 1024},
            timeout_seconds=900,
            max_retries=0,
            response_format="json",
        )
        assert result["success"] is True
        assert result["raw_text"] == '{"key": "value"}'
        assert result["timeout"] is False
        assert result["error"] is None
        assert result["response_size_chars"] > 0
        assert result["timeout_seconds"] == 900
        assert result["max_retries"] == 0
        assert result["options"]["num_predict"] == 1024
        assert result["response_format"] == "json"
        assert result["expected_wall_seconds"] == 900
        assert result["wall_timeout_exceeded"] is False
        assert mock_post.call_args.kwargs["json"]["format"] == "json"


def test_generate_with_image_timeout():
    import requests as req
    with patch("src.ollama_client.requests.post") as mock_post:
        mock_post.side_effect = req.exceptions.Timeout()
        result = ollama_client.generate_with_image(
            "qwen2.5vl:latest", "prompt", "b64"
        )
        assert result["success"] is False
        assert result["timeout"] is True
        assert result["error"] == "timeout"
        assert result["failure_category"] == "timeout"


def test_generate_with_image_general_error():
    with patch("src.ollama_client.requests.post") as mock_post:
        mock_post.side_effect = Exception("connection error")
        result = ollama_client.generate_with_image(
            "gemma4:latest", "prompt", "b64"
        )
        assert result["success"] is False
        assert "connection error" in result["error"]


def test_generate_uses_model_specific_runtime_policy():
    with patch("src.ollama_client.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"response": "{}"})
        result = ollama_client.generate_with_image(
            "qwen2.5vl:latest",
            "prompt",
            "b64",
            options={"num_predict": 2048},
            timeout_seconds=900,
            max_retries=0,
        )

    assert result["success"] is True
    assert result["timeout_seconds"] == 900
    assert result["max_retries"] == 0
    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["options"]["num_predict"] == 2048


def test_generate_truncates_long_response():
    long_text = "x" * 300000
    with patch("src.ollama_client.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"response": long_text})
        result = ollama_client.generate_with_image("model:latest", "p", "b64")
        assert result["response_size_chars"] <= 200000
