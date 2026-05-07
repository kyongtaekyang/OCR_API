import io

from fastapi.testclient import TestClient

from src import api


client = TestClient(api.app)


def _fake_analysis(**kwargs):
    model_names = kwargs.get("model_names") or ["qwen2"]
    return {
        "run_name": kwargs.get("run_name") or "api_test",
        "inference_mode": kwargs.get("inference_mode") or "local",
        "input_type": "ocr_json" if kwargs.get("ocr_json_path") else "image",
        "models": [
            {
                "model_name": model_names[0],
                "provider": "ollama_local",
                "success": True,
                "duration_seconds": 0.1,
                "parse_success": True,
                "schema_compliance_score": 1.0,
                "analysis": {"ok": True},
                "validation": {"valid": True},
                "error": None,
            }
        ],
        "winner": {"best_api_candidate": "qwen2"},
        "artifacts": {},
    }


def test_ocr_json_upload_success(monkeypatch):
    monkeypatch.setattr(api, "run_writing_analysis", lambda **kwargs: _fake_analysis(**kwargs))
    response = client.post(
        "/analyze-writing",
        files={"ocr_json_file": ("ocr.json", io.BytesIO(b'{"handwritten_text":"x","source":{"type":"json","filename":"x","ocr_engine":"manual"}}'), "application/json")},
        data={"run_name": "api_test_001", "inference_mode": "remote"},
    )
    assert response.status_code == 200
    assert response.json()["input_type"] == "ocr_json"


def test_image_file_without_ocr_json_success(monkeypatch):
    monkeypatch.setattr(api, "run_writing_analysis", lambda **kwargs: _fake_analysis(**kwargs))
    response = client.post(
        "/analyze-writing",
        files={"image_file": ("sample.jpg", io.BytesIO(b"image-bytes"), "image/jpeg")},
        data={"run_name": "image_api_test_001", "inference_mode": "remote"},
    )
    assert response.status_code == 200
    assert response.json()["input_type"] == "image"


def test_missing_image_and_ocr_json_fails():
    response = client.post("/analyze-writing", data={"run_name": "bad"})
    assert response.status_code == 400


def test_invalid_extension_fails():
    response = client.post(
        "/analyze-writing",
        files={"ocr_json_file": ("ocr.txt", io.BytesIO(b"{}"), "text/plain")},
    )
    assert response.status_code == 400


def test_oversized_file_fails(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    response = client.post(
        "/analyze-writing",
        files={"ocr_json_file": ("ocr.json", io.BytesIO(b"{}"), "application/json")},
    )
    assert response.status_code == 413


def test_one_model_failure_still_returns(monkeypatch):
    def fake_run(**kwargs):
        result = _fake_analysis(**kwargs)
        result["models"].append({
            "model_name": "gemma",
            "provider": "aws_remote",
            "success": False,
            "duration_seconds": 0.2,
            "parse_success": False,
            "schema_compliance_score": 0.0,
            "analysis": None,
            "validation": {"valid": False},
            "error": "timeout",
        })
        return result

    monkeypatch.setattr(api, "run_writing_analysis", fake_run)
    response = client.post(
        "/analyze-writing",
        files={"ocr_json_file": ("ocr.json", io.BytesIO(b"{}"), "application/json")},
        data={"inference_mode": "remote"},
    )
    assert response.status_code == 200
    models = response.json()["models"]
    assert any(m["success"] for m in models)
    assert any(not m["success"] for m in models)


def test_qwen2_model_specific_endpoint_uses_local(monkeypatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return _fake_analysis(**kwargs)

    monkeypatch.setattr(api, "run_writing_analysis", fake_run)
    response = client.post(
        "/analyze-writing/qwen2",
        files={"ocr_json_file": ("ocr.json", io.BytesIO(b"{}"), "application/json")},
    )
    assert response.status_code == 200
    assert captured["model_names"] == ["qwen2"]
    assert captured["inference_mode"] == "local"


def test_gemma_model_specific_endpoint_uses_local(monkeypatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return _fake_analysis(**kwargs)

    monkeypatch.setattr(api, "run_writing_analysis", fake_run)
    response = client.post(
        "/analyze-writing/gemma",
        files={"ocr_json_file": ("ocr.json", io.BytesIO(b"{}"), "application/json")},
    )
    assert response.status_code == 200
    assert captured["model_names"] == ["gemma"]
    assert captured["inference_mode"] == "local"


def test_aws_model_artifacts_endpoint(monkeypatch):
    monkeypatch.setenv("MODEL_ARTIFACT_BUCKET", "model-bucket")
    response = client.get("/aws/model-artifacts")
    assert response.status_code == 200
    models = response.json()["models"]
    assert {m["model_name"] for m in models} == {"qwen2", "gemma"}
    assert all(m["s3_uri"].startswith("s3://model-bucket/") for m in models)


def test_test_page_loads():
    response = client.get("/test-page")
    assert response.status_code == 200
    assert "Writing API Model Test Page" in response.text
    assert "/analyze-writing/qwen2" in response.text
