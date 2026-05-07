import os
from typing import Any

from src.config_loader import load_models_config


def build_model_artifact_manifest(model_config: dict[str, Any]) -> dict[str, Any]:
    storage = model_config.get("model_storage") or {}
    bucket_env = storage.get("bucket_env", "MODEL_ARTIFACT_BUCKET")
    region_env = storage.get("region_env", "AWS_REGION")
    bucket = os.getenv(bucket_env, "")
    key = storage.get("key", "")

    return {
        "model_name": model_config.get("name"),
        "provider": model_config.get("provider"),
        "runtime": storage.get("runtime", "ollama"),
        "source_local_model": storage.get("source_local_model") or model_config.get("ollama_model"),
        "storage_provider": storage.get("storage_provider", "s3"),
        "bucket_env": bucket_env,
        "bucket_configured": bool(bucket),
        "region_env": region_env,
        "region": os.getenv(region_env, ""),
        "s3_key": key,
        "s3_uri": f"s3://{bucket}/{key}" if bucket and key else "",
        "upload_plan": [
            "Export or package the local Ollama model artifact on a controlled build machine.",
            "Upload the artifact directory to the configured S3 key.",
            "Start the GPU inference runtime with permission to read the S3 artifact.",
            "Set the model inference endpoint environment variable for this API.",
        ],
    }


def list_model_artifact_manifests() -> list[dict[str, Any]]:
    return [
        build_model_artifact_manifest(model)
        for model in load_models_config().get("models", [])
        if model.get("enabled", True)
    ]
