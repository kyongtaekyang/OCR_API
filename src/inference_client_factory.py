from typing import Any

from src.ollama_client import generate_with_image
from src.remote_inference_client import RemoteInferenceClient


class LocalOllamaInferenceClient:
    provider = "ollama_local"

    def generate(
        self,
        model_config: dict[str, Any],
        prompt: str,
        ocr_json: dict,
        image_base64: str | None = None,
    ) -> dict[str, Any]:
        model_id = model_config.get("ollama_model") or f"{model_config['name']}:latest"
        result = generate_with_image(
            model_id,
            prompt,
            image_base64,
            options=model_config.get("options"),
            timeout_seconds=model_config.get("timeout_seconds"),
            max_retries=model_config.get("max_retries"),
            response_format=model_config.get("format"),
        )
        result["provider"] = self.provider
        result["model_name"] = model_config.get("name", model_id.split(":")[0])
        return result


def create_inference_client(model_config: dict[str, Any], inference_mode: str):
    if inference_mode == "remote":
        return RemoteInferenceClient(model_config)
    if inference_mode == "local":
        return LocalOllamaInferenceClient()
    raise ValueError(f"Unsupported inference_mode: {inference_mode}")
