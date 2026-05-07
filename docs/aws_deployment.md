# AWS Deployment Design

## Recommended Architecture

```text
Client
  -> API Gateway or ALB
  -> FastAPI container on ECS/Fargate or ECS on EC2
  -> qwen2/gemma GPU model inference endpoint
  -> S3 model artifact bucket
  -> S3 input/output artifact bucket
  -> CloudWatch Logs
  -> Secrets Manager for API keys
```

## Model Storage Options

- S3 model artifact storage for immutable model versions.
- EBS model cache for a single EC2/GPU serving node.
- EFS shared model cache when multiple serving nodes need the same artifact.
- SageMaker endpoint for managed model hosting.
- ECS or EC2 GPU serving when you operate the runtime directly.

Recommended keys for the current local Ollama models:

```text
s3://<MODEL_ARTIFACT_BUCKET>/ollama/qwen2.5vl/latest/
s3://<MODEL_ARTIFACT_BUCKET>/ollama/gemma4/latest/
```

The FastAPI app does not log or expose the bucket secret configuration. It sends only the configured artifact URI and source model name to the inference runtime. The GPU runtime must have IAM permission to read the artifact bucket.

## API Server Role

The FastAPI server receives file uploads or OCR JSON, loads prompt files, builds the model prompt, attaches the model artifact manifest, orchestrates qwen2 and gemma endpoint calls, stores raw and parsed artifacts, validates JSON output, optionally compares against expected output, and returns the final response JSON.

## Model Server Role

The model server owns qwen2 and gemma inference, GPU runtime setup, model loading and warmup, S3 artifact download/cache, request execution, and JSON response generation. It should expose a stable HTTP endpoint that accepts prompt, OCR JSON, optional image base64, model storage metadata, and deterministic generation options.

Expected request fields include:

```json
{
  "model": "qwen2",
  "source_local_model": "qwen2.5vl:latest",
  "model_storage": {
    "storage_provider": "s3",
    "s3_uri": "s3://bucket/ollama/qwen2.5vl/latest/",
    "runtime": "ollama"
  },
  "input": {
    "prompt": "...",
    "ocr_json": {}
  },
  "options": {
    "temperature": 0,
    "response_format": "json"
  }
}
```

## Why Lambda Is Not Suitable For Inference

Lambda is not a good runtime for large VLM inference because of cold start cost, GPU requirements, execution time limits, memory limits, and model artifact size. Lambda can still be useful as a lightweight orchestration or routing layer in front of a GPU-backed compute layer.

## Recommended Rollout

- Development and validation: local FastAPI plus remote inference endpoint.
- Production phase 1: ECS on EC2 GPU with ALB and S3 artifact storage.
- Production phase 2: SageMaker endpoint or EKS GPU node groups for autoscaling and stronger operational controls.
- Results: store input images, OCR JSON, raw outputs, parsed JSON, reports, and logs in S3.
- Logs: stream application and model server logs to CloudWatch.
- Secrets: store API keys in Secrets Manager and inject as environment variables.

## Production Notes

- Keep model inference separate from the API server when possible.
- Use request IDs and run names to correlate FastAPI logs, model logs, and S3 artifacts.
- Mask endpoint URLs and never log API keys, presigned URLs, or full secrets.
- Use timeout and retry policies that match model latency.
- Prefer asynchronous job orchestration for long-running inference.
