# AWS Model Runtime TODO

This document tracks the work required to run the current local Ollama models from AWS storage and a GPU inference runtime. It is a TODO list, not a completed deployment.

## Target Models

- `qwen2` API alias: local source model `qwen2.5vl:latest`
- `gemma` API alias: local source model `gemma4:latest`

## Planned Storage Layout

```text
s3://<MODEL_ARTIFACT_BUCKET>/ollama/qwen2.5vl/latest/
s3://<MODEL_ARTIFACT_BUCKET>/ollama/gemma4/latest/
```

## TODO

1. Decide the model artifact format for Ollama-compatible serving.
2. Export or package the local model artifacts from a controlled build machine.
3. Create the S3 model artifact bucket and upload the packaged model directories.
4. Create IAM permissions so the GPU inference runtime can read only the required S3 prefixes.
5. Build a GPU runtime image that can download/cache the S3 artifact and serve inference.
6. Expose qwen2 and gemma runtime endpoints behind ALB, API Gateway, SageMaker, ECS on EC2 GPU, or EKS GPU nodes.
7. Configure `.env` values:
   - `MODEL_ARTIFACT_BUCKET`
   - `QWEN2_INFERENCE_ENDPOINT`
   - `QWEN2_API_KEY`
   - `GEMMA_INFERENCE_ENDPOINT`
   - `GEMMA_API_KEY`
8. Switch API calls from local validation mode to `inference_mode=remote`.
9. Add integration tests marked with `@pytest.mark.integration` for real AWS endpoint calls.
10. Add CloudWatch logs, metrics, alarms, and S3 artifact retention policy.

## Current Local Validation Path

Use `http://localhost:8000/test-page` to validate the API flow against local Ollama before the AWS runtime is available.
