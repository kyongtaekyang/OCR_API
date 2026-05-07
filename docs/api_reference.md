# Writing Benchmark API Reference

Base URL:

```text
http://localhost:8002
```

## 1. Dashboard

### `GET /`

웹 대시보드를 반환합니다.

주요 기능:

- 이미지 업로드
- 프롬프트 2개 입력
- 정답 분석지 JSON 입력
- docTR OCR 실행
- qwen2/gemma 분석 실행
- 모델별 산출 JSON 및 비교 보고서 표시

## 2. Health and Model Status

### `GET /health`

환경 상태 점검 결과를 반환합니다.

```json
{
  "overall": "PASS",
  "passed": 23,
  "total": 23,
  "checks": []
}
```

### `GET /models`

설정된 모델과 Ollama 사용 가능 여부를 반환합니다.

```json
{
  "models": [
    {
      "name": "qwen2",
      "provider": "ollama_local",
      "enabled": true,
      "ollama_model": "qwen2.5vl:latest",
      "available": true
    }
  ]
}
```

## 3. Dashboard Job API

### `POST /api/run`

이미지, 프롬프트, 정답 분석지를 업로드하고 백그라운드 작업을 시작합니다.

Form fields:

| 필드 | 타입 | 필수 | 설명 |
|---|---|---:|---|
| `image` | file | 예 | 작문 이미지 |
| `system_prompt` | string | 예 | 시스템 프롬프트 |
| `output_prompt` | string | 예 | 출력 프롬프트 |
| `expected` | string | 아니오 | 정답 분석지 JSON 문자열 |
| `run_name` | string | 아니오 | 실행 이름 |

응답:

```json
{
  "job_id": "..."
}
```

curl 예:

```powershell
curl -X POST "http://localhost:8002/api/run" ^
  -F "image=@data/input_images/sample.jpg" ^
  -F "system_prompt=<data/prompts/system_prompt.txt" ^
  -F "output_prompt=<data/prompts/output_prompt.txt" ^
  -F "expected=<data/expected_outputs/sample_expected.json" ^
  -F "run_name=sample_dashboard_run"
```

### `GET /api/run/{job_id}`

작업 상태와 완료 결과를 반환합니다.

실행 중:

```json
{
  "status": "running",
  "step": "이미지를 OCR JSON으로 변환 중...",
  "progress": 0.12,
  "result": null,
  "error": null
}
```

완료 시 주요 구조:

```json
{
  "status": "complete",
  "progress": 1.0,
  "result": {
    "run_name": "sample_dashboard_run",
    "ocr_json": {},
    "expected_output": {},
    "models": [
      {
        "model_name": "qwen2.5vl",
        "success": true,
        "parsed_output": {},
        "detail_diff": {},
        "comparison_report": "results/reports/.../qwen2_analysis_comparison.md"
      }
    ]
  }
}
```

## 4. Direct Writing Analysis API

### `POST /analyze-writing`

이미지 또는 OCR JSON을 입력받아 설정된 모델 전체를 실행합니다.

Form fields:

| 필드 | 타입 | 필수 | 설명 |
|---|---|---:|---|
| `image_file` | file | 조건부 | 이미지 입력. `ocr_json_file`이 없으면 필요 |
| `ocr_json_file` | file | 조건부 | OCR JSON 입력. 있으면 이미지보다 우선 |
| `system_prompt_file` | file | 아니오 | 시스템 프롬프트 파일 |
| `output_prompt_file` | file | 아니오 | 출력 프롬프트 파일 |
| `expected_output_file` | file | 아니오 | 정답 분석지 JSON |
| `run_name` | string | 아니오 | 실행 이름 |
| `models` | string | 아니오 | `qwen2,gemma` 형식 |
| `inference_mode` | string | 아니오 | `local` 또는 `remote` |

```powershell
curl -X POST "http://localhost:8002/analyze-writing" ^
  -F "image_file=@data/input_images/sample.jpg" ^
  -F "expected_output_file=@data/expected_outputs/sample_expected.json" ^
  -F "inference_mode=local" ^
  -F "run_name=api_sample_run"
```

### `POST /analyze-writing/qwen2`

`qwen2`만 로컬 Ollama로 실행합니다.

```powershell
curl -X POST "http://localhost:8002/analyze-writing/qwen2" ^
  -F "image_file=@data/input_images/sample.jpg" ^
  -F "expected_output_file=@data/expected_outputs/sample_expected.json"
```

### `POST /analyze-writing/gemma`

`gemma`만 로컬 Ollama로 실행합니다.

```powershell
curl -X POST "http://localhost:8002/analyze-writing/gemma" ^
  -F "image_file=@data/input_images/sample.jpg" ^
  -F "expected_output_file=@data/expected_outputs/sample_expected.json"
```

## 5. Validation API

### `POST /validate-output`

모델 산출 JSON 파일이 요구 스키마를 만족하는지 점검합니다.

```powershell
curl -X POST "http://localhost:8002/validate-output" ^
  -F "output_json_file=@results/parsed_outputs/sample_run/gemma_parsed.json"
```

## 6. Legacy Benchmark API

### `POST /benchmark`

기존 CLI 호환용 엔드포인트입니다. 최신 docTR OCR 기반 흐름을 사용할 때는 `/api/run` 또는 `/analyze-writing` 사용을 권장합니다.

Form fields:

| 필드 | 타입 | 필수 |
|---|---|---:|
| `image` | file | 예 |
| `system_prompt` | file | 예 |
| `output_prompt` | file | 예 |
| `expected_output` | file | 예 |
| `run_name` | string | 아니오 |

## 7. Response Artifact Mapping

| 응답/파일 | 설명 |
|---|---|
| `ocr_json` | docTR OCR 결과 |
| `models[].analysis` | API 응답의 모델 산출 JSON |
| `models[].comparison` | 정답 JSON과 산출 JSON 비교 |
| `benchmark_summary.models[].parsed_output` | 대시보드 표시용 모델 산출 JSON |
| `benchmark_summary.expected_output` | 제공 정답 분석 JSON |
| `comparison_report` | 모델별 Markdown 비교 보고서 경로 |

## 8. 권장 호출 순서

1. `POST /api/run`
2. `GET /api/run/{job_id}`를 주기적으로 polling
3. `status == complete`일 때 `result.ocr_json`, `result.expected_output`, `result.models[]` 사용
4. 모델별 산출 JSON과 비교 결과 저장 또는 화면 표시
