# Writing Benchmark 및 API 사용 가이드

작성일: 2026-05-07  
프로젝트 경로: `C:\Project\writing_api`

## 1. 개요

이 시스템은 학생 영어 작문 이미지와 정답 분석지 JSON을 입력받아, 두 로컬 Ollama 모델(`qwen2.5vl:latest`, `gemma4:latest`)의 분석 결과를 정답 분석지와 비교하는 벤치마크/API 도구입니다.

현재 파이프라인은 이미지를 모델에 직접 넣지 않습니다. 먼저 `docTR` OCR로 이미지를 텍스트화하고 OCR JSON을 만든 뒤, OCR JSON과 사용자가 입력한 프롬프트 2개를 각 모델에 전달합니다.

## 2. 전체 처리 흐름

1. 이미지 파일 입력
2. 프롬프트 2개 입력: `system_prompt`, `output_prompt`
3. 정답 분석지 JSON 입력
4. 이미지 파일을 `docTR`로 OCR 처리
5. OCR 결과를 `results/ocr/<run_name>/ocr_result.json`으로 저장
6. OCR JSON + 프롬프트 2개를 `qwen2`, `gemma` 모델에 각각 입력
7. 모델 분석 결과 JSON 생성
8. 모델 분석 JSON을 정답 분석지 JSON과 동일한 키/중첩 구조로 정규화
9. 정답 분석지 JSON과 모델별 산출 JSON 비교
10. 모델별 항목 차이 보고서 생성

## 3. 실행 준비

```powershell
cd C:\Project\writing_api
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

필수 OCR 관련 패키지:

- `python-doctr`
- `torch`
- `torchvision`

Ollama 모델 준비:

```powershell
ollama list
ollama pull qwen2.5vl:latest
ollama pull gemma4:latest
```

상태 점검:

```powershell
python -m src.main --health-check
```

## 4. 웹 대시보드 사용법

서버 실행:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8002
```

접속:

```text
http://localhost:8002/
```

화면 입력 순서:

1. 학생 작문 이미지 업로드
2. 시스템 프롬프트 입력
3. 출력 프롬프트 입력
4. 정답 분석지 JSON 입력
5. 실행 이름 입력 또는 비워둠
6. `벤치마크 실행` 클릭

결과 화면 표시 순서:

1. `docTR OCR 결과 JSON`
2. `제공 정답 분석 JSON`
3. `qwen2 산출 분석 JSON`
4. `qwen2: 제공 정답 JSON vs 산출 JSON 비교`
5. `gemma 산출 분석 JSON`
6. `gemma: 제공 정답 JSON vs 산출 JSON 비교`
7. 종합 비교 보고서

## 5. 정답 분석지 JSON

정답 분석지는 모델 산출 JSON의 기준 포맷입니다. 모델 결과는 최종적으로 이 정답 분석지와 동일한 JSON 키 구조로 정규화됩니다.

대표 구조:

```json
{
  "metadata": {
    "course_type": "",
    "class": "",
    "title": "",
    "title_corrected": "",
    "topic": "",
    "topic_corrected": "",
    "writing_type": ""
  },
  "original_writing": "",
  "corrected_writing": "",
  "scoring_analysis": {},
  "writing_performance": {},
  "error_explanations": [],
  "error_analysis": {},
  "overall_comments": ""
}
```

주의:

- 정답 분석지 값은 비교 기준입니다.
- 모델 프롬프트에는 정답 JSON의 값이 아니라 구조만 참조하도록 전달됩니다.
- 최종 저장되는 모델 JSON은 정답 JSON과 같은 구조로 맞춘 결과입니다.
- 원본 모델 응답은 `results/raw_outputs/<run_name>/`에서 확인합니다.

## 6. 산출 파일 위치

실행 이름이 `<run_name>`일 때 주요 결과는 다음 위치에 저장됩니다.

| 경로 | 설명 |
|---|---|
| `results/ocr/<run_name>/ocr_result.json` | docTR OCR 결과 JSON |
| `results/raw_outputs/<run_name>/<model>_raw.txt` | 모델 원본 응답 |
| `results/parsed_outputs/<run_name>/<model>_parsed.json` | 정답 JSON 구조로 정규화된 모델 분석 JSON |
| `results/comparisons/<run_name>/<model>_comparison.json` | 모델별 비교 메트릭 |
| `results/reports/<run_name>/<model>_analysis_comparison.md` | 모델별 항목 차이 분석 보고서 |
| `results/reports/<run_name>/benchmark_summary.json` | 전체 실행 요약 |
| `results/reports/<run_name>/benchmark_report.md` | 전체 벤치마크 리포트 |
| `results/reports/<run_name>/combined_eval_prompt.txt` | 실제 사용된 통합 프롬프트 |
| `results/logs/<run_name>/run.log` | 실행 로그 |

## 7. 비교 보고서 항목

| 항목 | 설명 |
|---|---|
| JSON 파싱 성공 여부 | 모델 응답이 JSON으로 파싱되었는지 |
| 스키마 일치도 | 필수 키와 점수 구조가 얼마나 맞는지 |
| 원문 유사도 | `original_writing` 유사도 |
| 교정문 유사도 | `corrected_writing` 유사도 |
| 오류 태그 F1 | `[G]`, `[V]`, `[O]`, `[P]`, `[S]`, `[C]` 오류 태그 비교 |
| 점수 차이 | 문법, 어휘, 글 흐름, 총점 차이 |
| 오류 개수 차이 | `error_analysis`의 유형별 개수 차이 |
| 항목별 차이 | JSON leaf path 단위 expected/actual 비교 |

## 8. 모델 설정

모델 설정 파일:

```text
config/models.json
```

주요 옵션:

```json
{
  "temperature": 0,
  "num_predict": 2048,
  "num_ctx": 4096,
  "keep_alive": "0s"
}
```

권장 사항:

- `num_predict`가 너무 크면 Ollama runner가 메모리 문제로 중단될 수 있습니다.
- `keep_alive: "0s"`는 모델 호출 후 메모리 해제를 유도합니다.
- `qwen2`는 기본적으로 compact prompt 모드를 사용합니다.

## 9. 문제 해결

### `No module named 'doctr'`

서버를 실행한 Python 환경에 `python-doctr`가 설치되어 있지 않은 상태입니다.

```powershell
pip install python-doctr torch torchvision
```

### docTR 캐시 권한 오류

시스템은 docTR 캐시를 프로젝트 내부 `.cache/doctr`에 저장하도록 설정합니다. 그래도 권한 오류가 나면 캐시 폴더를 삭제 후 다시 실행합니다.

```powershell
Remove-Item -Recurse -Force .cache\doctr
```

### Ollama runner stopped

메모리 또는 모델 로딩 문제일 가능성이 큽니다.

조치:

- `config/models.json`의 `num_predict`를 낮춥니다.
- `num_ctx`를 낮춥니다.
- Ollama 모델을 unload합니다.

```powershell
$body=@{model='qwen2.5vl:latest';keep_alive='0s'} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:11434/api/generate -Method Post -Body $body -ContentType 'application/json'
```

## 10. 테스트

```powershell
pytest
```

주요 테스트:

- API 업로드/응답 테스트
- OCR JSON 스키마 테스트
- 프롬프트 생성 테스트
- 비교 메트릭 테스트
- JSON validator 테스트
