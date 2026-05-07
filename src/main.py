"""CLI entry point for the writing benchmark system."""
import argparse
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from src.health_check import run_health_check, print_health_check
from src.benchmark_runner import run_benchmark
from src.analysis_service import run_writing_analysis
from src.utils import timestamp


def _parse_args():
    parser = argparse.ArgumentParser(description="Writing Image Benchmark CLI")
    parser.add_argument("--health-check", action="store_true",
                        help="Run environment and model health check")
    parser.add_argument("--image", help="Path to writing image file")
    parser.add_argument("--ocr-json", help="Path to OCR result JSON file")
    parser.add_argument("--system-prompt", help="Path to system_prompt.txt")
    parser.add_argument("--output-prompt", help="Path to output_prompt.txt")
    parser.add_argument("--prompt", help="Path to pre-built combined prompt (overrides --system-prompt/--output-prompt)")
    parser.add_argument("--expected", help="Path to expected output JSON")
    parser.add_argument("--inference-mode", choices=["local", "remote"], default=None,
                        help="Inference mode: local Ollama or remote AWS endpoint")
    parser.add_argument("--models", help="Comma-separated model names, e.g. qwen2,gemma")
    parser.add_argument("--run-name", help="Name for this benchmark run",
                        default=f"run_{timestamp()}")
    return parser.parse_args()


def _print_summary(summary: dict) -> None:
    SEP = "─" * 60
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK COMPLETE: {summary['run_name']}")
    print(f"  {summary.get('started_at','')[:19]} → {summary.get('ended_at','')[:19]}")
    print(f"{'=' * 60}")

    winner = summary.get("winner", {})
    print(f"\n  WINNER: {winner.get('best_overall_model','N/A')}")
    print(f"    정확도 1위:   {winner.get('best_accuracy_model','N/A')}")
    print(f"    속도 1위:     {winner.get('fastest_model','N/A')}")
    print(f"    API 적합 1위: {winner.get('best_api_candidate','N/A')}")

    for m in summary.get("models", []):
        print(f"\n{SEP}")
        suc_str = "✓ 성공" if m.get("success") else "✗ 실패"
        print(f"  모델: {m['ollama_model']}  [{suc_str}]")
        print(f"    처리시간: {m.get('duration_seconds','N/A')} s  |  "
              f"Retry: {m.get('retry_count',0)}  |  "
              f"Timeout: {m.get('timeout',False)}")
        print(f"    JSON 파싱:    {'✓' if m.get('json_parse_success') else '✗'}  |  "
              f"스키마 유형: {m.get('schema_type','unknown')}")
        print(f"    점수 수학:    {'✓ valid' if m.get('score_math_valid') else '✗ invalid'}  |  "
              f"이상 감지: {len(m.get('score_anomalies',[]))}건")
        print(f"\n    ── 정답 대비 정확도 ──")
        print(f"    전체 Accuracy:      {m.get('overall_accuracy_score',0):.4f}")
        print(f"    API Readiness:      {m.get('api_readiness_score',0):.4f}")
        print(f"    스키마 준수율:      {m.get('schema_compliance_score',0):.4f}")
        print(f"    필드 완성도:        {m.get('required_key_completeness',0):.4f}")
        print(f"    Original 유사도:    {m.get('original_writing_similarity',0):.4f}")
        print(f"    Corrected 유사도:   {m.get('corrected_writing_similarity',0):.4f}")
        print(f"    오류 태그 F1:       {m.get('error_tag_f1',0):.4f}")
        ts_diff = m.get('total_score_difference')
        print(f"    총점 차이:          {ts_diff if ts_diff is not None else 'N/A'}")

        dd = m.get("detail_diff") or {}
        ic = dd.get("internal_consistency") or {}
        if ic:
            print(f"\n    ── 내부 일관성 (추론 지표) ──")
            print(f"    태그/설명/분석 수: {ic.get('tag_count_in_original',0)} / "
                  f"{ic.get('error_explanations_count',0)} / "
                  f"{ic.get('error_analysis_total',0)}  "
                  f"{'→ 일치 ✓' if ic.get('tag_exp_analysis_consistent') else '→ 불일치 ✗'}")
            print(f"    점수 수학 유효:   {'✓' if ic.get('subtotal_math_valid') else '✗'}  |  "
                  f"성적 수학 유효: {'✓' if ic.get('writing_performance_math_valid') else '✗'}  |  "
                  f"총점 공식: {'✓' if ic.get('total_score_formula_valid') else '✗'}")
            print(f"    무표 수정 감지:   {'⚠ 있음' if ic.get('silent_fix_detected') else '없음'}")
            print(f"    일관성 점수:      {ic.get('consistency_score',0):.0%}")

        flags = dd.get("summary_flags") or {}
        tag_f1_list = dd.get("per_tag_f1") or []
        if tag_f1_list:
            print(f"\n    ── 오류 유형별 F1 ──")
            for t in tag_f1_list:
                if t.get("status") == "n/a":
                    continue
                bar = "█" * int(t["f1"] * 10) + "░" * (10 - int(t["f1"] * 10))
                print(f"    {t['label']:<22} F1={t['f1']:.3f}  [{bar}]  "
                      f"(정답:{t['expected_count']} 모델:{t['actual_count']})  {t['status']}")

    print(f"\n{SEP}")
    print(f"\n  결과 저장 위치: results/reports/{summary['run_name']}/")
    print(f"    benchmark_summary.json")
    print(f"    benchmark_report.md")
    html_path = summary.get("html_report_path")
    if html_path:
        print(f"    benchmark_report.html  ← 상세 HTML 리포트")
    print()


def main():
    args = _parse_args()

    if args.health_check:
        result = run_health_check()
        print_health_check(result)
        sys.exit(0 if result["overall"] == "PASS" else 1)

    mode = args.inference_mode or "local"

    if args.ocr_json or mode == "remote":
        if not args.ocr_json and not args.image:
            print("ERROR: Missing required arguments: --ocr-json or --image")
            sys.exit(1)
        missing_prompt = []
        if not args.system_prompt:
            missing_prompt.append("--system-prompt")
        if not args.output_prompt:
            missing_prompt.append("--output-prompt")
        if missing_prompt:
            print(f"ERROR: Missing required arguments: {', '.join(missing_prompt)}")
            sys.exit(1)
        summary = run_writing_analysis(
            run_name=args.run_name,
            image_path=args.image,
            ocr_json_path=args.ocr_json,
            system_prompt_path=args.system_prompt,
            output_prompt_path=args.output_prompt,
            expected_output_path=args.expected,
            model_names=[m.strip() for m in args.models.split(",")] if args.models else None,
            inference_mode=mode,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        sys.exit(0)

    required = {"image": args.image, "expected": args.expected}
    if not args.prompt:
        required["system_prompt"] = args.system_prompt
        required["output_prompt"] = args.output_prompt

    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"ERROR: Missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        sys.exit(1)

    summary = run_benchmark(
        image_path=args.image,
        system_prompt_path=args.system_prompt or "",
        output_prompt_path=args.output_prompt or "",
        expected_path=args.expected,
        run_name=args.run_name,
        prompt_path=args.prompt,
    )

    _print_summary(summary)


if __name__ == "__main__":
    main()
