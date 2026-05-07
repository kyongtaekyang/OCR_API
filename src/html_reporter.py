"""Rich HTML benchmark report generator (정답 분석지 비교 전용)."""

import html as _html_lib
import json
from pathlib import Path
from src.utils import PROJECT_ROOT, ensure_dir, sanitize_run_name


# ── helpers ──────────────────────────────────────────────────────────────────

def _e(v) -> str:
    return _html_lib.escape(str(v)) if v is not None else ""


def _fmt(v, d: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return _e(str(v))


def _score_color(v: float | None) -> str:
    if v is None:
        return "#94a3b8"
    if float(v) >= 0.8:
        return "#15803d"
    if float(v) >= 0.5:
        return "#b45309"
    return "#b91c1c"


def _status_color(status: str) -> str:
    return {
        "match": "#15803d", "perfect": "#15803d", "exact": "#15803d",
        "within_1": "#15803d", "matched": "#15803d",
        "near_match": "#15803d",
        "off_by_small": "#b45309", "within_3": "#b45309",
        "over_tagged": "#b45309", "under_tagged": "#b45309",
        "partial": "#b45309", "tag_mismatch": "#b45309", "extra": "#b45309",
        "off_by_large": "#b91c1c", "off": "#b91c1c",
        "missed": "#b91c1c", "different": "#b91c1c",
        "missing": "#94a3b8", "n/a": "#94a3b8",
    }.get(status, "#1e293b")


def _badge(text: str, fg: str = "#1d4ed8", bg: str = "#dbeafe") -> str:
    return (f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:999px;font-size:.72rem;font-weight:700">{_e(text)}</span>')


def _ok(ok: bool, yes: str = "✓ PASS", no: str = "✗ FAIL") -> str:
    return _badge(yes, "#15803d", "#dcfce7") if ok else _badge(no, "#b91c1c", "#fee2e2")


def _bar(v: float | None, w: int = 130) -> str:
    if v is None:
        return "—"
    pct = round(min(max(float(v), 0.0), 1.0) * 100)
    c = _score_color(float(v))
    return (f'<div style="display:flex;align-items:center;gap:6px">'
            f'<div style="width:{w}px;background:#e2e8f0;border-radius:4px;height:11px;overflow:hidden">'
            f'<div style="width:{pct}%;height:100%;background:{c};border-radius:4px"></div></div>'
            f'<span style="font-size:.78rem;color:#334155">{pct}%</span></div>')


def _match(ok: bool) -> str:
    return ('<span style="color:#15803d;font-weight:700">✓</span>'
            if ok else '<span style="color:#b91c1c">✗</span>')


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f1f5f9;color:#1e293b;line-height:1.6;font-size:14px}
.wrap{max-width:1300px;margin:0 auto;padding:24px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px 24px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.card-title{font-size:.95rem;font-weight:700;color:#0f172a;margin-bottom:14px;padding-bottom:10px;border-bottom:2px solid #f1f5f9}
table{width:100%;border-collapse:collapse;font-size:.82rem}
thead th{background:#f8fafc;color:#475569;text-align:left;padding:8px 12px;border-bottom:2px solid #e2e8f0;font-size:.74rem;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
td{padding:7px 12px;border-bottom:1px solid #f8fafc;vertical-align:middle}
tr:last-child td{border-bottom:none}
.mono{font-family:'Courier New',monospace;font-size:.8rem}
.muted{color:#64748b}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.model-border-ok{border-left:4px solid #15803d}
.model-border-fail{border-left:4px solid #b91c1c}
textarea{width:100%;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;font-family:'Courier New',monospace;font-size:.78rem;line-height:1.9;resize:vertical;min-height:70px;color:#1e293b}
details summary{cursor:pointer;font-weight:600;color:#2563eb;font-size:.82rem;padding:6px 0;user-select:none}
details[open] summary{margin-bottom:10px}
hr.sec{border:none;border-top:1px solid #f1f5f9;margin:12px 0}
"""


# ── section builders ──────────────────────────────────────────────────────────

def _sec_header(summary: dict) -> str:
    run = summary.get("run_name", "")
    img = summary.get("input_image", "")
    started = (summary.get("started_at") or "")[:19].replace("T", " ")
    ended = (summary.get("ended_at") or "")[:19].replace("T", " ")
    return (
        f'<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);color:#fff;'
        f'border-radius:12px;padding:24px;margin-bottom:16px">'
        f'<div style="font-size:1.5rem;font-weight:800">📊 Writing Benchmark Report</div>'
        f'<div style="color:#94a3b8;margin-top:8px;font-size:.85rem">'
        f'실행: <strong style="color:#e2e8f0">{_e(run)}</strong> &nbsp;|&nbsp; '
        f'이미지: <strong style="color:#e2e8f0">{_e(img)}</strong><br>'
        f'{_e(started)} → {_e(ended)}</div></div>'
    )


def _sec_winner(winner: dict) -> str:
    labels = {
        "best_overall_model": "종합 최우수",
        "best_accuracy_model": "정확도 1위",
        "fastest_model": "처리속도 1위",
        "best_api_candidate": "API 적합 1위",
        "best_aws_candidate": "AWS 배포 1위",
    }
    rows = "".join(
        f'<tr><td style="color:#bfdbfe;font-size:.8rem;padding:3px 12px 3px 0">{_e(labels.get(k,k))}</td>'
        f'<td style="font-weight:700;color:#fff;padding:3px 0">{_e(v)}</td></tr>'
        for k, v in winner.items()
    )
    best = winner.get("best_overall_model", "N/A")
    return (
        f'<div style="background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;'
        f'border-radius:12px;padding:20px 24px;margin-bottom:16px;'
        f'display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start">'
        f'<div><div style="font-size:.75rem;color:#bfdbfe;text-transform:uppercase;letter-spacing:.08em">종합 최우수 모델</div>'
        f'<div style="font-size:2rem;font-weight:800;margin-top:4px">{_e(best)}</div></div>'
        f'<table style="width:auto"><tbody>{rows}</tbody></table></div>'
    )


def _sec_overview(models: list) -> str:
    def row(m):
        suc = m.get("success", False)
        return (
            f'<tr>'
            f'<td><strong>{_e(m.get("ollama_model",""))}</strong>'
            f'<br><span class="muted" style="font-size:.75rem">{_e(m.get("prompt_mode",""))}</span></td>'
            f'<td>{_ok(suc,"✓ 성공","✗ 실패")}</td>'
            f'<td class="mono">{_fmt(m.get("duration_seconds"),1)} s</td>'
            f'<td>{_bar(m.get("overall_accuracy_score"))}</td>'
            f'<td>{_bar(m.get("api_readiness_score"))}</td>'
            f'<td>{_ok(m.get("json_parse_success",False),"✓","✗")}</td>'
            f'<td>{_bar(m.get("schema_compliance_score"))}</td>'
            f'<td>{_bar(m.get("required_key_completeness"))}</td>'
            f'<td>{_ok(m.get("score_math_valid",False),"✓","✗")}</td>'
            f'</tr>'
        )
    rows = "".join(row(m) for m in models)
    return (
        f'<div class="card"><div class="card-title">📋 전체 비교 요약</div>'
        f'<table><thead><tr>'
        f'<th>모델</th><th>상태</th><th>처리시간</th>'
        f'<th style="min-width:140px">정확도 (vs 정답)</th>'
        f'<th style="min-width:140px">API 적합성</th>'
        f'<th>JSON</th><th style="min-width:120px">스키마 준수</th>'
        f'<th style="min-width:120px">필드 완성도</th><th>점수 수학</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _sec_time_chart(models: list) -> str:
    max_dur = max((m.get("duration_seconds") or 0 for m in models), default=1) or 1
    colors = ["#2563eb", "#16a34a", "#ca8a04", "#dc2626"]
    bars = ""
    for i, m in enumerate(models):
        dur = m.get("duration_seconds") or 0
        pct = round(dur / max_dur * 100)
        c = colors[i % len(colors)]
        limit = m.get("timeout_seconds") or 300
        retry = m.get("retry_count", 0)
        retry_str = f" (재시도 {retry}회)" if retry else ""
        bars += (
            f'<div style="margin-bottom:14px">'
            f'<div style="display:flex;justify-content:space-between;font-size:.82rem;color:#475569;margin-bottom:4px">'
            f'<strong>{_e(m.get("ollama_model",""))}</strong>'
            f'<span class="mono">{_fmt(dur,1)} s / {limit} s 제한{_e(retry_str)}</span></div>'
            f'<div style="background:#e2e8f0;border-radius:6px;height:22px;overflow:hidden">'
            f'<div style="width:{pct}%;height:100%;background:{c};border-radius:6px"></div></div></div>'
        )
    return (
        f'<div class="card"><div class="card-title">⏱ 처리 시간 비교</div>'
        f'{bars}'
        f'<p class="muted" style="font-size:.78rem">막대 길이 기준: 최대 {_fmt(max_dur,1)} s</p></div>'
    )


# ── per-model subsections ─────────────────────────────────────────────────────

def _sub_score_table(dd: dict) -> str:
    sub = (dd or {}).get("subcategory_diffs")
    if not sub:
        return '<p class="muted">점수 비교 데이터 없음</p>'
    rows = ""
    for cat, entries in sub.items():
        for e in entries:
            st = e.get("status", "missing")
            c = _status_color(st)
            delta = e.get("score_delta")
            delta_s = f'{delta:+.1f}' if delta is not None and delta != 0 else ("±0" if delta == 0 else "—")
            max_s = f'/{_fmt(e.get("max_score"),0)}' if e.get("max_score") is not None else ""
            rows += (
                f'<tr>'
                f'<td class="muted">{_e(cat)}</td>'
                f'<td><strong>{_e(e.get("subcategory",""))}</strong></td>'
                f'<td class="mono">{_fmt(e.get("expected_score"),1)}{max_s}</td>'
                f'<td class="mono">{_fmt(e.get("actual_score"),1)}{max_s}</td>'
                f'<td class="mono" style="color:{c};font-weight:600">{_e(delta_s)}</td>'
                f'<td><span style="color:{c};font-size:.75rem;font-weight:600">{_e(st)}</span></td>'
                f'</tr>'
            )
    return (
        f'<table><thead><tr>'
        f'<th>카테고리</th><th>항목</th><th>정답</th><th>모델</th><th>차이</th><th>상태</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def _sub_wp_diffs(dd: dict) -> str:
    wp = (dd or {}).get("writing_performance_diffs")
    if not wp:
        return '<p class="muted">데이터 없음</p>'
    rows = ""
    for w in wp:
        st = w.get("status", "missing")
        c = _status_color(st)
        rows += (
            f'<tr>'
            f'<td><strong>{_e(w.get("domain",""))}</strong></td>'
            f'<td class="mono">{_fmt(w.get("expected_score"),1)}</td>'
            f'<td class="mono">{_fmt(w.get("actual_score"),1)}</td>'
            f'<td class="mono" style="color:{c};font-weight:600">{_fmt(w.get("delta"),1)}</td>'
            f'<td><span style="color:{c};font-size:.75rem;font-weight:600">{_e(st)}</span></td>'
            f'</tr>'
        )
    return (
        f'<table><thead><tr>'
        f'<th>도메인</th><th>정답 점수</th><th>모델 점수</th><th>차이</th><th>상태</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def _sub_tag_f1(dd: dict) -> str:
    tags = (dd or {}).get("per_tag_f1")
    if not tags:
        return '<p class="muted">데이터 없음</p>'
    rows = ""
    for t in tags:
        f1 = t.get("f1", 0)
        st = t.get("status", "")
        c = _status_color(st)
        dim = 'style="opacity:.45"' if st == "n/a" else ""
        rows += (
            f'<tr {dim}>'
            f'<td><strong>{_e(t.get("label",""))}</strong></td>'
            f'<td class="mono">{t.get("expected_count",0)}</td>'
            f'<td class="mono">{t.get("actual_count",0)}</td>'
            f'<td class="mono">{t.get("tp",0)} / {t.get("fp",0)} / {t.get("fn",0)}</td>'
            f'<td class="mono">{_fmt(t.get("precision"),3)}</td>'
            f'<td class="mono">{_fmt(t.get("recall"),3)}</td>'
            f'<td>{_bar(f1, 100)}</td>'
            f'<td><span style="color:{c};font-size:.75rem;font-weight:600">{_e(st)}</span></td>'
            f'</tr>'
        )
    return (
        f'<table><thead><tr>'
        f'<th>오류 유형</th><th>정답 수</th><th>모델 수</th><th>TP/FP/FN</th>'
        f'<th>Precision</th><th>Recall</th><th>F1</th><th>상태</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def _sub_error_analysis(dd: dict) -> str:
    ea = (dd or {}).get("error_analysis_diff")
    if not ea:
        return '<p class="muted">데이터 없음</p>'
    rows = ""
    for cat, v in ea.items():
        d = v.get("diff", 0)
        rows += (
            f'<tr>'
            f'<td>{_e(cat)}</td>'
            f'<td class="mono">{_e(v.get("expected",0))}</td>'
            f'<td class="mono">{_e(v.get("actual",0))}</td>'
            f'<td class="mono" style="color:{"#15803d" if d==0 else "#b91c1c"}">'
            f'{("+"+str(d)) if d>0 else str(d)}</td>'
            f'<td>{_match(v.get("match",False))}</td>'
            f'</tr>'
        )
    return (
        f'<table><thead><tr>'
        f'<th>오류 유형</th><th>정답</th><th>모델</th><th>차이</th><th>일치</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def _sub_metadata(dd: dict) -> str:
    meta = (dd or {}).get("metadata_diffs")
    if not meta:
        return '<p class="muted">데이터 없음</p>'
    rows = ""
    for m in meta:
        match = m.get("match", False)
        bg = "#f0fdf4" if match else "#fff7f7"
        rows += (
            f'<tr style="background:{bg}">'
            f'<td class="muted">{_e(m.get("field",""))}</td>'
            f'<td class="mono">{_e(m.get("expected_value",""))}</td>'
            f'<td class="mono">{_e(m.get("actual_value",""))}</td>'
            f'<td>{_match(match)}</td>'
            f'</tr>'
        )
    return (
        f'<table><thead><tr>'
        f'<th>필드</th><th>정답</th><th>모델</th><th>일치</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def _sub_text_diffs(dd: dict) -> str:
    texts = (dd or {}).get("text_char_diffs")
    if not texts:
        return '<p class="muted">데이터 없음</p>'
    rows = ""
    for t in texts:
        rows += (
            f'<tr>'
            f'<td><strong>{_e(t.get("field",""))}</strong></td>'
            f'<td class="mono">{_e(t.get("expected_len",0))}</td>'
            f'<td class="mono">{_e(t.get("actual_len",0))}</td>'
            f'<td class="mono">{_e(t.get("equal_chars",0))}</td>'
            f'<td class="mono" style="color:#b91c1c">{_e(t.get("insert_chars",0))}</td>'
            f'<td class="mono" style="color:#b45309">{_e(t.get("delete_chars",0))}</td>'
            f'<td class="mono" style="color:#7c3aed">{_e(t.get("replace_chars",0))}</td>'
            f'<td>{_bar(t.get("char_accuracy"))}</td>'
            f'<td>{_bar(t.get("similarity"))}</td>'
            f'</tr>'
        )
    return (
        f'<table><thead><tr>'
        f'<th>필드</th><th>정답 길이</th><th>모델 길이</th><th>일치</th>'
        f'<th>삽입</th><th>삭제</th><th>교체</th><th>문자 정확도</th><th>유사도</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def _sub_explanations(dd: dict) -> str:
    diffs = (dd or {}).get("error_explanation_diffs")
    if not diffs:
        return '<p class="muted">오류 설명 데이터 없음</p>'
    rows = ""
    for d in diffs[:30]:
        st = d.get("status", "")
        c = _status_color(st)
        bg = "#f0fdf4" if st == "matched" else ("#fff7f7" if st in ("missed",) else "#fffbf0")
        rows += (
            f'<tr style="background:{bg}">'
            f'<td class="mono muted">{d.get("index",0)+1}</td>'
            f'<td class="mono">{_e(d.get("expected_error") or "—")}</td>'
            f'<td class="mono">{_e(d.get("actual_error") or "—")}</td>'
            f'<td>{_e(d.get("expected_tag") or "—")}</td>'
            f'<td>{_e(d.get("actual_tag") or "—")}</td>'
            f'<td>{_match(d.get("tag_match",False))}</td>'
            f'<td><span style="color:{c};font-size:.75rem;font-weight:600">{_e(st)}</span></td>'
            f'</tr>'
        )
    note = (f'<p class="muted" style="margin-top:8px;font-size:.75rem">최대 30개 표시 (전체 {len(diffs)}개)</p>'
            if len(diffs) > 30 else "")
    return (
        f'<table><thead><tr>'
        f'<th>#</th><th>정답 오류</th><th>모델 오류</th>'
        f'<th>정답 태그</th><th>모델 태그</th><th>태그 일치</th><th>상태</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>{note}'
    )


def _short_value(v, max_len: int = 180) -> str:
    if isinstance(v, (dict, list)):
        text = json.dumps(v, ensure_ascii=False)
    elif v is None:
        text = "—"
    else:
        text = str(v)
    return text if len(text) <= max_len else text[:max_len] + "..."


def _sub_field_diffs(dd: dict) -> str:
    rows_data = (dd or {}).get("field_diffs") or []
    if not rows_data:
        return '<p class="muted">정답지 항목별 비교 데이터 없음</p>'
    rows = ""
    for d in rows_data:
        st = d.get("status", "")
        c = _status_color(st)
        bg = "#f0fdf4" if st in ("match", "near_match") else ("#fff7f7" if st in ("different", "off_by_large", "missing") else "#fffbf0")
        rows += (
            f'<tr style="background:{bg}">'
            f'<td class="mono muted">{_e(d.get("path",""))}</td>'
            f'<td class="mono">{_e(_short_value(d.get("expected_value")))}</td>'
            f'<td class="mono">{_e(_short_value(d.get("actual_value")))}</td>'
            f'<td>{_bar(d.get("similarity"), 90)}</td>'
            f'<td><span style="color:{c};font-size:.75rem;font-weight:600">{_e(st)}</span></td>'
            f'</tr>'
        )
    return (
        f'<table><thead><tr>'
        f'<th>정답지 항목</th><th>정답 입력값</th><th>모델 출력값</th><th>유사도</th><th>상태</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
        f'<p class="muted" style="margin-top:8px;font-size:.75rem">최대 300개 leaf 항목 표시</p>'
    )


def _sub_consistency(dd: dict) -> str:
    ic = (dd or {}).get("internal_consistency")
    if not ic:
        return '<p class="muted">데이터 없음</p>'
    cs = ic.get("consistency_score", 0)
    c = _score_color(cs)
    cnt_rows = "".join(
        f'<tr><td class="muted" style="padding:4px 12px 4px 0">{_e(l)}</td>'
        f'<td class="mono"><strong>{_e(v)}</strong></td></tr>'
        for l, v in [
            ("original_writing 태그 수", ic.get("tag_count_in_original", 0)),
            ("error_explanations 수", ic.get("error_explanations_count", 0)),
            ("error_analysis.total_errors", ic.get("error_analysis_total", 0)),
        ]
    )
    ok3 = ic.get("tag_exp_analysis_consistent", False)
    chk_rows = "".join(
        f'<tr><td style="padding:4px 12px 4px 0">{_e(l)}</td><td>{_ok(v)}</td></tr>'
        for l, v in [
            ("태그 / 설명 / 분석 수 일치", ok3),
            ("서브토탈 수학 유효", ic.get("subtotal_math_valid", False)),
            ("Writing Perf. 수학 유효", ic.get("writing_performance_math_valid", False)),
            ("총점 공식 유효", ic.get("total_score_formula_valid", False)),
            ("무표 수정 미감지 (silent fix)", not ic.get("silent_fix_detected", False)),
        ]
    )
    return (
        f'<div class="grid2">'
        f'<div><h4 style="font-size:.82rem;color:#475569;margin-bottom:6px">카운트</h4>'
        f'<table><tbody>{cnt_rows}</tbody></table></div>'
        f'<div><h4 style="font-size:.82rem;color:#475569;margin-bottom:6px">규칙 준수</h4>'
        f'<table><tbody>{chk_rows}</tbody></table></div></div>'
        f'<div style="margin-top:14px;padding:14px;border-radius:8px;border:2px solid {c};'
        f'background:{"#f0fdf4" if cs>=0.8 else "#fff7f7"}">'
        f'<span class="muted" style="font-size:.82rem">내부 일관성 점수</span>'
        f'<span style="font-size:1.6rem;font-weight:800;color:{c};margin-left:12px">'
        f'{round(cs*100)}%</span>'
        f'<span class="muted" style="margin-left:6px;font-size:.8rem">'
        f'({int(cs*5+.01)}/5 체크 통과)</span></div>'
    )


def _sub_texts(exp_orig: str, exp_corr: str, parsed_json: dict | None) -> str:
    act_orig = (parsed_json or {}).get("original_writing", "")
    act_corr = (parsed_json or {}).get("corrected_writing", "")
    return (
        f'<div class="grid2" style="margin-bottom:12px">'
        f'<div><div class="muted" style="margin-bottom:4px;font-size:.78rem">🎯 정답 — Original Writing</div>'
        f'<textarea rows="5" readonly>{_e(exp_orig)}</textarea></div>'
        f'<div><div class="muted" style="margin-bottom:4px;font-size:.78rem">🤖 모델 — Original Writing</div>'
        f'<textarea rows="5" readonly>{_e(act_orig)}</textarea></div></div>'
        f'<div class="grid2">'
        f'<div><div class="muted" style="margin-bottom:4px;font-size:.78rem">🎯 정답 — Corrected Writing</div>'
        f'<textarea rows="4" readonly>{_e(exp_corr)}</textarea></div>'
        f'<div><div class="muted" style="margin-bottom:4px;font-size:.78rem">🤖 모델 — Corrected Writing</div>'
        f'<textarea rows="4" readonly>{_e(act_corr)}</textarea></div></div>'
    )


def _sec_model(
    model: dict,
    comp: dict,
    expected: dict | None,
    parsed_json: dict | None,
) -> str:
    name = model.get("ollama_model", "unknown")
    suc = model.get("success", False)
    dd = comp.get("detail_diff") or {}
    flags = dd.get("summary_flags", {})

    flag_chips = ""
    chip_map = [
        ("metadata_all_match", "메타데이터 전부 일치", "#15803d", "#dcfce7"),
        ("writing_performance_all_exact", "성적 완전 일치", "#15803d", "#dcfce7"),
        ("any_subcategory_mismatch", "서브점수 불일치 있음", "#b45309", "#fef3c7"),
        ("any_tag_f1_below_threshold", "태그 F1 낮음 (< 0.5)", "#b91c1c", "#fee2e2"),
        ("internally_consistent", "내부 일관성 OK", "#15803d", "#dcfce7"),
    ]
    for key, label, fg, bg in chip_map:
        if flags.get(key):
            flag_chips += _badge(label, fg, bg) + " "

    anomalies = model.get("score_anomalies", [])
    anomaly_html = ""
    if anomalies:
        items = "".join(
            f'<li class="mono">{_e(a.get("path"))}: {_e(a.get("issue"))} '
            f'(점수={_e(a.get("score"))}, 최대={_e(a.get("max_score"))})</li>'
            for a in anomalies
        )
        anomaly_html = (
            f'<div style="background:#fff7f7;border:1px solid #fecaca;border-radius:8px;'
            f'padding:12px;margin-bottom:12px">'
            f'<strong style="color:#b91c1c">⚠ 점수 이상 감지 ({len(anomalies)}건)</strong>'
            f'<ul style="margin-top:8px;padding-left:18px;font-size:.82rem">{items}</ul></div>'
        )

    exp_orig = (expected or {}).get("original_writing", "")
    exp_corr = (expected or {}).get("corrected_writing", "")

    sections = [
        ("🧾 정답지 전체 항목 비교 (입력값 vs 모델)", _sub_field_diffs(dd)),
        ("📊 서브카테고리 점수 비교 (정답 vs 모델)", _sub_score_table(dd)),
        ("🏆 Writing Performance 도메인 비교", _sub_wp_diffs(dd)),
        ("🏷 오류 태그 유형별 F1 점수 (G/V/O/P/S/C)", _sub_tag_f1(dd)),
        ("📝 원문/수정문 텍스트 비교", _sub_texts(exp_orig, exp_corr, parsed_json)),
        ("🔤 문자 수준 차이 분석", _sub_text_diffs(dd)),
        ("📋 오류 설명 정렬 비교 (index 순서)", _sub_explanations(dd)),
        ("📉 오류 분석 카운트 비교", _sub_error_analysis(dd)),
        ("🗂 메타데이터 필드 정확도", _sub_metadata(dd)),
        ("⚙ 내부 일관성 (추론 지표)", _sub_consistency(dd)),
    ]
    secs_html = "".join(
        f'<details open><summary>{_e(t)}</summary>'
        f'<div style="margin-top:10px">{c}</div></details>'
        f'<hr class="sec">'
        for t, c in sections
    )

    bc = "#15803d" if suc else "#b91c1c"
    cls = "model-border-ok" if suc else "model-border-fail"
    raw_comp = {k: v for k, v in comp.items() if k != "detail_diff"}

    return (
        f'<div class="card {cls}">'
        f'<div class="card-title">'
        f'<span style="color:{bc};font-size:1.1rem">{"✓" if suc else "✗"}</span> '
        f'<span>{_e(name)}</span> '
        f'<span style="font-size:.78rem;color:#64748b;font-weight:400">'
        f'{_fmt(model.get("duration_seconds"),1)} s &nbsp;|&nbsp; '
        f'Accuracy: {_fmt(model.get("overall_accuracy_score"))} &nbsp;|&nbsp; '
        f'API: {_fmt(model.get("api_readiness_score"))}'
        f'</span></div>'
        + ('<div style="margin-bottom:10px">' + flag_chips + '</div>' if flag_chips else '')
        +
        f'{anomaly_html}'
        f'{secs_html}'
        f'<details><summary>🔍 원본 비교 JSON 보기</summary>'
        f'<pre style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;'
        f'padding:12px;font-size:.72rem;overflow-x:auto;max-height:280px;margin-top:8px">'
        f'{_e(json.dumps(raw_comp, ensure_ascii=False, indent=2, default=str))}</pre>'
        f'</details></div>'
    )


def _sec_side_by_side(models: list, comps: list) -> str:
    """Score comparison table for exactly 2 models."""
    if len(models) < 2:
        return ""
    m0, m1 = models[0], models[1]
    d0 = (comps[0].get("detail_diff") or {}).get("subcategory_diffs", {})
    d1 = (comps[1].get("detail_diff") or {}).get("subcategory_diffs", {})

    rows = ""
    for cat in sorted(set(list(d0) + list(d1))):
        e0 = {e["subcategory"]: e for e in d0.get(cat, [])}
        e1 = {e["subcategory"]: e for e in d1.get(cat, [])}
        for key in sorted(set(list(e0) + list(e1))):
            v0 = e0.get(key, {})
            v1 = e1.get(key, {})
            d0v = v0.get("score_delta")
            d1v = v1.get("score_delta")
            w0 = w1 = False
            if d0v is not None and d1v is not None:
                if d0v < d1v:
                    w0 = True
                elif d1v < d0v:
                    w1 = True

            def cell(v, winner):
                st = v.get("status", "missing")
                c = _status_color(st)
                delta = v.get("score_delta")
                ds = f'{delta:+.1f}' if delta is not None and delta != 0 else ("±0" if delta == 0 else "—")
                bold = "font-weight:700" if winner else ""
                star = " ★" if winner else ""
                return (f'<td class="mono" style="color:{c};{bold}">'
                        f'{_fmt(v.get("actual_score"),1)} ({_e(ds)}){_e(star)}</td>')

            exp_score = v0.get("expected_score") or v1.get("expected_score")
            rows += (
                f'<tr><td class="muted">{_e(cat)}</td><td>{_e(key)}</td>'
                f'<td class="mono muted">{_fmt(exp_score,1)}</td>'
                f'{cell(v0, w0)}{cell(v1, w1)}</tr>'
            )

    return (
        f'<div class="card"><div class="card-title">⚖ 두 모델 서브점수 비교 (★ = 정답에 더 가까운 쪽)</div>'
        f'<table><thead><tr>'
        f'<th>카테고리</th><th>항목</th><th>정답</th>'
        f'<th>{_e(m0.get("ollama_model",""))}</th>'
        f'<th>{_e(m1.get("ollama_model",""))}</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>'
    )


# ── public API ────────────────────────────────────────────────────────────────

def generate_html_report(
    summary: dict,
    comparisons: list[dict],
    expected: dict | None = None,
    parsed_jsons: list | None = None,
) -> str:
    """Generate a complete self-contained HTML benchmark report."""
    models = summary.get("models", [])
    winner = summary.get("winner", {})
    parsed_jsons = parsed_jsons or [None] * len(models)
    comparisons = comparisons or [{}] * len(models)

    parts = [
        f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Benchmark: {_e(summary.get("run_name",""))}</title>'
        f'<style>{_CSS}</style></head><body><div class="wrap">',
        _sec_header(summary),
        _sec_winner(winner),
        _sec_overview(models),
        _sec_time_chart(models),
    ]

    for m, comp, pj in zip(models, comparisons, parsed_jsons):
        parts.append(_sec_model(m, comp, expected, pj))

    if len(models) == 2:
        parts.append(_sec_side_by_side(models, comparisons))

    parts.append(
        f'<p class="muted" style="text-align:center;padding:24px 0;font-size:.8rem">'
        f'Writing Benchmark System | {_e(summary.get("run_name",""))}'
        f'</p></div></body></html>'
    )
    return "\n".join(parts)


def save_html_report(
    summary: dict,
    comparisons: list[dict],
    run_name: str,
    expected: dict | None = None,
    parsed_jsons: list | None = None,
) -> Path:
    path = (ensure_dir(PROJECT_ROOT / "results" / "reports" / sanitize_run_name(run_name))
            / "benchmark_report.html")
    path.write_text(
        generate_html_report(summary, comparisons, expected, parsed_jsons),
        encoding="utf-8",
    )
    return path
