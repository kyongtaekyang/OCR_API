"""FastAPI application — Writing Benchmark Dashboard."""
import html
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

from src.health_check import run_health_check
from src.analysis_service import run_writing_analysis
from src.aws_model_storage import list_model_artifact_manifests
from src.config_loader import get_enabled_models, load_models_config
from src.json_validator import validate_output
from src.ollama_client import is_model_available
from src.benchmark_runner import run_benchmark
from src.prompt_builder import build_combined_prompt
from src.utils import PROJECT_ROOT, sanitize_run_name, timestamp, validate_image_extension

app = FastAPI(title="Writing Benchmark API", version="2.0.0")

_jobs: dict[str, dict] = {}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
JSON_EXTENSIONS = {".json"}
PROMPT_EXTENSIONS = {".txt"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _e(v) -> str:
    return html.escape(str(v)) if v is not None else ""


def _max_upload_bytes() -> int:
    return int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024


async def _save_upload(upload: UploadFile, target: Path, allowed_extensions: set[str], label: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid {label} extension: {suffix}")
    content = await upload.read()
    if len(content) > _max_upload_bytes():
        raise HTTPException(status_code=413, detail=f"{label} exceeds MAX_UPLOAD_MB")
    target.write_bytes(content)
    return target


def _parse_model_names(models: str | None) -> list[str] | None:
    if not models:
        return None
    return [m.strip() for m in models.split(",") if m.strip()]


async def _analyze_writing_impl(
    image_file, ocr_json_file, system_prompt_file, output_prompt_file,
    expected_output_file, run_name, models, inference_mode, forced_model=None,
):
    if not image_file and not ocr_json_file:
        raise HTTPException(status_code=400, detail="image_file or ocr_json_file is required")
    model_names = [forced_model] if forced_model else _parse_model_names(models)
    run_prefix = forced_model or "api"
    run_name = sanitize_run_name(run_name or f"{run_prefix}_run_{timestamp()}")
    mode = inference_mode or os.getenv("INFERENCE_MODE", "local")
    if forced_model:
        mode = "local"
    if mode not in {"local", "remote"}:
        raise HTTPException(status_code=400, detail="inference_mode must be local or remote")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        image_path = None
        ocr_path = None
        if ocr_json_file:
            ocr_path = tmp_dir / "ocr_input.json"
            await _save_upload(ocr_json_file, ocr_path, JSON_EXTENSIONS, "ocr_json_file")
        elif image_file:
            suffix = Path(image_file.filename or "image.jpg").suffix.lower()
            image_path = tmp_dir / f"image_input{suffix}"
            await _save_upload(image_file, image_path, IMAGE_EXTENSIONS, "image_file")
        sys_path = None
        if system_prompt_file:
            sys_path = tmp_dir / "system_prompt.txt"
            await _save_upload(system_prompt_file, sys_path, PROMPT_EXTENSIONS, "system_prompt_file")
        out_path = None
        if output_prompt_file:
            out_path = tmp_dir / "output_prompt.txt"
            await _save_upload(output_prompt_file, out_path, PROMPT_EXTENSIONS, "output_prompt_file")
        expected_path = None
        if expected_output_file:
            expected_path = tmp_dir / "expected_output.json"
            await _save_upload(expected_output_file, expected_path, JSON_EXTENSIONS, "expected_output_file")
        result = run_writing_analysis(
            run_name=run_name,
            image_path=str(image_path) if image_path else None,
            ocr_json_path=str(ocr_path) if ocr_path else None,
            system_prompt_path=str(sys_path) if sys_path else None,
            output_prompt_path=str(out_path) if out_path else None,
            expected_output_path=str(expected_path) if expected_path else None,
            model_names=model_names,
            inference_mode=mode,
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _latest_expected_text() -> str:
    p = PROJECT_ROOT / "data" / "expected_outputs"
    files = sorted(p.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True) if p.exists() else []
    if not files:
        return ""
    try:
        return files[0].read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_prompt_file(filename: str) -> str:
    p = PROJECT_ROOT / "data" / "prompts" / filename
    try:
        return p.read_text(encoding="utf-8") if p.exists() else ""
    except Exception:
        return ""


# ── job runner (background thread) ───────────────────────────────────────────

def _load_parsed_outputs(run_name: str, models: list) -> None:
    """Attach parsed model output JSON to each model entry in-place."""
    parsed_dir = PROJECT_ROOT / "results" / "parsed_outputs" / run_name
    if not parsed_dir.exists():
        return
    aliases: dict[str, set[str]] = {}
    try:
        for cfg in get_enabled_models():
            names = {
                cfg.get("name", ""),
                cfg.get("ollama_model", ""),
                str(cfg.get("ollama_model", "")).split(":")[0],
            }
            for name in [n for n in names if n]:
                aliases.setdefault(name, set()).update(names)
    except Exception:
        aliases = {}
    for model in models:
        base_candidates = {
            model.get("model_name", ""),
            model.get("ollama_model", ""),
            model.get("ollama_model", "").split(":")[0],
        }
        candidates = set()
        for name in [n for n in base_candidates if n]:
            candidates.add(name)
            candidates.update(aliases.get(name, set()))
        for name in candidates:
            if not name:
                continue
            pf = parsed_dir / f"{name}_parsed.json"
            if pf.exists():
                try:
                    model["parsed_output"] = json.loads(pf.read_text(encoding="utf-8"))
                except Exception:
                    pass
                break


def _run_job(job_id: str, img_path: str, sys_prompt_text: str, out_prompt_text: str,
             exp_path: str, run_name: str, tmp_dir: Path):
    stop_heartbeat = threading.Event()
    try:
        # Build combined prompt from two separate prompts
        sys_file = Path(tmp_dir) / "system_prompt.txt"
        out_file = Path(tmp_dir) / "output_prompt.txt"
        sys_file.write_text(sys_prompt_text, encoding="utf-8")
        out_file.write_text(out_prompt_text, encoding="utf-8")
        combined_text = build_combined_prompt(str(sys_file), str(out_file))
        prompt_file = Path(tmp_dir) / "combined_prompt.txt"
        prompt_file.write_text(combined_text, encoding="utf-8")

        models_cfg = get_enabled_models()
        _ctx: dict = {
            "idx": 0,
            "total": max(len(models_cfg), 1),
            "name": "",
            "started_at": None,
            "last_progress": 0.08,
        }

        def heartbeat_progress():
            while not stop_heartbeat.wait(2):
                if _jobs.get(job_id, {}).get("status") != "running":
                    return
                started_at = _ctx.get("started_at")
                name = _ctx.get("name")
                if not started_at or not name:
                    continue
                idx = _ctx["idx"]
                total = _ctx["total"]
                m_cfg = models_cfg[idx] if idx < len(models_cfg) else {}
                timeout_seconds = max(int(m_cfg.get("timeout_seconds") or 300), 1)
                elapsed = time.monotonic() - started_at
                start_frac = 0.15 + 0.70 * (idx / max(total, 1))
                end_frac = 0.15 + 0.70 * ((idx + 1) / max(total, 1))
                model_frac = min(elapsed / timeout_seconds, 0.90)
                progress = round(start_frac + model_frac * (end_frac - start_frac), 3)
                if progress > _ctx.get("last_progress", 0):
                    _ctx["last_progress"] = progress
                    _jobs[job_id].update({
                        "step": f"모델 응답 대기 중: {name} ({idx+1}/{total}) [{int(elapsed)}초 경과]",
                        "progress": progress,
                    })

        def on_progress(idx: int, total: int, model_name: str):
            _ctx["idx"] = idx
            _ctx["total"] = total
            _ctx["name"] = model_name
            _ctx["started_at"] = time.monotonic()
            progress = round(0.15 + 0.70 * (idx / max(total, 1)), 3)
            _ctx["last_progress"] = progress
            _jobs[job_id].update({
                "step": f"모델 실행 중: {model_name} ({idx+1}/{total})",
                "progress": progress,
            })

        def on_model_progress(chars_so_far: int):
            idx = _ctx["idx"]
            total = _ctx["total"]
            name = _ctx["name"]
            m_cfg = models_cfg[idx] if idx < len(models_cfg) else {}
            num_predict = (m_cfg.get("options") or {}).get("num_predict", 2048)
            model_frac = min(chars_so_far / max(num_predict * 4, 1), 0.95)
            start_frac = 0.15 + 0.70 * (idx / max(total, 1))
            end_frac = 0.15 + 0.70 * ((idx + 1) / max(total, 1))
            progress = start_frac + model_frac * (end_frac - start_frac)
            _ctx["last_progress"] = max(_ctx.get("last_progress", 0), round(progress, 3))
            _jobs[job_id].update({
                "step": f"모델 생성 중: {name} ({idx+1}/{total}) [{chars_so_far:,}자]",
                "progress": round(progress, 3),
            })

        _jobs[job_id].update({"step": "이미지 및 프롬프트 로딩 중...", "progress": 0.08})
        heartbeat_thread = threading.Thread(target=heartbeat_progress, daemon=True)
        heartbeat_thread.start()

        summary = run_benchmark(
            image_path=img_path,
            system_prompt_path="",
            output_prompt_path="",
            expected_path=exp_path,
            run_name=run_name,
            prompt_path=str(prompt_file),
            progress_callback=on_progress,
            model_progress_callback=on_model_progress,
        )

        # Attach parsed model outputs (for detailed report rendering)
        _load_parsed_outputs(run_name, summary.get("models", []))

        _jobs[job_id].update({
            "status": "complete",
            "step": "완료!",
            "progress": 1.0,
            "result": summary,
        })
    except Exception as exc:
        _jobs[job_id].update({"status": "error", "step": "오류 발생", "error": str(exc)})
    finally:
        stop_heartbeat.set()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── page HTML ─────────────────────────────────────────────────────────────────

def _run_job(job_id: str, img_path: str, sys_prompt_text: str, out_prompt_text: str,
             exp_path: str, run_name: str, tmp_dir: Path):
    try:
        sys_file = Path(tmp_dir) / "system_prompt.txt"
        out_file = Path(tmp_dir) / "output_prompt.txt"
        sys_file.write_text(sys_prompt_text, encoding="utf-8")
        out_file.write_text(out_prompt_text, encoding="utf-8")

        models_cfg = get_enabled_models()
        _jobs[job_id].update({
            "step": f"이미지를 OCR JSON으로 변환 중... 분석 모델 {len(models_cfg)}개 대기",
            "progress": 0.12,
        })

        result = run_writing_analysis(
            run_name=run_name,
            image_path=img_path,
            system_prompt_path=str(sys_file),
            output_prompt_path=str(out_file),
            expected_output_path=exp_path or None,
            inference_mode="local",
        )
        summary = result.get("benchmark_summary", {})
        api_models = {m.get("model_name"): m for m in result.get("models", [])}
        for model_summary in summary.get("models", []):
            names = {
                model_summary.get("model_name", ""),
                model_summary.get("ollama_model", ""),
                model_summary.get("ollama_model", "").split(":")[0],
            }
            api_model = next((api_models.get(name) for name in names if api_models.get(name)), None)
            if api_model:
                model_summary["parsed_output"] = api_model.get("analysis")
                model_summary["validation"] = api_model.get("validation")
                model_summary["comparison"] = api_model.get("comparison")
                model_summary["comparison_report"] = api_model.get("comparison_report")
                model_summary["json_parse_success"] = api_model.get(
                    "parse_success",
                    model_summary.get("json_parse_success", False),
                )

        _load_parsed_outputs(run_name, summary.get("models", []))
        summary["ocr_json"] = result.get("ocr_json")
        summary["artifacts"] = result.get("artifacts", {})
        if exp_path:
            try:
                summary["expected_output"] = json.loads(Path(exp_path).read_text(encoding="utf-8"))
            except Exception:
                pass

        _jobs[job_id].update({
            "status": "complete",
            "step": "완료!",
            "progress": 1.0,
            "result": summary,
        })
    except Exception as exc:
        _jobs[job_id].update({"status": "error", "step": "오류 발생", "error": str(exc)})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_page(health_badge: str, model_badges: str) -> str:
    return _PAGE_TEMPLATE.replace("%%HEALTH%%", health_badge).replace("%%MODELS%%", model_badges)


_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Writing Benchmark</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --blue:#2563eb;--dark:#1d4ed8;--green:#16a34a;--red:#dc2626;--amber:#d97706;
  --bg:#f1f5f9;--card:#fff;--border:#e2e8f0;--text:#1e293b;--muted:#64748b;--r:10px
}
body{font-family:Segoe UI,system-ui,Arial,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
a{color:var(--blue);text-decoration:none}

/* header */
.app-header{position:sticky;top:0;z-index:100;background:#fff;border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:10px 28px;gap:12px;box-shadow:0 1px 4px #0001}
.header-title{font-size:1.1rem;font-weight:700}
.header-right{margin-left:auto;display:flex;align-items:center;gap:8px;flex-wrap:wrap}

/* layout */
main{max-width:1440px;margin:0 auto;padding:24px 24px 80px}
.section{margin-bottom:20px}
.sec-title{font-size:.78rem;font-weight:700;color:var(--muted);text-transform:uppercase;
  letter-spacing:.08em;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.sec-title::after{content:'';flex:1;height:1px;background:var(--border)}

/* cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:18px}
.card-title{font-size:.88rem;font-weight:700;margin-bottom:12px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}

/* 3-col input grid */
.input-grid{display:grid;grid-template-columns:1fr 1.5fr 1.5fr;gap:14px;margin-bottom:14px}
@media(max-width:1024px){.input-grid{grid-template-columns:1fr 1fr}}
@media(max-width:640px){.input-grid{grid-template-columns:1fr}}

/* drop zone */
.drop-zone{border:2px dashed var(--border);border-radius:8px;min-height:200px;
  display:flex;align-items:center;justify-content:center;cursor:pointer;
  position:relative;transition:border-color .2s,background .2s;background:#fafafa}
.drop-zone:hover,.drop-zone.drag-over{border-color:var(--blue);background:#eff6ff}
.drop-hint{text-align:center;color:var(--muted);pointer-events:none;padding:16px}
.drop-icon{font-size:2rem;margin-bottom:6px}
.drop-hint p{font-size:.8rem;margin-top:3px}
#img-preview{max-width:100%;max-height:200px;border-radius:6px;display:none;object-fit:contain}
.img-filename{font-size:.75rem;color:var(--muted);margin-top:5px;word-break:break-all}

/* textarea */
.ta{width:100%;border:1px solid var(--border);border-radius:6px;padding:9px;
  font-family:ui-monospace,monospace;font-size:.78rem;line-height:1.5;resize:vertical;
  min-height:200px;color:var(--text);transition:border-color .2s}
.ta:focus{outline:none;border-color:var(--blue)}
.ta-sm{min-height:120px}
.card-actions{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}

/* buttons */
.btn{display:inline-flex;align-items:center;gap:5px;border:none;border-radius:6px;
  padding:7px 13px;font-size:.83rem;font-weight:600;cursor:pointer;white-space:nowrap;transition:background .15s,opacity .15s}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-primary{background:var(--blue);color:#fff}
.btn-primary:hover:not(:disabled){background:var(--dark)}
.btn-outline{background:#fff;color:var(--text);border:1px solid var(--border)}
.btn-outline:hover:not(:disabled){background:#f8fafc}
.btn-ghost{background:transparent;color:var(--muted)}
.btn-ghost:hover{color:var(--text)}
.btn-lg{padding:10px 26px;font-size:.95rem}
.text-input{border:1px solid var(--border);border-radius:6px;padding:7px 11px;
  font-size:.85rem;color:var(--text);outline:none;transition:border-color .2s}
.text-input:focus{border-color:var(--blue)}

/* badge */
.badge{display:inline-block;border-radius:999px;padding:3px 9px;font-size:.72rem;font-weight:700}
.b-ok{background:#dcfce7;color:#15803d}
.b-err{background:#fee2e2;color:#b91c1c}
.b-info{background:#dbeafe;color:#1d4ed8}
.b-warn{background:#fef9c3;color:#92400e}

/* run bar */
.run-bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}

/* progress */
.progress-wrap{background:#f8fafc;border:1px solid var(--border);border-radius:8px;
  padding:14px;margin-bottom:12px;display:none}
.progress-header{display:flex;justify-content:space-between;margin-bottom:7px;font-size:.85rem}
.progress-step{font-weight:500}
.progress-pct{color:var(--blue);font-weight:700}
.progress-outer{height:9px;background:#e2e8f0;border-radius:999px;overflow:hidden}
.progress-inner{height:100%;background:linear-gradient(90deg,var(--blue),#7c3aed);
  border-radius:999px;transition:width .5s ease;width:0%}

/* error panel */
.err-panel{background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;
  padding:14px;margin-bottom:12px;display:none}
.err-hdr{font-weight:700;color:var(--red);margin-bottom:6px;font-size:.88rem}
.err-body{font-family:monospace;font-size:.8rem;color:#7f1d1d;white-space:pre-wrap;word-break:break-all}

/* ── Result sections ── */
.report-block{margin-bottom:24px}
.report-header-bar{background:linear-gradient(135deg,#1e3a8a 0%,#3730a3 100%);
  color:#fff;border-radius:var(--r) var(--r) 0 0;padding:14px 20px;
  display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.report-body{background:var(--card);border:1px solid var(--border);border-top:none;
  border-radius:0 0 var(--r) var(--r);padding:20px}
.report-model-name{font-size:1rem;font-weight:700}
.report-meta{font-size:.78rem;opacity:.8;margin-left:auto}

/* meta chips */
.meta-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.meta-chip{background:#f8fafc;border:1px solid var(--border);border-radius:6px;
  padding:6px 10px;font-size:.8rem}
.meta-chip .ml{font-size:.68rem;color:var(--muted);font-weight:700;text-transform:uppercase;display:block;margin-bottom:2px}

/* writing boxes */
.writing-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
@media(max-width:700px){.writing-grid{grid-template-columns:1fr}}
.wbox{border-radius:8px;padding:14px;border:1px solid var(--border)}
.wbox-orig{background:#fafafa}
.wbox-corr{background:#eff6ff;border-color:#93c5fd}
.wbox-label{font-size:.7rem;font-weight:700;text-transform:uppercase;color:var(--muted);
  letter-spacing:.05em;margin-bottom:8px}
.wbox-text{font-size:.88rem;line-height:1.8;white-space:pre-wrap;word-break:break-word}
.tag-G{background:#fee2e2;color:#991b1b;border-radius:3px;padding:1px 3px;font-weight:700;font-size:.85em}
.tag-V{background:#fef3c7;color:#92400e;border-radius:3px;padding:1px 3px;font-weight:700;font-size:.85em}
.tag-S{background:#fce7f3;color:#9d174d;border-radius:3px;padding:1px 3px;font-weight:700;font-size:.85em}
.tag-O{background:#ede9fe;color:#5b21b6;border-radius:3px;padding:1px 3px;font-weight:700;font-size:.85em}
.tag-P{background:#e0f2fe;color:#075985;border-radius:3px;padding:1px 3px;font-weight:700;font-size:.85em}
.tag-C{background:#d1fae5;color:#065f46;border-radius:3px;padding:1px 3px;font-weight:700;font-size:.85em}

/* analysis grid */
.analysis-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
@media(max-width:700px){.analysis-grid{grid-template-columns:1fr}}
.block-label{font-size:.72rem;font-weight:700;color:var(--muted);text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:10px}

/* error count table */
.err-tbl{width:100%;border-collapse:collapse;font-size:.83rem}
.err-tbl th{background:#f8fafc;padding:7px 10px;text-align:left;font-size:.72rem;
  font-weight:700;color:var(--muted);text-transform:uppercase;border-bottom:2px solid var(--border)}
.err-tbl td{padding:7px 10px;border-bottom:1px solid var(--border)}
.err-tbl tr.total-row td{font-weight:700;background:#f8fafc}

/* score bars */
.score-item{margin-bottom:10px}
.score-row{display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:4px}
.score-name{font-weight:600}
.score-val{font-weight:700}
.bar-outer{height:8px;background:#e2e8f0;border-radius:999px;overflow:hidden}
.bar-inner{height:100%;border-radius:999px;transition:width .7s ease}
.total-score-wrap{margin-top:12px;padding-top:10px;border-top:1px solid var(--border)}
.total-score-num{font-size:1.6rem;font-weight:700;text-align:center;margin-bottom:4px}

/* error explanations */
.exp-list{display:flex;flex-direction:column;gap:8px;margin-bottom:14px}
.exp-item{background:#fafafa;border:1px solid var(--border);border-radius:7px;padding:10px 12px;font-size:.83rem;line-height:1.6}
.exp-word{font-weight:700;color:var(--red)}
.exp-en{color:var(--text)}
.exp-ko{color:var(--muted);font-size:.78rem;margin-top:2px}

/* overall comment */
.comment-box{background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:14px;
  font-size:.88rem;line-height:1.7}
.comment-label{font-size:.7rem;font-weight:700;color:#15803d;text-transform:uppercase;margin-bottom:6px}

/* ── Comparison report ── */
/* JSON outputs */
.json-stack{display:flex;flex-direction:column;gap:14px;margin-bottom:18px}
.json-panel{background:#fff;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.json-panel-h{display:flex;align-items:center;gap:8px;justify-content:space-between;
  padding:10px 14px;background:#f8fafc;border-bottom:1px solid var(--border);font-weight:700;font-size:.86rem}
.json-panel pre{margin:0;padding:14px;max-height:360px;overflow:auto;background:#0f172a;color:#e2e8f0;
  font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre}
.json-compare{padding:12px 14px;border-top:1px solid var(--border);background:#fff}
.json-compare-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-bottom:10px}
.metric-tile{border:1px solid var(--border);border-radius:7px;padding:8px 10px;background:#fafafa}
.metric-tile .k{display:block;font-size:.68rem;color:var(--muted);font-weight:700;text-transform:uppercase;margin-bottom:3px}
.metric-tile .v{font-weight:800}

.winner-banner{background:linear-gradient(135deg,#1e3a8a,#4338ca);color:#fff;
  border-radius:var(--r);padding:20px 24px;margin-bottom:14px}
.winner-label{font-size:.7rem;font-weight:700;letter-spacing:.1em;opacity:.7;
  text-transform:uppercase;margin-bottom:10px}
.winner-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.winner-item .wl{font-size:.68rem;opacity:.72;text-transform:uppercase;margin-bottom:3px}
.winner-item .wv{font-size:.95rem;font-weight:700}

.cmp-table{width:100%;border-collapse:collapse;font-size:.85rem}
.cmp-table th{background:#f8fafc;padding:9px 12px;text-align:left;font-size:.75rem;
  font-weight:700;color:var(--muted);text-transform:uppercase;border-bottom:2px solid var(--border)}
.cmp-table td{padding:9px 12px;border-bottom:1px solid var(--border);vertical-align:middle}
.cmp-table tr:hover td{background:#fafafa}
.diff-up{color:var(--red);font-weight:700}
.diff-dn{color:var(--green);font-weight:700}
</style>
</head>
<body>

<header class="app-header">
  <span style="font-size:1.2rem">✍</span>
  <span class="header-title">Writing Benchmark</span>
  <div class="header-right">
    %%HEALTH%%
    %%MODELS%%
    <a href="/test-page" class="btn btn-ghost" style="font-size:.8rem">API 점검</a>
    <a href="/docs" class="btn btn-ghost" style="font-size:.8rem">API 문서</a>
  </div>
</header>

<main>

<!-- ═══ 1. 입력 설정 ═══ -->
<div class="section">
  <div class="sec-title">입력 설정</div>

  <div class="input-grid">
    <!-- 이미지 -->
    <div class="card">
      <div class="card-title">🖼️ 이미지</div>
      <div id="drop-zone" class="drop-zone" role="button" aria-label="이미지 선택">
        <input type="file" id="image-input" accept="image/*" style="display:none">
        <img id="img-preview">
        <div id="drop-hint" class="drop-hint">
          <div class="drop-icon">📁</div>
          <p><strong>클릭</strong>하거나 드래그</p>
          <p>JPG · PNG · WEBP</p>
        </div>
      </div>
      <div id="img-filename" class="img-filename"></div>
    </div>

    <!-- 시스템 프롬프트 -->
    <div class="card">
      <div class="card-title">📝 시스템 프롬프트</div>
      <textarea id="sys-prompt" class="ta" rows="10" placeholder="시스템 역할 및 지시사항을 입력하세요..."></textarea>
      <div class="card-actions">
        <button class="btn btn-outline" onclick="loadSysPrompt()">서버에서 불러오기</button>
        <button class="btn btn-ghost" onclick="document.getElementById('sys-prompt').value=''">초기화</button>
      </div>
    </div>

    <!-- 출력 프롬프트 -->
    <div class="card">
      <div class="card-title">📋 출력 프롬프트</div>
      <textarea id="out-prompt" class="ta" rows="10" placeholder="출력 형식 및 스키마를 입력하세요..."></textarea>
      <div class="card-actions">
        <button class="btn btn-outline" onclick="loadOutPrompt()">서버에서 불러오기</button>
        <button class="btn btn-ghost" onclick="document.getElementById('out-prompt').value=''">초기화</button>
      </div>
    </div>
  </div>

  <!-- 정답 분석지 -->
  <div class="card">
    <div class="card-title">📊 정답 분석지 <span class="badge b-info">선택 — 비교 보고서 생성에 사용</span></div>
    <textarea id="expected" class="ta ta-sm" placeholder="정답 분석지 JSON을 입력하거나 파일에서 불러오세요. 입력하지 않으면 각 모델 보고서만 표시됩니다."></textarea>
    <div class="card-actions">
      <label class="btn btn-outline" for="expected-file" style="cursor:pointer">📂 파일 선택</label>
      <input type="file" id="expected-file" accept=".json" style="display:none">
      <button class="btn btn-outline" onclick="loadExpected()">서버에서 불러오기</button>
      <button class="btn btn-ghost" onclick="document.getElementById('expected').value=''">초기화</button>
    </div>
  </div>
</div>

<!-- ═══ 2. 실행 ═══ -->
<div class="section">
  <div class="sec-title">실행</div>
  <div class="run-bar">
    <input id="run-name" type="text" class="text-input" style="width:220px" placeholder="실행 이름 (선택)">
    <button id="run-btn" class="btn btn-primary btn-lg" onclick="startBenchmark()">▶ 벤치마크 실행</button>
  </div>
  <div id="progress-wrap" class="progress-wrap">
    <div class="progress-header">
      <span id="progress-step" class="progress-step">준비 중...</span>
      <span id="progress-pct" class="progress-pct">0%</span>
    </div>
    <div class="progress-outer"><div id="progress-bar" class="progress-inner"></div></div>
  </div>
  <div id="err-panel" class="err-panel">
    <div class="err-hdr">⚠ 오류</div>
    <pre id="err-body" class="err-body"></pre>
  </div>
</div>

<!-- ═══ 3. 결과 ═══ -->
<div id="results-root" style="display:none">
  <div class="section">
    <div class="sec-title">분석 결과</div>
    <div id="json-results"></div>
    <div id="model-reports"></div>
  </div>
  <div id="cmp-section" class="section" style="display:none">
    <div class="sec-title">비교 보고서</div>
    <div id="cmp-report"></div>
  </div>
</div>

</main>

<script>
'use strict';
const S = { imageFile: null, pollTimer: null };

document.addEventListener('DOMContentLoaded', () => {
  setupDrop();
  setupExpFile();
  loadSysPrompt();
  loadOutPrompt();
  loadExpected();
});

/* ── image drop ── */
function setupDrop() {
  const zone = document.getElementById('drop-zone');
  zone.addEventListener('click', () => document.getElementById('image-input').click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag-over'); if (e.dataTransfer.files[0]) setImg(e.dataTransfer.files[0]); });
  document.getElementById('image-input').addEventListener('change', e => { if (e.target.files[0]) setImg(e.target.files[0]); });
}
function setImg(file) {
  S.imageFile = file;
  const r = new FileReader();
  r.onload = ev => {
    const p = document.getElementById('img-preview');
    p.src = ev.target.result; p.style.display = 'block';
    document.getElementById('drop-hint').style.display = 'none';
    document.getElementById('img-filename').textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
  };
  r.readAsDataURL(file);
}

/* ── prompts ── */
async function loadSysPrompt() {
  try { const { content } = await fetch('/api/sys-prompt').then(r => r.json()); if (content) document.getElementById('sys-prompt').value = content; } catch (_) {}
}
async function loadOutPrompt() {
  try { const { content } = await fetch('/api/out-prompt').then(r => r.json()); if (content) document.getElementById('out-prompt').value = content; } catch (_) {}
}

/* ── expected ── */
function setupExpFile() {
  document.getElementById('expected-file').addEventListener('change', e => {
    const f = e.target.files[0]; if (!f) return;
    const r = new FileReader();
    r.onload = ev => { document.getElementById('expected').value = ev.target.result; };
    r.readAsText(f, 'utf-8');
  });
}
async function loadExpected() {
  try { const { expected } = await fetch('/api/expected').then(r => r.json()); if (expected) document.getElementById('expected').value = expected; } catch (_) {}
}

/* ── benchmark ── */
async function startBenchmark() {
  if (!S.imageFile) { showErr('이미지를 선택해주세요.'); return; }
  const sysP = document.getElementById('sys-prompt').value.trim();
  const outP = document.getElementById('out-prompt').value.trim();
  if (!sysP) { showErr('시스템 프롬프트를 입력해주세요.'); return; }
  if (!outP) { showErr('출력 프롬프트를 입력해주세요.'); return; }
  const exp = document.getElementById('expected').value.trim();
  if (exp) { try { JSON.parse(exp); } catch (e) { showErr('정답 분석지 JSON 형식 오류: ' + e.message); return; } }

  hideErr();
  document.getElementById('results-root').style.display = 'none';
  setRunning(true);
  showProg(0.04, '요청 전송 중...');

  const fd = new FormData();
  fd.append('image', S.imageFile);
  fd.append('system_prompt', sysP);
  fd.append('output_prompt', outP);
  if (exp) fd.append('expected', exp);
  const rn = document.getElementById('run-name').value.trim();
  if (rn) fd.append('run_name', rn);

  try {
    const r = await fetch('/api/run', { method: 'POST', body: fd });
    if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || 'HTTP ' + r.status); }
    const { job_id } = await r.json();
    poll(job_id);
  } catch (e) { showErr(e.message); setRunning(false); }
}

function poll(jobId) {
  if (S.pollTimer) clearInterval(S.pollTimer);
  S.pollTimer = setInterval(async () => {
    try {
      const r = await fetch('/api/run/' + jobId);
      if (!r.ok) throw new Error('poll ' + r.status);
      const job = await r.json();
      showProg(job.progress || 0, job.step || '처리 중...');
      if (job.status === 'complete') {
        clearInterval(S.pollTimer); setRunning(false); showProg(1.0, '✓ 완료!');
        renderResults(job.result);
      } else if (job.status === 'error') {
        clearInterval(S.pollTimer); setRunning(false); showErr(job.error || '알 수 없는 오류');
      }
    } catch (e) { clearInterval(S.pollTimer); setRunning(false); showErr('연결 오류: ' + e.message); }
  }, 2000);
}

/* ── UI helpers ── */
function setRunning(on) {
  const b = document.getElementById('run-btn');
  b.disabled = on;
  b.textContent = on ? '⏳ 실행 중...' : '▶ 벤치마크 실행';
}
function showProg(frac, text) {
  const w = document.getElementById('progress-wrap'); w.style.display = 'block';
  const p = Math.min(100, Math.round(frac * 100));
  document.getElementById('progress-bar').style.width = p + '%';
  document.getElementById('progress-step').textContent = text;
  document.getElementById('progress-pct').textContent = p + '%';
}
function showErr(msg) { const p = document.getElementById('err-panel'); p.style.display = 'block'; document.getElementById('err-body').textContent = msg; p.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
function hideErr() { document.getElementById('err-panel').style.display = 'none'; }
function esc(s) { if (s == null) return '—'; return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function fmt(v, d = 4) { return v == null ? '—' : Number(v).toFixed(d); }
function pct(v) { return v == null ? '—' : Math.round(v * 100) + '%'; }
function scol(v) { if (v == null) return '#64748b'; const p = v * 100; return p >= 80 ? '#16a34a' : p >= 60 ? '#d97706' : '#dc2626'; }

/* ════════════════════════════════════════
   결과 렌더링
   ════════════════════════════════════════ */
function renderResults(summary) {
  const models = summary.models || [];
  const winner = summary.winner || {};
  const hasExpected = models.some(m => m.overall_accuracy_score != null && m.detail_diff);

  /* 모델별 보고서 */
  document.getElementById('json-results').innerHTML = buildJsonResults(summary, models);
  const reportsDiv = document.getElementById('model-reports');
  reportsDiv.innerHTML = models.map(m => buildModelReport(m)).join('');

  /* 비교 보고서 */
  const cmpSection = document.getElementById('cmp-section');
  if (hasExpected) {
    cmpSection.style.display = 'block';
    document.getElementById('cmp-report').innerHTML = buildCmpReport(models, winner);
  } else {
    cmpSection.style.display = 'none';
  }

  document.getElementById('results-root').style.display = 'block';
  document.getElementById('results-root').scrollIntoView({ behavior: 'smooth' });
}

/* ─────────────────────────────────────
   모델 분석 보고서
   ───────────────────────────────────── */
function prettyJson(obj) {
  if (obj == null) return 'null';
  try { return JSON.stringify(obj, null, 2); } catch (_) { return String(obj); }
}
function jsonPanel(title, obj, meta = '') {
  return `<div class="json-panel">
    <div class="json-panel-h"><span>${esc(title)}</span><span class="badge b-info">${esc(meta)}</span></div>
    <pre>${esc(prettyJson(obj))}</pre>
  </div>`;
}
function metricTile(label, value, colorValue = null) {
  const col = colorValue == null ? 'var(--text)' : scol(colorValue);
  return `<div class="metric-tile"><span class="k">${esc(label)}</span><span class="v" style="color:${col}">${esc(value)}</span></div>`;
}
function buildModelJsonCompare(m) {
  const c = m.comparison || {};
  const dd = m.detail_diff || c.detail_diff || {};
  const ea = c.error_analysis_category_difference || m.error_analysis_category_difference || {};
  let h = `<div class="json-compare">
    <div class="block-label">제공 정답 JSON vs 산출 JSON 비교</div>
    <div class="json-compare-grid">
      ${metricTile('JSON 파싱', (m.json_parse_success || c.json_parse_success) ? '성공' : '실패', (m.json_parse_success || c.json_parse_success) ? 1 : 0)}
      ${metricTile('스키마 일치도', fmt(m.schema_compliance_score ?? c.schema_compliance_score), m.schema_compliance_score ?? c.schema_compliance_score)}
      ${metricTile('전체 정확도', fmt(m.overall_accuracy_score ?? c.overall_accuracy_score), m.overall_accuracy_score ?? c.overall_accuracy_score)}
      ${metricTile('원문 유사도', fmt(m.original_writing_similarity ?? c.original_writing_similarity), m.original_writing_similarity ?? c.original_writing_similarity)}
      ${metricTile('교정문 유사도', fmt(m.corrected_writing_similarity ?? c.corrected_writing_similarity), m.corrected_writing_similarity ?? c.corrected_writing_similarity)}
      ${metricTile('오류 태그 F1', fmt(m.error_tag_f1 ?? c.error_tag_f1), m.error_tag_f1 ?? c.error_tag_f1)}
      ${metricTile('총점 차이', m.total_score_difference ?? c.total_score_difference ?? '—')}
      ${metricTile('오류 개수 차이', c.total_errors_difference ?? m.total_errors_difference ?? '—')}
    </div>`;
  const scoreDiffs = dd.writing_performance_diffs || [];
  if (scoreDiffs.length) {
    h += `<table class="cmp-table"><thead><tr><th>항목</th><th>정답</th><th>산출</th><th>차이</th><th>상태</th></tr></thead><tbody>`;
    scoreDiffs.forEach(x => {
      h += `<tr><td>${esc(x.domain)}</td><td>${esc(x.expected_score ?? '—')}</td><td>${esc(x.actual_score ?? '—')}</td><td>${esc(x.delta ?? '—')}</td><td>${esc(x.status ?? '')}</td></tr>`;
    });
    h += `</tbody></table>`;
  }
  const fieldDiffs = (dd.field_diffs || []).filter(x => x.status !== 'match').slice(0, 30);
  if (fieldDiffs.length) {
    h += `<div class="block-label" style="margin-top:12px">항목별 차이</div>
      <table class="cmp-table"><thead><tr><th>JSON 경로</th><th>정답</th><th>산출</th><th>유사도</th><th>상태</th></tr></thead><tbody>`;
    fieldDiffs.forEach(x => {
      h += `<tr>
        <td><code>${esc(x.path)}</code></td>
        <td>${esc(String(x.expected_value ?? '—')).slice(0, 180)}</td>
        <td>${esc(String(x.actual_value ?? '—')).slice(0, 180)}</td>
        <td>${esc(x.similarity ?? '—')}</td>
        <td>${esc(x.status ?? '')}</td>
      </tr>`;
    });
    h += `</tbody></table>`;
  }
  if (Object.keys(ea).length) {
    h += `<div class="block-label" style="margin-top:12px">오류 유형별 개수 차이</div>
      <table class="cmp-table"><thead><tr><th>오류 유형</th><th>정답</th><th>산출</th><th>차이</th><th>일치</th></tr></thead><tbody>`;
    Object.entries(ea).forEach(([k, v]) => h += `<tr><td>${esc(k)}</td><td>${esc(v.expected ?? '—')}</td><td>${esc(v.actual ?? '—')}</td><td>${esc(v.diff ?? v)}</td><td>${esc(v.match ?? '')}</td></tr>`);
    h += `</tbody></table>`;
  }
  h += `</div>`;
  return h;
}
function buildJsonResults(summary, models) {
  const expected = summary.expected_output || null;
  let h = `<div class="json-stack">`;
  if (summary.ocr_json) h += jsonPanel('1. docTR OCR 결과 JSON', summary.ocr_json, '이미지 텍스트 전환');
  if (expected) h += jsonPanel('2. 제공 정답 분석 JSON', expected, '기준 JSON');
  models.forEach((m, idx) => {
    const output = m.parsed_output || m.analysis || null;
    h += `<div class="json-panel">
      <div class="json-panel-h"><span>${idx + 3}. ${esc(m.model_name || m.ollama_model)} 산출 분석 JSON</span><span class="badge ${m.success ? 'b-ok' : 'b-err'}">${m.success ? '생성 성공' : '생성 실패'}</span></div>
      <pre>${esc(prettyJson(output))}</pre>
      ${expected ? buildModelJsonCompare(m) : ''}
    </div>`;
  });
  h += `</div>`;
  return h;
}

function numericScore(v) {
  if (v == null || v === '' || v === 'N/A') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
function scoreValue(obj) {
  if (obj && typeof obj === 'object') return numericScore(obj.score);
  return numericScore(obj);
}
function maxValue(obj) {
  if (obj && typeof obj === 'object') return numericScore(obj.max_score);
  return null;
}
function summarizeDomain(scoring, perf, domain, defaultMax) {
  const perfItem = perf?.[domain] || {};
  const perfScore = scoreValue(perfItem);
  const perfMax = maxValue(perfItem) || defaultMax;
  if (perfScore != null) return { score: perfScore, max: perfMax };
  const domainObj = scoring?.[domain] || {};
  const subtotal = domainObj?.subtotal || {};
  const subtotalScore = scoreValue(subtotal);
  if (subtotalScore != null) return { score: subtotalScore, max: maxValue(subtotal) || defaultMax };
  let score = 0;
  let max = 0;
  Object.entries(domainObj || {}).forEach(([key, value]) => {
    if (key === 'subtotal') return;
    const s = scoreValue(value);
    if (s == null) return;
    score += s;
    max += maxValue(value) || 0;
  });
  return score || max ? { score, max: max || defaultMax } : { score: null, max: defaultMax };
}
function normalizeScoreSummary(p) {
  const scoring = p.scoring_analysis || {};
  const perf = p.writing_performance || p.writing_analysis?.writing_performance || {};
  const grammar = summarizeDomain(scoring, perf, 'grammar', 30);
  const vocabulary = summarizeDomain(scoring, perf, 'vocabulary', 30);
  const writing_flow = summarizeDomain(scoring, perf, 'writing_flow', 40);
  const totalPerf = perf?.total_score || perf?.overall_score;
  let totalScore = scoreValue(totalPerf);
  let totalMax = maxValue(totalPerf) || 100;
  if (totalScore == null && [grammar, vocabulary, writing_flow].every(x => x.score != null)) {
    totalScore = grammar.score + vocabulary.score + writing_flow.score;
    totalMax = grammar.max + vocabulary.max + writing_flow.max;
  }
  return { grammar, vocabulary, writing_flow, total_score: { score: totalScore, max: totalMax } };
}
function normalizeErrorAnalysis(p) {
  const raw = p.error_analysis || {};
  const counts = {
    grammar: numericScore(raw.grammar) || 0,
    vocabulary: numericScore(raw.vocabulary) || 0,
    word_order: numericScore(raw.word_order) || 0,
    spelling: numericScore(raw.spelling) || 0,
    punctuation: numericScore(raw.punctuation) || 0,
    coherence: numericScore(raw.coherence) || 0,
  };
  const hasRawCounts = ['grammar', 'vocabulary', 'word_order', 'spelling', 'punctuation', 'coherence']
    .some(k => raw[k] != null && numericScore(raw[k]) != null);
  const tagMap = { G: 'grammar', V: 'vocabulary', O: 'word_order', S: 'spelling', P: 'punctuation', C: 'coherence' };
  if (!hasRawCounts) {
    for (const match of String(p.original_writing || '').matchAll(/\[(G|V|O|S|P|C)\]/g)) counts[tagMap[match[1]]]++;
    (p.error_explanations || []).forEach(e => {
      const tag = String(e.tag || e.type || e.category || '').toUpperCase().charAt(0);
      if (tagMap[tag]) counts[tagMap[tag]]++;
    });
  }
  const rawTotal = numericScore(raw.total_errors);
  counts.total_errors = rawTotal != null ? rawTotal : Object.values(counts).reduce((a, b) => a + b, 0);
  Object.keys(counts).forEach(k => { if (counts[k] === 0 && raw[k] == null) counts[k] = null; });
  return counts;
}

function buildModelReport(m) {
  const p = m.parsed_output || m.analysis || {};
  const meta = p.metadata || {};
  const ea = normalizeErrorAnalysis(p);
  const scoreSummary = normalizeScoreSummary(p);
  const exps = p.error_explanations || [];
  const suc = m.success;
  const mname = esc(m.model_name || m.ollama_model);

  let h = `<div class="report-block">`;

  /* 헤더 바 */
  h += `<div class="report-header-bar">
    <span class="badge ${suc ? 'b-ok' : 'b-err'}">${suc ? '✓ 성공' : '✗ 실패'}</span>
    <span class="report-model-name">${mname} 분석 보고서</span>
    <span class="report-meta">처리 ${fmt(m.duration_seconds, 1)}초 &nbsp;|&nbsp; JSON 파싱 ${m.json_parse_success ? '✓' : '✗'}</span>
  </div>`;

  h += `<div class="report-body">`;

  if (!suc && !p.original_writing) {
    h += `<p style="color:var(--red)">${esc(m.error || '모델 실행 실패')}</p></div></div>`;
    return h;
  }

  /* 메타데이터 */
  const metaFields = [
    ['제목', meta.title], ['수정 제목', meta.title_corrected],
    ['클래스', meta.class], ['글 유형', meta.writing_type],
    ['주제', meta.topic], ['수업 유형', meta.course_type],
  ];
  h += `<div class="meta-row">`;
  metaFields.forEach(([l, v]) => {
    if (v) h += `<div class="meta-chip"><span class="ml">${l}</span>${esc(v)}</div>`;
  });
  h += `</div>`;

  /* 원문 / 교정문 */
  h += `<div class="writing-grid">
    <div class="wbox wbox-orig">
      <div class="wbox-label">원문 (Original Writing)</div>
      <div class="wbox-text">${highlightTags(p.original_writing)}</div>
    </div>
    <div class="wbox wbox-corr">
      <div class="wbox-label">교정문 (Corrected Writing)</div>
      <div class="wbox-text">${esc(p.corrected_writing || '—')}</div>
    </div>
  </div>`;

  /* 오류 분석 + 채점 */
  h += `<div class="analysis-grid">`;

  /* 오류 유형별 개수 */
  h += `<div>
    <div class="block-label">오류 유형별 개수</div>
    <table class="err-tbl">
      <thead><tr><th>오류 유형</th><th>개수</th></tr></thead>
      <tbody>
        ${erow('문법 (Grammar)', ea.grammar)}
        ${erow('어휘 (Vocabulary)', ea.vocabulary)}
        ${erow('어순 (Word Order)', ea.word_order)}
        ${erow('철자 (Spelling)', ea.spelling)}
        ${erow('구두점 (Punctuation)', ea.punctuation)}
        ${erow('일관성 (Coherence)', ea.coherence)}
        ${erow('합계', ea.total_errors, true)}
      </tbody>
    </table>
  </div>`;

  /* 도메인별 채점 */
  const scoreMap = {
    '문법 (Grammar)':    { score: scoreSummary.grammar.score,      max: scoreSummary.grammar.max },
    '어휘 (Vocabulary)': { score: scoreSummary.vocabulary.score,   max: scoreSummary.vocabulary.max },
    '글 흐름 (Flow)':    { score: scoreSummary.writing_flow.score, max: scoreSummary.writing_flow.max },
  };
  const totalScore = scoreSummary.total_score.score;
  const totalMax = scoreSummary.total_score.max;
  h += `<div>
    <div class="block-label">채점 결과</div>`;
  Object.entries(scoreMap).forEach(([label, { score, max }]) => {
    const bw = score != null ? Math.min(100, Math.round(score / max * 100)) : 0;
    const col = scol(score != null ? score / max : null);
    h += `<div class="score-item">
      <div class="score-row">
        <span class="score-name">${label}</span>
        <span class="score-val" style="color:${col}">${score != null ? score : '—'} / ${max}</span>
      </div>
      <div class="bar-outer"><div class="bar-inner" style="width:${bw}%;background:${col}"></div></div>
    </div>`;
  });
  const tcol = scol(totalScore != null ? totalScore / totalMax : null);
  h += `<div class="total-score-wrap">
    <div class="block-label">총점</div>
    <div class="total-score-num" style="color:${tcol}">${totalScore != null ? totalScore : '—'} <span style="font-size:1rem;color:var(--muted)">/ ${totalMax}</span></div>
    <div class="bar-outer" style="height:12px"><div class="bar-inner" style="width:${totalScore != null ? Math.min(100, Math.round(totalScore / totalMax * 100)) : 0}%;background:${tcol}"></div></div>
  </div>`;
  h += `</div>`;  /* end score col */

  h += `</div>`;  /* end analysis-grid */

  /* 오류 상세 설명 */
  if (exps.length) {
    h += `<div class="block-label" style="margin-bottom:8px">오류 상세 설명</div>
    <div class="exp-list">`;
    exps.forEach((e, i) => {
      h += `<div class="exp-item">
        <span class="exp-word">${i + 1}. "${esc(e.error)}"</span>
        &nbsp;→&nbsp;
        <span class="exp-en">${esc(e.explanation_en || e.explanation || '')}</span>
        ${e.explanation && e.explanation !== e.explanation_en ? `<div class="exp-ko">${esc(e.explanation)}</div>` : ''}
      </div>`;
    });
    h += `</div>`;
  }

  /* 전체 코멘트 */
  if (p.overall_comments) {
    h += `<div class="comment-box">
      <div class="comment-label">전체 코멘트</div>
      ${esc(p.overall_comments)}
    </div>`;
  }

  h += `</div></div>`;  /* report-body, report-block */
  return h;
}

function highlightTags(text) {
  if (!text) return '—';
  return esc(text).replace(/\[(G|V|S|O|P|C)\]/g, (_, t) => `<span class="tag-${t}">[${t}]</span>`);
}
function erow(label, val, total = false) {
  const cls = total ? ' class="total-row"' : '';
  return `<tr${cls}><td>${label}</td><td>${val != null ? val : '—'}</td></tr>`;
}
function getExpScore(model, domain) {
  return (model?.detail_diff?.writing_performance_diffs || []).find(x => x.domain === domain)?.expected_score ?? null;
}

/* ─────────────────────────────────────
   비교 보고서
   ───────────────────────────────────── */
function buildCmpReport(models, winner) {
  let h = '';

  /* 종합 평가 배너 */
  h += `<div class="winner-banner">
    <div class="winner-label">종합 평가</div>
    <div class="winner-grid">
      ${witem('🏆 종합 1위', winner.best_overall_model)}
      ${witem('🎯 정확도 1위', winner.best_accuracy_model)}
      ${witem('⚡ 속도 1위', winner.fastest_model)}
      ${witem('🔌 API 최적', winner.best_api_candidate)}
    </div>
  </div>`;

  /* 점수 비교표 */
  h += `<div class="card" style="margin-bottom:14px">
    <div class="card-title">📐 점수 비교 (정답지 vs 모델)</div>
    <div style="overflow-x:auto"><table class="cmp-table"><thead><tr>
      <th>항목</th><th>정답지</th>`;
  models.forEach(m => h += `<th>${esc(m.model_name || m.ollama_model)}</th>`);
  h += `</tr></thead><tbody>`;

  [
    { domain: 'grammar',      label: '문법 점수',    max: 30 },
    { domain: 'vocabulary',   label: '어휘 점수',    max: 30 },
    { domain: 'writing_flow', label: '글흐름 점수',  max: 40 },
    { domain: 'total_score',  label: '총점',         max: 100 },
  ].forEach(({ domain, label, max }) => {
    const expScore = getExpScore(models[0], domain);
    h += `<tr><td><b>${label}</b> <span style="color:var(--muted)">/ ${max}</span></td>`;
    h += `<td>${expScore != null ? expScore : '—'}</td>`;
    models.forEach(m => {
      const wpd = (m.detail_diff?.writing_performance_diffs || []).find(x => x.domain === domain);
      const actual = wpd?.actual_score;
      const delta = wpd?.delta;
      let cell = actual != null ? String(actual) : '—';
      if (delta != null) {
        const cls = delta > 3 ? 'diff-up' : delta < -3 ? 'diff-dn' : '';
        cell += ` <span class="${cls}">(Δ${delta > 0 ? '+' : ''}${fmt(delta, 1)})</span>`;
      }
      h += `<td>${cell}</td>`;
    });
    h += `</tr>`;
  });
  h += `</tbody></table></div></div>`;

  /* 정확도 지표 */
  h += `<div class="card" style="margin-bottom:14px">
    <div class="card-title">📊 정확도 지표</div>
    <div style="overflow-x:auto"><table class="cmp-table"><thead><tr><th>지표</th>`;
  models.forEach(m => h += `<th>${esc(m.model_name || m.ollama_model)}</th>`);
  h += `</tr></thead><tbody>`;

  [
    { key: 'overall_accuracy_score',          label: '전체 정확도' },
    { key: 'api_readiness_score',             label: 'API 준비도' },
    { key: 'schema_compliance_score',         label: '스키마 준수' },
    { key: 'required_key_completeness',       label: '필드 완성도' },
    { key: 'error_tag_f1',                    label: '오류 태그 F1' },
    { key: 'original_writing_similarity',     label: '원문 유사도' },
    { key: 'corrected_writing_similarity',    label: '교정문 유사도' },
  ].forEach(({ key, label }) => {
    h += `<tr><td><b>${label}</b></td>`;
    models.forEach(m => {
      const v = m[key];
      const col = scol(v);
      h += `<td><span style="color:${col};font-weight:700">${v != null ? v.toFixed(4) + ' (' + pct(v) + ')' : '—'}</span></td>`;
    });
    h += `</tr>`;
  });
  h += `</tbody></table></div></div>`;

  /* 오류 태그 F1 */
  const tags = (models[0]?.detail_diff?.per_tag_f1 || []).filter(t => t.status !== 'n/a');
  if (tags.length) {
    h += `<div class="card">
      <div class="card-title">🏷 오류 유형별 F1</div>
      <div style="overflow-x:auto"><table class="cmp-table"><thead><tr>
        <th>유형</th><th>정답 수</th>`;
    models.forEach(m => h += `<th>${esc(m.model_name || m.ollama_model)}</th>`);
    h += `</tr></thead><tbody>`;
    tags.forEach(t => {
      h += `<tr><td><b>${esc(t.label)}</b></td><td>${t.expected_count}</td>`;
      models.forEach(m => {
        const mt = (m.detail_diff?.per_tag_f1 || []).find(x => x.tag === t.tag);
        const f1 = mt?.f1;
        const col = scol(f1);
        const statusText = mt ? `<span style="color:var(--muted);font-size:.72rem"> ${mt.status}</span>` : '';
        h += `<td><span style="color:${col};font-weight:700">${f1 != null ? f1.toFixed(3) : '—'}</span>${statusText}</td>`;
      });
      h += `</tr>`;
    });
    h += `</tbody></table></div></div>`;
  }

  return h;
}

function witem(label, val) {
  return `<div class="winner-item"><div class="wl">${label}</div><div class="wv">${esc(val) || '—'}</div></div>`;
}
</script>
</body>
</html>
"""


# ── test page ─────────────────────────────────────────────────────────────────

def _build_test_page() -> str:
    return """\
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Writing API Model Test Page</title>
  <style>
    body{font-family:Segoe UI,Arial,sans-serif;background:#f8fafc;color:#0f172a;margin:0;padding:28px}
    main{max-width:1100px;margin:0 auto}
    h1{font-size:1.5rem;margin-bottom:6px}
    .muted{color:#64748b;font-size:.9rem}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}
    @media(max-width:800px){.grid{grid-template-columns:1fr}}
    .card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:18px;box-shadow:0 1px 3px #0001}
    label{display:block;font-weight:600;font-size:.86rem;margin:12px 0 5px}
    input,select{width:100%;box-sizing:border-box;border:1px solid #cbd5e1;border-radius:7px;padding:8px}
    button{border:0;border-radius:7px;padding:10px 16px;background:#2563eb;color:#fff;font-weight:700;cursor:pointer}
    button.secondary{background:#475569}
    button:disabled{opacity:.55;cursor:not-allowed}
    pre{background:#0f172a;color:#e2e8f0;border-radius:8px;padding:14px;overflow:auto;max-height:540px;font-size:.78rem}
    .row{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
    .pill{display:inline-block;border-radius:999px;padding:4px 10px;font-size:.78rem;font-weight:700;background:#e2e8f0;margin:3px}
    .ok{background:#dcfce7;color:#166534}.bad{background:#fee2e2;color:#991b1b}.warn{background:#fef3c7;color:#92400e}
  </style>
</head>
<body>
<main>
  <div style="margin-bottom:12px"><a href="/" style="font-size:.85rem;color:#2563eb">← 벤치마크 대시보드</a></div>
  <h1>Writing API Model Test Page</h1>
  <p class="muted">Ollama 모델 실행 검증 및 상태 확인 페이지입니다.</p>

  <div class="grid">
    <section class="card">
      <h2>Run Model API</h2>
      <form id="runForm">
        <label>Target</label>
        <select id="target">
          <option value="qwen2">qwen2 only (/analyze-writing/qwen2)</option>
          <option value="gemma">gemma only (/analyze-writing/gemma)</option>
          <option value="both">both (/analyze-writing)</option>
        </select>
        <label>Inference Mode (both 선택 시)</label>
        <select id="mode">
          <option value="local">local Ollama</option>
          <option value="remote">remote AWS endpoint</option>
        </select>
        <label>Run Name</label>
        <input id="runName" value="test_page_run">
        <label>OCR JSON (.json, 우선)</label>
        <input id="ocrFile" type="file" accept=".json,application/json">
        <label>Image (.jpg/.png/.webp, 선택)</label>
        <input id="imageFile" type="file" accept=".jpg,.jpeg,.png,.webp,image/*">
        <label>Expected Output (.json, 선택)</label>
        <input id="expectedFile" type="file" accept=".json,application/json">
        <div class="row">
          <button id="runBtn" type="submit">Run Analysis</button>
          <button class="secondary" type="button" onclick="loadStatus()">상태 새로고침</button>
        </div>
      </form>
    </section>

    <section class="card">
      <h2>상태</h2>
      <div id="statusBadges"></div>
      <h3 style="margin-top:12px">Health</h3>
      <pre id="healthOut">로딩 중...</pre>
      <h3>Models</h3>
      <pre id="modelsOut">로딩 중...</pre>
    </section>
  </div>

  <section class="card" style="margin-top:16px">
    <h2>API 결과</h2>
    <pre id="resultOut">아직 실행 없음.</pre>
  </section>
</main>
<script>
async function asJson(r) { const t = await r.text(); try { return JSON.parse(t); } catch { return {raw:t}; } }
function show(id,v) { document.getElementById(id).textContent = JSON.stringify(v,null,2); }
function badge(text,cls) { return `<span class="pill ${cls}">${text}</span>`; }
async function loadStatus() {
  const [health,models] = await Promise.all([fetch('/health').then(asJson),fetch('/models').then(asJson)]);
  show('healthOut',health); show('modelsOut',models);
  const overall = health.overall==='PASS'?'ok':'warn';
  const mb = (models.models||[]).map(m=>badge(`${m.name}: ${m.available?'local ok':'missing'}`,m.available?'ok':'bad')).join('');
  document.getElementById('statusBadges').innerHTML = badge(`health ${health.overall}`,overall)+mb;
}
document.getElementById('runForm').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = document.getElementById('runBtn'); btn.disabled=true;
  document.getElementById('resultOut').textContent='실행 중...';
  try {
    const target=document.getElementById('target').value;
    const mode=document.getElementById('mode').value;
    const form=new FormData();
    const rn=document.getElementById('runName').value; if(rn) form.append('run_name',rn);
    const ocr=document.getElementById('ocrFile').files[0];
    const img=document.getElementById('imageFile').files[0];
    const exp=document.getElementById('expectedFile').files[0];
    if(ocr) form.append('ocr_json_file',ocr);
    if(img) form.append('image_file',img);
    if(exp) form.append('expected_output_file',exp);
    let url='/analyze-writing';
    if(target==='qwen2') url='/analyze-writing/qwen2';
    if(target==='gemma') url='/analyze-writing/gemma';
    if(target==='both'){form.append('models','qwen2,gemma');form.append('inference_mode',mode);}
    const r=await fetch(url,{method:'POST',body:form});
    show('resultOut',{status:r.status,ok:r.ok,body:await asJson(r)});
  } catch(err){show('resultOut',{error:String(err)});}
  finally{btn.disabled=false;}
});
loadStatus();
</script>
</body>
</html>
"""


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    health = run_health_check()
    health_ok = health.get("overall") == "PASS"
    hbadge = (
        f'<span class="badge {"b-ok" if health_ok else "b-err"}">'
        f'Health {"PASS" if health_ok else "FAIL"} '
        f'{health.get("passed",0)}/{health.get("total",0)}</span>'
    )
    mbadges = "".join(
        f'<span class="badge {"b-ok" if is_model_available(m["ollama_model"]) else "b-err"}">'
        f'{_e(m["name"])}</span>'
        for m in get_enabled_models()
    )
    return _build_page(hbadge, mbadges)


@app.get("/test-page", response_class=HTMLResponse)
def test_page():
    return _build_test_page()


# ── Data API ──────────────────────────────────────────────────────────────────

@app.get("/api/sys-prompt")
async def api_sys_prompt():
    return JSONResponse({"content": _read_prompt_file("system_prompt.txt")})


@app.get("/api/out-prompt")
async def api_out_prompt():
    return JSONResponse({"content": _read_prompt_file("output_prompt.txt")})


@app.get("/api/expected")
async def api_expected():
    return JSONResponse({"expected": _latest_expected_text()})


@app.post("/api/run")
async def api_run(
    image: UploadFile = File(...),
    system_prompt: str = Form(...),
    output_prompt: str = Form(...),
    expected: str = Form(default=""),
    run_name: str = Form(default=""),
):
    if not run_name:
        run_name = f"api_{timestamp()}"
    run_name = sanitize_run_name(run_name)

    if expected:
        try:
            json.loads(expected)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"정답 분석지 JSON 오류: {exc}")

    tmp_dir = Path(tempfile.mkdtemp())
    image_suffix = Path(image.filename or "image.jpg").suffix.lower() or ".jpg"
    img_path = tmp_dir / f"image{image_suffix}"
    validate_image_extension(img_path)
    img_path.write_bytes(await image.read())

    exp_path = ""
    if expected:
        ep = tmp_dir / "expected.json"
        ep.write_text(expected, encoding="utf-8")
        exp_path = str(ep)

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "step": "시작 중...", "progress": 0.03, "result": None, "error": None}

    threading.Thread(
        target=_run_job,
        args=(job_id, str(img_path), system_prompt, output_prompt, exp_path, run_name, tmp_dir),
        daemon=True,
    ).start()

    return JSONResponse({"job_id": job_id})


@app.get("/api/run/{job_id}")
async def api_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job)


# ── Health / Models ───────────────────────────────────────────────────────────

@app.get("/health")
def health_endpoint():
    return JSONResponse(run_health_check())


@app.get("/models")
def models_endpoint():
    models = []
    for m in load_models_config().get("models", []):
        endpoint_env = m.get("endpoint_env")
        ollama_model = m.get("ollama_model")
        storage = m.get("model_storage") or {}
        bucket_env = storage.get("bucket_env")
        models.append({
            "name": m.get("name"),
            "provider": m.get("provider", "ollama_local"),
            "enabled": m.get("enabled", True),
            "ollama_model": ollama_model,
            "available": is_model_available(ollama_model) if ollama_model else None,
            "endpoint_configured": bool(os.getenv(endpoint_env or "")) if endpoint_env else False,
            "endpoint_env": endpoint_env,
            "model_artifact_bucket_configured": bool(os.getenv(bucket_env or "")) if bucket_env else False,
            "model_artifact_key": storage.get("key"),
        })
    return JSONResponse({"models": models})


@app.get("/aws/model-artifacts")
def aws_model_artifacts_endpoint():
    return JSONResponse({"models": list_model_artifact_manifests()})


# ── Writing analysis endpoints ────────────────────────────────────────────────

@app.post("/analyze-writing")
async def analyze_writing(
    image_file: UploadFile | None = File(default=None),
    ocr_json_file: UploadFile | None = File(default=None),
    system_prompt_file: UploadFile | None = File(default=None),
    output_prompt_file: UploadFile | None = File(default=None),
    expected_output_file: UploadFile | None = File(default=None),
    run_name: str = Form(default=""),
    models: str | None = Form(default=None),
    inference_mode: str | None = Form(default=None),
):
    return await _analyze_writing_impl(
        image_file=image_file, ocr_json_file=ocr_json_file,
        system_prompt_file=system_prompt_file, output_prompt_file=output_prompt_file,
        expected_output_file=expected_output_file, run_name=run_name,
        models=models, inference_mode=inference_mode,
    )


@app.post("/analyze-writing/qwen2")
async def analyze_writing_qwen2(
    image_file: UploadFile | None = File(default=None),
    ocr_json_file: UploadFile | None = File(default=None),
    system_prompt_file: UploadFile | None = File(default=None),
    output_prompt_file: UploadFile | None = File(default=None),
    expected_output_file: UploadFile | None = File(default=None),
    run_name: str = Form(default=""),
):
    return await _analyze_writing_impl(
        image_file=image_file, ocr_json_file=ocr_json_file,
        system_prompt_file=system_prompt_file, output_prompt_file=output_prompt_file,
        expected_output_file=expected_output_file, run_name=run_name,
        models=None, inference_mode="local", forced_model="qwen2",
    )


@app.post("/analyze-writing/gemma")
async def analyze_writing_gemma(
    image_file: UploadFile | None = File(default=None),
    ocr_json_file: UploadFile | None = File(default=None),
    system_prompt_file: UploadFile | None = File(default=None),
    output_prompt_file: UploadFile | None = File(default=None),
    expected_output_file: UploadFile | None = File(default=None),
    run_name: str = Form(default=""),
):
    return await _analyze_writing_impl(
        image_file=image_file, ocr_json_file=ocr_json_file,
        system_prompt_file=system_prompt_file, output_prompt_file=output_prompt_file,
        expected_output_file=expected_output_file, run_name=run_name,
        models=None, inference_mode="local", forced_model="gemma",
    )


@app.post("/validate-output")
async def validate_output_endpoint(output_json_file: UploadFile = File(...)):
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        output_path = tmp_dir / "model_output.json"
        await _save_upload(output_json_file, output_path, JSON_EXTENSIONS, "output_json_file")
        parsed = json.loads(output_path.read_text(encoding="utf-8"))
        return JSONResponse(validate_output(parsed))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/benchmark")
async def benchmark_compat(
    image: UploadFile = File(...),
    system_prompt: UploadFile = File(...),
    output_prompt: UploadFile = File(...),
    expected_output: UploadFile = File(...),
    run_name: str = Form(default=""),
):
    """Legacy CLI compatibility endpoint."""
    if not run_name:
        run_name = f"api_run_{timestamp()}"
    run_name = sanitize_run_name(run_name)
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        image_suffix = Path(image.filename or "image.jpg").suffix.lower() or ".jpg"
        img_path = tmp_dir / f"image{image_suffix}"
        validate_image_extension(img_path)
        img_path.write_bytes(await image.read())
        sys_path = tmp_dir / "system_prompt.txt"
        sys_path.write_bytes(await system_prompt.read())
        out_path = tmp_dir / "output_prompt.txt"
        out_path.write_bytes(await output_prompt.read())
        exp_path = tmp_dir / "expected.json"
        exp_path.write_bytes(await expected_output.read())
        summary = run_benchmark(
            image_path=str(img_path),
            system_prompt_path=str(sys_path),
            output_prompt_path=str(out_path),
            expected_path=str(exp_path),
            run_name=run_name,
        )
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
