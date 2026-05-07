import json
import os
import gc
from pathlib import Path

from src.ocr_schema import OCRResult
from src.utils import PROJECT_ROOT, ensure_dir


_DOCTR_PREDICTOR = None


class OCRProcessor:
    """Adapter boundary for OCR engines such as Textract, Vision, or Tesseract."""

    def process_image(self, image_path: str) -> OCRResult:
        path = Path(image_path)
        try:
            return self._process_image_with_doctr(path)
        except Exception as exc:
            return OCRResult(
                metadata={},
                handwritten_text="",
                lines=[],
                source={"type": "image", "filename": path.name, "ocr_engine": "mock"},
                warnings=[f"docTR OCR failed: {exc}", "OCR failed; handwritten_text is empty"],
            )

    def _process_image_with_doctr(self, path: Path) -> OCRResult:
        predictor = _get_doctr_predictor()
        document = _load_doctr_document(path)
        result = predictor(document)
        exported = result.export()
        lines = _extract_doctr_lines(exported)
        handwritten_text = "\n".join(line["text"] for line in lines).strip()
        warnings = []
        if not handwritten_text:
            warnings.append("docTR returned no recognized text")
        return OCRResult(
            metadata={},
            handwritten_text=handwritten_text,
            lines=lines,
            source={"type": "image", "filename": path.name, "ocr_engine": "doctr"},
            warnings=warnings,
        )

    def load_ocr_json(self, ocr_json_path: str) -> OCRResult:
        path = Path(ocr_json_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("source", {})
        data["source"].setdefault("type", "json")
        data["source"].setdefault("filename", path.name)
        data["source"].setdefault("ocr_engine", "manual")
        return OCRResult.model_validate(data)


def _get_doctr_predictor():
    global _DOCTR_PREDICTOR
    if _DOCTR_PREDICTOR is None:
        cache_dir = ensure_dir(PROJECT_ROOT / ".cache" / "doctr")
        os.environ.setdefault("DOCTR_CACHE_DIR", str(cache_dir))
        os.environ.setdefault("DOCTR_MULTIPROCESSING_DISABLE", "TRUE")
        from doctr.models import ocr_predictor

        _DOCTR_PREDICTOR = ocr_predictor(pretrained=True)
    return _DOCTR_PREDICTOR


def release_doctr_predictor() -> None:
    global _DOCTR_PREDICTOR
    _DOCTR_PREDICTOR = None
    gc.collect()
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_doctr_document(path: Path):
    from doctr.io import DocumentFile

    if path.suffix.lower() == ".pdf":
        return DocumentFile.from_pdf(str(path))
    return DocumentFile.from_images(str(path))


def _extract_doctr_lines(exported: dict) -> list[dict]:
    ocr_lines = []
    line_no = 1
    for page in exported.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = line.get("words", [])
                text = " ".join(str(word.get("value", "")).strip() for word in words).strip()
                if not text:
                    continue
                confidences = [
                    float(word.get("confidence"))
                    for word in words
                    if isinstance(word.get("confidence"), (int, float))
                ]
                confidence = sum(confidences) / len(confidences) if confidences else 0.0
                ocr_lines.append({
                    "line_no": line_no,
                    "text": text,
                    "confidence": round(max(0.0, min(1.0, confidence)), 4),
                })
                line_no += 1
    return ocr_lines
