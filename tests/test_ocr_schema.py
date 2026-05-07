import pytest
from pydantic import ValidationError

from src.ocr_schema import OCRResult


def _valid_ocr():
    return {
        "metadata": {"class": "A", "title": "T"},
        "handwritten_text": "I like apples.",
        "lines": [{"line_no": 1, "text": "I like apples.", "confidence": 0.98}],
        "source": {"type": "json", "filename": "ocr.json", "ocr_engine": "manual"},
    }


def test_ocr_json_valid_case():
    result = OCRResult.model_validate(_valid_ocr())
    assert result.handwritten_text == "I like apples."
    assert result.lines[0].confidence == 0.98


def test_missing_metadata_defaults():
    data = _valid_ocr()
    data["metadata"] = {}
    result = OCRResult.model_validate(data)
    dumped = result.model_dump(by_alias=True)
    assert dumped["metadata"]["class"] == "not provided"
    assert dumped["metadata"]["level"] == "not provided"


def test_confidence_range_validation():
    data = _valid_ocr()
    data["lines"][0]["confidence"] = 1.5
    with pytest.raises(ValidationError):
        OCRResult.model_validate(data)


def test_invalid_source_type():
    data = _valid_ocr()
    data["source"]["type"] = "pdf"
    with pytest.raises(ValidationError):
        OCRResult.model_validate(data)


def test_empty_handwritten_text_warning():
    data = _valid_ocr()
    data["handwritten_text"] = ""
    result = OCRResult.model_validate(data)
    assert "handwritten_text is empty" in result.warnings
