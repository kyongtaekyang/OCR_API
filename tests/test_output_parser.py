import pytest
from src.output_parser import extract_json_object, parse_model_output


VALID_JSON = '{"metadata": {"class": "A"}, "original_writing": "hello"}'


def test_extract_plain_json():
    result = extract_json_object(VALID_JSON)
    assert result is not None
    import json
    parsed = json.loads(result)
    assert parsed["metadata"]["class"] == "A"


def test_extract_json_with_preamble():
    raw = f"Here is the evaluation:\n{VALID_JSON}\nThank you."
    result = extract_json_object(raw)
    assert result is not None
    import json
    assert json.loads(result)["metadata"]["class"] == "A"


def test_extract_json_with_code_fence():
    raw = f"```json\n{VALID_JSON}\n```"
    result = extract_json_object(raw)
    assert result is not None
    import json
    assert json.loads(result)["metadata"]["class"] == "A"


def test_extract_returns_none_for_no_json():
    assert extract_json_object("No JSON here at all.") is None


def test_extract_returns_none_for_empty():
    assert extract_json_object("") is None


def test_parse_model_output_success():
    result = parse_model_output(VALID_JSON)
    assert result["parse_success"] is True
    assert result["parsed_json"] is not None
    assert result["error"] is None


def test_parse_model_output_with_preamble():
    raw = f"Some text before. {VALID_JSON} Some text after."
    result = parse_model_output(raw)
    assert result["parse_success"] is True


def test_parse_model_output_invalid_json():
    result = parse_model_output("{ invalid json :::}")
    assert result["parse_success"] is False
    assert result["parsed_json"] is None
    assert result["error"] is not None


def test_parse_model_output_empty():
    result = parse_model_output("")
    assert result["parse_success"] is False


def test_parse_model_output_no_crash_on_bad_input():
    result = parse_model_output("completely not json at all")
    assert result["parse_success"] is False
    assert "error" in result
