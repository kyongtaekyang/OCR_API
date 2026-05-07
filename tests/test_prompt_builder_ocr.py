from src.prompt_builder import build_prompt_from_ocr_json


def test_ocr_json_included_in_prompt():
    prompt = build_prompt_from_ocr_json(
        "SYSTEM ORIGINAL",
        "OUTPUT ORIGINAL",
        {"handwritten_text": "I goed home."},
    )
    assert '"handwritten_text": "I goed home."' in prompt


def test_system_prompt_original_is_preserved():
    system_prompt = "Line 1\n\nLine 2 with exact spacing"
    prompt = build_prompt_from_ocr_json(system_prompt, "OUTPUT", {})
    assert system_prompt in prompt


def test_output_prompt_original_is_preserved():
    output_prompt = '{"required": true}\nDo not change this.'
    prompt = build_prompt_from_ocr_json("SYSTEM", output_prompt, {})
    assert output_prompt in prompt


def test_json_only_instruction_is_included():
    prompt = build_prompt_from_ocr_json("SYSTEM", "OUTPUT", {})
    assert "Return only one valid JSON object." in prompt
    assert "Do not output markdown." in prompt
    assert "Do not output code fences." in prompt
