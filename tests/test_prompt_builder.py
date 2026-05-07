import pytest
import tempfile
from pathlib import Path
from src.prompt_builder import build_combined_prompt, build_compact_eval_prompt, save_combined_prompt


@pytest.fixture
def prompt_files(tmp_path):
    sys_p = tmp_path / "system_prompt.txt"
    out_p = tmp_path / "output_prompt.txt"
    sys_p.write_text("You are an expert EFL assessor.", encoding="utf-8")
    out_p.write_text("Return a JSON object.", encoding="utf-8")
    return str(sys_p), str(out_p)


def test_build_combined_contains_both(prompt_files):
    sys_p, out_p = prompt_files
    result = build_combined_prompt(sys_p, out_p)
    assert "expert EFL assessor" in result
    assert "Return a JSON object" in result


def test_build_combined_contains_json_only_rule(prompt_files):
    sys_p, out_p = prompt_files
    result = build_combined_prompt(sys_p, out_p)
    assert "Return only one valid JSON object" in result


def test_build_combined_contains_no_markdown_rule(prompt_files):
    sys_p, out_p = prompt_files
    result = build_combined_prompt(sys_p, out_p)
    assert "Do not output markdown" in result


def test_build_combined_raises_if_system_prompt_missing(tmp_path):
    out_p = tmp_path / "output_prompt.txt"
    out_p.write_text("output", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        build_combined_prompt(str(tmp_path / "nonexistent.txt"), str(out_p))


def test_build_combined_raises_if_output_prompt_missing(tmp_path):
    sys_p = tmp_path / "system_prompt.txt"
    sys_p.write_text("system", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        build_combined_prompt(str(sys_p), str(tmp_path / "nonexistent.txt"))


def test_save_combined_prompt(tmp_path, prompt_files):
    sys_p, out_p = prompt_files
    combined = build_combined_prompt(sys_p, out_p)
    out = tmp_path / "combined.txt"
    result_path = save_combined_prompt(combined, str(out))
    assert result_path.exists()
    assert result_path.read_text(encoding="utf-8") == combined


def test_compact_prompt_contains_required_contract():
    prompt = build_compact_eval_prompt()
    assert "Return only one valid JSON object" in prompt
    assert '"metadata"' in prompt
    assert '"error_analysis"' in prompt
    assert "Tag count must equal error_explanations count" in prompt
