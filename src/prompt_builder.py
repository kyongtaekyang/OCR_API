import json
from pathlib import Path


COMBINED_HEADER = """You are an expert English writing assessor for EFL students.

You will receive a direct image of a student's handwritten English writing assignment.

=== OUTPUT RULES ===
Return only one valid JSON object.
Do not output anything outside JSON.
Do not output markdown.
Do not output code fences.
Do not output explanation outside JSON.

"""

COMBINED_FOOTER = """

=== MANDATORY VALIDATION BEFORE OUTPUT ===
- Return only one valid JSON object.
- Do not output markdown.
- Do not output code fences.
- Do not output explanation outside JSON.
- Do not silently fix untagged errors.
- Every changed word in corrected_writing must have a corresponding tag in original_writing.
- Tag count = error_explanations count = error_analysis.total_errors.
- scoring_analysis subtotal must equal sum of sub-scores.
- writing_performance percentages must be mathematically valid.
- total_score must be mathematically valid.
- All scores must be <= max_score.
"""


def build_combined_prompt(system_prompt_path: str, output_prompt_path: str) -> str:
    sys_path = Path(system_prompt_path)
    out_path = Path(output_prompt_path)

    if not sys_path.exists():
        raise FileNotFoundError(f"system_prompt not found: {system_prompt_path}")
    if not out_path.exists():
        raise FileNotFoundError(f"output_prompt not found: {output_prompt_path}")

    system_text = sys_path.read_text(encoding="utf-8").strip()
    output_text = out_path.read_text(encoding="utf-8").strip()

    combined = (
        COMBINED_HEADER
        + "=== ASSESSMENT SYSTEM RULES ===\n"
        + system_text
        + "\n\n=== OUTPUT FORMAT REFERENCE ===\n"
        + output_text
        + COMBINED_FOOTER
    )
    return combined


def build_compact_eval_prompt() -> str:
    """Build a shorter prompt for VLMs that fail on the full benchmark prompt."""
    return """You are an expert EFL English writing assessor. Read only the student's handwritten writing in the image. Ignore printed instructions. Return only one valid JSON object. Do not use markdown or code fences.

Use this exact top-level JSON structure:
{
  "metadata": {"course_type":"keystone|zoom|not provided", "class":"...", "title":"...", "title_corrected":"...", "topic":"...", "topic_corrected":"...", "writing_type":"..."},
  "original_writing":"student text with error tags [G][V][O][P][S][C] on erroneous words only",
  "corrected_writing":"corrected text without tags",
  "scoring_analysis": {
    "grammar": {"sentence_accuracy":{"score":0,"max_score":10}, "verb_tense_consistency":{"score":0,"max_score":10}, "article_preposition":{"score":0,"max_score":10}, "subtotal":{"score":0,"max_score":30}},
    "vocabulary": {"word_variety":{"score":0,"max_score":10}, "appropriateness":{"score":0,"max_score":10}, "expression_naturalness":{"score":0,"max_score":10}, "subtotal":{"score":0,"max_score":30}},
    "writing_flow": {"structure_organization":{"score":0,"max_score":15}, "sentence_variety":{"score":0,"max_score":10}, "coherence_transitions":{"score":0,"max_score":15}, "subtotal":{"score":0,"max_score":40}}
  },
  "writing_performance": {"grammar":{"score":0,"percentage":0}, "vocabulary":{"score":0,"percentage":0}, "writing_flow":{"score":0,"percentage":0}, "total_score":{"score":0,"max_score":100,"percentage":0}},
  "error_explanations": [{"error":"...", "explanation_en":"...", "explanation":"..."}],
  "error_analysis": {"grammar":0, "vocabulary":0, "word_order":0, "punctuation":0, "spelling":0, "coherence":0, "total_errors":0},
  "overall_comments":"brief supportive comment"
}

Rules:
- Level containing Zoom => zoom; Keystone => keystone; missing/unreadable => not provided.
- Error tags: [G] grammar, [V] vocabulary, [O] word order, [P] punctuation/capitalization, [S] spelling, [C] coherence.
- Put every error tag directly before the erroneous word in original_writing.
- Corrected_writing must only fix tagged errors and must not contain tags.
- Tag count must equal error_explanations count and error_analysis.total_errors.
- Keep error_explanations concise. Do not repeat the same error item unless it is a distinct occurrence in the writing.
- Subtotals must equal child score sums.
- Percentages and total_score must be mathematically valid.
- Missing metadata fields must be "not provided".
- Close the JSON object completely.
"""


def build_prompt_from_ocr_json(system_prompt: str, output_prompt: str, ocr_json: dict) -> str:
    ocr_text = json.dumps(ocr_json, ensure_ascii=False, indent=2)
    return (
        "[ORIGINAL SYSTEM PROMPT - DO NOT MODIFY]\n"
        + system_prompt
        + "\n\n[ORIGINAL OUTPUT PROMPT - DO NOT MODIFY]\n"
        + output_prompt
        + "\n\n[OCR RESULT JSON]\n"
        + ocr_text
        + "\n\n[EXECUTION INSTRUCTION]\n"
        + "Use the OCR RESULT JSON as the recognized student writing input.\n"
        + "Apply the original system prompt and output prompt exactly.\n"
        + "Return only one valid JSON object.\n"
        + "Do not output markdown.\n"
        + "Do not output code fences.\n"
        + "Do not output explanation outside JSON.\n"
    )


def build_compact_prompt_from_ocr_json(ocr_json: dict) -> str:
    ocr_text = json.dumps(ocr_json, ensure_ascii=False, indent=2)
    return build_compact_eval_prompt() + "\n\nOCR RESULT JSON:\n" + ocr_text


def save_combined_prompt(prompt: str, output_path: str) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(prompt, encoding="utf-8")
    return p
