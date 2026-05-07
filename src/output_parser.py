import json
import re


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _extract_by_brace_balance(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_object(raw_text: str) -> str | None:
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()

    # Try direct parse first
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    # Strip code fences and retry
    no_fence = _strip_code_fences(cleaned)
    try:
        json.loads(no_fence)
        return no_fence
    except json.JSONDecodeError:
        pass

    # Brace-balance extraction
    candidate = _extract_by_brace_balance(no_fence) or _extract_by_brace_balance(cleaned)
    if candidate:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def parse_model_output(raw_text: str) -> dict:
    json_str = extract_json_object(raw_text)
    if json_str is None:
        return {
            "parse_success": False,
            "parsed_json": None,
            "error": "No JSON object found in output",
        }
    try:
        parsed = json.loads(json_str)
        return {"parse_success": True, "parsed_json": parsed, "error": None}
    except json.JSONDecodeError as e:
        return {
            "parse_success": False,
            "parsed_json": None,
            "error": f"JSONDecodeError: {e}",
        }
