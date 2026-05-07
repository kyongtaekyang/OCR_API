"""Security-oriented utility tests."""
from pathlib import Path

import pytest

from src.utils import safe_path, sanitize_run_name


def test_sanitize_run_name_removes_path_segments():
    assert sanitize_run_name("../bad/run") == "bad_run"


def test_sanitize_run_name_fallback_for_empty_value():
    assert sanitize_run_name("").startswith("run_")


def test_safe_path_blocks_traversal(tmp_path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../outside")


def test_safe_path_allows_child_path(tmp_path):
    resolved = safe_path(tmp_path, "child/file.txt")
    assert resolved == (tmp_path / "child" / "file.txt").resolve()
