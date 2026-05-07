import logging
import re
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def safe_path(base: Path, relative: str) -> Path:
    """Resolve relative path under base and block path traversal."""
    resolved = (base / relative).resolve()
    base_resolved = base.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Path traversal detected: {relative}")
    return resolved


def sanitize_run_name(run_name: str) -> str:
    """Return a filesystem-safe run name for result directories."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name or "").strip("._-")
    if not cleaned:
        cleaned = f"run_{timestamp()}"
    return cleaned[:100]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(run_name: str) -> logging.Logger:
    run_name = sanitize_run_name(run_name)
    log_dir = ensure_dir(PROJECT_ROOT / "results" / "logs" / run_name)
    log_file = log_dir / "run.log"

    logger = logging.getLogger(f"benchmark.{run_name}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_image_extension(path: Path) -> None:
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    if path.suffix.lower() not in allowed:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
