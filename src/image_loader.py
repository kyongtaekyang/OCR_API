import base64
from pathlib import Path
from src.utils import validate_image_extension


def validate_image_path(path: str) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")
    validate_image_extension(p)
    return p


def load_image_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def load_image_as_base64(path: str) -> str:
    p = validate_image_path(path)
    data = load_image_bytes(p)
    return image_to_base64(data)
