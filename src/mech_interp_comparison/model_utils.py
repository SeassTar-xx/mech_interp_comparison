"""Model discovery helpers that avoid downloads by default."""

from __future__ import annotations

from pathlib import Path

from .paths import MODEL_ROOT


def list_local_model_candidates(model_root: Path = MODEL_ROOT) -> list[Path]:
    """List top-level local model candidates without loading model weights."""
    if not model_root.exists():
        return []
    return sorted(path for path in model_root.iterdir() if path.is_dir())


def resolve_local_model(name: str, model_root: Path = MODEL_ROOT) -> Path:
    """Resolve a local model directory by exact name and fail if it is absent."""
    candidate = model_root / name
    if not candidate.exists():
        raise FileNotFoundError(
            f"Local model not found: {candidate}. Check {model_root} before requesting downloads."
        )
    if not candidate.is_dir():
        raise NotADirectoryError(f"Local model path is not a directory: {candidate}")
    return candidate
