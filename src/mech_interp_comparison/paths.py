"""Central path definitions for the mechanistic interpretability comparison project."""

from pathlib import Path

PROJECT_ROOT = Path("/mnt/20t/xuxin/mech_interp_comparison")
MODEL_ROOT = Path("/mnt/20t/xuhaoming/models")

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
REPO_DIR = PROJECT_ROOT / "repos"
CACHE_DIR = PROJECT_ROOT / ".cache"
TMP_DIR = PROJECT_ROOT / "tmp"


def project_path(*parts: str) -> Path:
    """Return a path inside the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def model_path(*parts: str) -> Path:
    """Return a path inside the read-preferred model root."""
    return MODEL_ROOT.joinpath(*parts)


def managed_directories() -> tuple[Path, ...]:
    """Return directories that should stay under the project root."""
    return (
        DATA_DIR,
        OUTPUT_DIR,
        LOG_DIR,
        REPO_DIR,
        CACHE_DIR,
        TMP_DIR,
    )
