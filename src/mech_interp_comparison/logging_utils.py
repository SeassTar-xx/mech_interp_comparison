"""Logging helpers for reproducible comparison runs."""

from __future__ import annotations

import logging
from pathlib import Path

from .paths import LOG_DIR


def configure_logging(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Create a logger that writes to stdout and, optionally, a project log file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        target = Path(log_file)
        if not target.is_absolute():
            target = LOG_DIR / target
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
