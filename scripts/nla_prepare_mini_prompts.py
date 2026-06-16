#!/usr/bin/env python3
"""Validate and prepare the NLA mini prompt dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shlex
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path("/mnt/20t/xuxin/mech_interp_comparison")
DEFAULT_PROMPTS = PROJECT_ROOT / "data" / "mini_prompts" / "nla_mini_safety_bias_sycophancy.jsonl"
REQUIRED_FIELDS = {"id", "category", "prompt", "expected_behavior_note", "counterfactual_group_id", "notes"}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_dir(base: Path, name: str | None) -> Path:
    out = Path(name) if name else base / "outputs" / "nla" / timestamp()
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_common_run_files(out: Path, args: argparse.Namespace) -> None:
    (out / "run_config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "command.txt").write_text(" ".join(shlex.quote(part) for part in sys.argv) + "\n", encoding="utf-8")
    lines = [f"python={sys.executable}", f"python_version={platform.python_version()}"]
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "TMPDIR", "PYTHONPATH"]:
        lines.append(f"{key}={os.environ.get(key, '')}")
    (out / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_logging(out: Path) -> logging.Logger:
    logger = logging.getLogger("nla_prepare_mini_prompts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(out / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = make_output_dir(args.project_root, args.output_dir)
    write_common_run_files(out, args)
    logger = configure_logging(out)
    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Prompt ids must be unique")
    category_counts: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
    output_path = out / "mini_prompts.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "input": str(args.input),
        "output": str(output_path),
        "count": len(rows),
        "category_counts": category_counts,
    }
    (out / "mini_prompt_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Prepared %d prompts", len(rows))
    logger.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
