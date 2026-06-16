#!/usr/bin/env python3
"""Score NLA AR reconstruction quality for generated explanations.

Attribution: this script imports NLACritic from kitft/nla-inference
(https://github.com/kitft/nla-inference), Apache License 2.0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path("/mnt/20t/xuxin/mech_interp_comparison")
REPO_PATH = PROJECT_ROOT / "repos" / "nla-inference"
DEFAULT_AR = Path("/mnt/20t/xuhaoming/models/nla-qwen2.5-7b-L20-ar")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_dir(base: Path, name: str | None) -> Path:
    out = Path(name) if name else base / "outputs" / "nla" / timestamp()
    out.mkdir(parents=True, exist_ok=True)
    return out


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_common_run_files(out: Path, args: argparse.Namespace) -> None:
    (out / "run_config.json").write_text(json.dumps(jsonable_args(args), indent=2, sort_keys=True), encoding="utf-8")
    (out / "command.txt").write_text(" ".join(shlex.quote(part) for part in sys.argv) + "\n", encoding="utf-8")
    lines = [
        f"python={sys.executable}",
        f"python_version={platform.python_version()}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
    ]
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "TMPDIR", "CUDA_VISIBLE_DEVICES", "PYTHONPATH"]:
        lines.append(f"{key}={os.environ.get(key, '')}")
    (out / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_logging(out: Path) -> logging.Logger:
    logger = logging.getLogger("nla_score_ar_reconstruction")
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


def load_nla_critic(repo_path: Path) -> Any:
    module_path = repo_path / "nla_inference.py"
    spec = importlib.util.spec_from_file_location("kitft_nla_inference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NLACritic


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--repo-path", type=Path, default=REPO_PATH)
    parser.add_argument("--ar-checkpoint", type=Path, default=DEFAULT_AR)
    parser.add_argument("--activation-file", type=Path, required=True)
    parser.add_argument("--nla-explanations", type=Path, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out = make_output_dir(args.project_root, args.output_dir)
    write_common_run_files(out, args)
    logger = configure_logging(out)
    if not args.ar_checkpoint.exists():
        raise FileNotFoundError(f"Missing AR checkpoint: {args.ar_checkpoint}")
    rows = read_jsonl(args.nla_explanations)
    data = np.load(args.activation_file, allow_pickle=False)
    activations = {str(row_id): vector for row_id, vector in zip(data["ids"], data["activations"])}
    NLACritic = load_nla_critic(args.repo_path)
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Loading AR critic from %s", args.ar_checkpoint)
    critic = NLACritic(str(args.ar_checkpoint), device=device)
    with (out / "nla_reconstruction_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            row_id = str(row["id"])
            if row.get("parsing_status") != "ok" or row_id not in activations:
                record = {
                    "id": row_id,
                    "status": "skipped",
                    "reason": "missing parsed explanation or activation",
                }
            else:
                mse, cos = critic.score(str(row["nla_explanation"]), activations[row_id])
                record = {
                    "id": row_id,
                    "status": "ok",
                    "mse_nrm": mse,
                    "cosine_similarity": cos,
                    "ar_checkpoint": str(args.ar_checkpoint),
                }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            logger.info("Scored AR reconstruction for %s with status %s", row_id, record["status"])
    status = {
        "status": "ok",
        "stage": "nla_ar_reconstruction",
        "ar_checkpoint": str(args.ar_checkpoint),
        "processed_count": len(rows),
    }
    (out / "artifact_status_ar.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
