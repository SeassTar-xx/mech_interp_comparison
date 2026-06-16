#!/usr/bin/env python3
"""Run an NLA AV smoke test from saved activations.

Attribution: this wrapper uses the public kitft/nla-inference repository as an
external inference dependency when available. Source:
https://github.com/kitft/nla-inference, Apache License 2.0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/mnt/20t/xuxin/mech_interp_comparison")
REPO_PATH = PROJECT_ROOT / "repos" / "nla-inference"
DEFAULT_AV = Path("/mnt/20t/xuhaoming/models/nla-qwen2.5-7b-L20-av")
DEFAULT_AR = Path("/mnt/20t/xuhaoming/models/nla-qwen2.5-7b-L20-ar")
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


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
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "TMPDIR", "CUDA_VISIBLE_DEVICES", "PYTHONPATH"]:
        lines.append(f"{key}={os.environ.get(key, '')}")
    (out / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_logging(out: Path) -> logging.Logger:
    logger = logging.getLogger("nla_run_av_smoke_test")
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


def load_nla_client_class(repo_path: Path) -> Any:
    module_path = repo_path / "nla_inference.py"
    if not module_path.exists():
        raise FileNotFoundError(f"NLA inference script not found: {module_path}")
    spec = importlib.util.spec_from_file_location("kitft_nla_inference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NLAClient


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_blocker(out: Path, reason: str, args: argparse.Namespace) -> None:
    status = {
        "status": "blocked",
        "stage": "nla_av_smoke_test",
        "reason": reason,
        "av_checkpoint": str(args.av_checkpoint),
        "ar_checkpoint": str(args.ar_checkpoint),
        "download_performed": False,
        "server_started": False,
    }
    (out / "artifact_status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    (out / "nla_explanations.jsonl").write_text("", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--repo-path", type=Path, default=REPO_PATH)
    parser.add_argument("--av-checkpoint", type=Path, default=DEFAULT_AV)
    parser.add_argument("--ar-checkpoint", type=Path, default=DEFAULT_AR)
    parser.add_argument("--activation-file", type=Path, required=True)
    parser.add_argument("--base-outputs", type=Path, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sglang-url", default="http://localhost:30000")
    parser.add_argument("--max-items", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    out = make_output_dir(args.project_root, args.output_dir)
    write_common_run_files(out, args)
    logger = configure_logging(out)
    if not args.av_checkpoint.exists():
        reason = (
            f"Missing NLA AV checkpoint: {args.av_checkpoint}. Expected released checkpoint "
            "kitft/nla-qwen2.5-7b-L20-av, but no local copy was found. Download requires explicit approval."
        )
        logger.error(reason)
        write_blocker(out, reason, args)
        return 2
    if not args.activation_file.exists():
        reason = f"Missing activation file: {args.activation_file}"
        logger.error(reason)
        write_blocker(out, reason, args)
        return 2

    try:
        NLAClient = load_nla_client_class(args.repo_path)
    except Exception as exc:
        reason = f"Failed to import kitft/nla-inference client: {type(exc).__name__}: {exc}"
        logger.exception(reason)
        write_blocker(out, reason, args)
        return 2

    data = np.load(args.activation_file, allow_pickle=False)
    activations = data["activations"][: args.max_items]
    ids = [str(item) for item in data["ids"][: args.max_items]]
    base_rows = {str(row.get("id")): row for row in read_jsonl(args.base_outputs)} if args.base_outputs else {}
    try:
        client = NLAClient(str(args.av_checkpoint), sglang_url=args.sglang_url)
    except Exception as exc:
        reason = f"Failed to initialize NLAClient or reach SGLang: {type(exc).__name__}: {exc}"
        logger.exception(reason)
        write_blocker(out, reason, args)
        return 2

    with (out / "nla_explanations.jsonl").open("w", encoding="utf-8") as handle:
        for row_id, activation in zip(ids, activations):
            raw = client.generate(
                activation,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                extract_explanation=False,
            )
            match = EXPLANATION_RE.search(raw)
            explanation = match.group(1).strip() if match else raw.strip()
            record = {
                "id": row_id,
                "prompt": base_rows.get(row_id, {}).get("prompt", ""),
                "activation_source": str(args.activation_file),
                "nla_explanation": explanation,
                "raw_nla_output": raw,
                "parsing_status": "ok" if match else "missing_explanation_tags",
                "notes": "Generated by kitft/nla-inference NLAClient.",
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            logger.info("Generated NLA explanation for %s", row_id)
    status = {
        "status": "ok",
        "stage": "nla_av_smoke_test",
        "av_checkpoint": str(args.av_checkpoint),
        "ar_checkpoint": str(args.ar_checkpoint),
        "ar_available": args.ar_checkpoint.exists(),
        "processed_count": len(activations),
        "download_performed": False,
        "server_started_by_script": False,
    }
    (out / "artifact_status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote NLA outputs to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
