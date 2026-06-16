#!/usr/bin/env python3
"""Check local NLA and Qwen artifacts without loading model weights."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path("/mnt/20t/xuxin/mech_interp_comparison")
MODEL_ROOT = Path("/mnt/20t/xuhaoming/models")
DEFAULT_KEYWORDS = (
    "Qwen2.5-7B-Instruct",
    "Qwen2.5",
    "qwen2.5",
    "nla",
    "nla-qwen2.5-7b-L20-av",
    "nla-qwen2.5-7b-L20-ar",
    "kitft",
    "activation verbalizer",
    "activation reconstructor",
)


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
    (out / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in sys.argv) + "\n", encoding="utf-8"
    )
    env_keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "TMPDIR",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
    ]
    lines = [
        f"python={sys.executable}",
        f"python_version={platform.python_version()}",
        f"platform={platform.platform()}",
    ]
    for key in env_keys:
        lines.append(f"{key}={os.environ.get(key, '')}")
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        lines.append("nvidia_smi=" + smi.stdout.strip().replace("\n", " | "))
    except FileNotFoundError:
        lines.append("nvidia_smi=not found")
    (out / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_logging(out: Path) -> logging.Logger:
    logger = logging.getLogger("nla_check_artifacts")
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


def path_matches(model_root: Path, keyword: str, max_depth: int) -> list[str]:
    keyword_lower = keyword.lower()
    root_parts = len(model_root.resolve().parts)
    matches: list[str] = []
    for path in model_root.rglob("*"):
        try:
            rel_depth = len(path.resolve().parts) - root_parts
        except FileNotFoundError:
            continue
        if rel_depth > max_depth:
            continue
        if keyword_lower in str(path).lower():
            matches.append(str(path))
    return sorted(matches)


def assess_hf_dir(path: Path) -> dict[str, object]:
    required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    present = {name: (path / name).exists() for name in required}
    shards = sorted(path.glob("*.safetensors"))
    index_exists = (path / "model.safetensors.index.json").exists()
    likely_complete = path.is_dir() and all(present.values()) and bool(shards)
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "required_files": present,
        "safetensors_count": len(shards),
        "has_safetensors_index": index_exists,
        "likely_complete": likely_complete,
    }


def render_markdown(status: dict[str, object]) -> str:
    lines: list[str] = [
        "# NLA Local Artifact Report",
        "",
        f"Date: {status['created_at']}",
        "",
        f"Project root: `{status['project_root']}`",
        "",
        f"Model root: `{status['model_root']}`",
        "",
        "## Search Scope",
        "",
        "Searched local filesystem metadata under the model root. No model weights were loaded, downloaded, or modified.",
        "",
        "## Keyword Search Results",
        "",
    ]
    keyword_results = status["keyword_results"]
    assert isinstance(keyword_results, dict)
    for keyword, matches in keyword_results.items():
        lines.extend([f"### `{keyword}`", ""])
        if matches:
            lines.append("```text")
            lines.extend(str(item) for item in matches[:120])
            if len(matches) > 120:
                lines.append(f"... truncated, total matches: {len(matches)}")
            lines.append("```")
        else:
            lines.append("No path-name matches found.")
        lines.append("")
    lines.extend(["## Artifact Assessment", ""])
    base = status["base_model"]
    assert isinstance(base, dict)
    if base["exists"]:
        lines.append(f"- Base model found: `{base['path']}`.")
        lines.append(f"- Base model safetensors shard count: `{base['safetensors_count']}`.")
        lines.append(f"- Base model completeness: {'likely complete' if base['likely_complete'] else 'possibly incomplete'}.")
    else:
        lines.append(f"- Base model missing: `{base['path']}`.")
    av = status["nla_av"]
    ar = status["nla_ar"]
    assert isinstance(av, dict) and isinstance(ar, dict)
    lines.append(f"- NLA AV checkpoint status: {av['status']}.")
    lines.append(f"- NLA AR checkpoint status: {ar['status']}.")
    lines.append(f"- Missing AV: `{str(not av['found']).lower()}`.")
    lines.append(f"- Missing AR: `{str(not ar['found']).lower()}`.")
    lines.append(f"- External download needed: `{str(status['external_download_needed']).lower()}`.")
    lines.extend(["", "## Notes", ""])
    lines.append("Released Qwen2.5-7B NLA AV/AR checkpoint names are documented by `kitft/nla-inference` as `kitft/nla-qwen2.5-7b-L20-av` and `kitft/nla-qwen2.5-7b-L20-ar`.")
    lines.append("Do not download these checkpoint repositories without explicit approval, because they may contain large model weights.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--report-path", type=Path, default=PROJECT_ROOT / "reports" / "nla_local_artifact_report.md")
    args = parser.parse_args()

    out = make_output_dir(args.project_root, args.output_dir)
    write_common_run_files(out, args)
    logger = configure_logging(out)
    logger.info("Checking NLA artifacts under %s", args.model_root)

    keyword_results = {
        keyword: path_matches(args.model_root, keyword, args.max_depth)
        for keyword in DEFAULT_KEYWORDS
    }
    base_model = assess_hf_dir(args.model_root / "Qwen2.5-7B-Instruct")
    av_candidates = path_matches(args.model_root, "nla-qwen2.5-7b-L20-av", args.max_depth)
    ar_candidates = path_matches(args.model_root, "nla-qwen2.5-7b-L20-ar", args.max_depth)
    av_status = {
        "expected_hf_repo": "kitft/nla-qwen2.5-7b-L20-av",
        "found": bool(av_candidates),
        "candidate_paths": av_candidates,
        "status": "candidate path-name matches found" if av_candidates else "not found by local path-name search",
    }
    ar_status = {
        "expected_hf_repo": "kitft/nla-qwen2.5-7b-L20-ar",
        "found": bool(ar_candidates),
        "candidate_paths": ar_candidates,
        "status": "candidate path-name matches found" if ar_candidates else "not found by local path-name search",
    }
    status = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(args.project_root),
        "model_root": str(args.model_root),
        "keyword_results": keyword_results,
        "base_model": base_model,
        "nla_av": av_status,
        "nla_ar": ar_status,
        "external_download_needed": not (av_status["found"] and ar_status["found"]),
        "download_performed": False,
    }
    (out / "artifact_status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    report = render_markdown(status)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")
    (out / "nla_local_artifact_report.md").write_text(report, encoding="utf-8")
    logger.info("Wrote %s", args.report_path)
    logger.info("Wrote %s", out / "artifact_status.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
