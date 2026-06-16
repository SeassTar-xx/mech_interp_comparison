#!/usr/bin/env python3
"""Create a day-1 NLA feasibility report from the latest smoke-test outputs."""

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
from typing import Any


PROJECT_ROOT = Path("/mnt/20t/xuxin/mech_interp_comparison")


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
    logger = logging.getLogger("nla_make_day1_report")
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_latest_with_file(root: Path, filename: str) -> Path | None:
    candidates = [path.parent for path in root.glob(f"*/{filename}") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def render_report(args: argparse.Namespace, extraction_dir: Path | None, av_dir: Path | None) -> str:
    extraction_status = {}
    if extraction_dir:
        extraction_status = read_json(extraction_dir / "artifact_status_extraction.json")
        if not extraction_status:
            extraction_status = read_json(extraction_dir / "artifact_status.json")
    av_status = read_json(av_dir / "artifact_status.json") if av_dir else {}
    ar_status = read_json(av_dir / "artifact_status_ar.json") if av_dir else {}
    base_rows = read_jsonl(extraction_dir / "base_model_outputs.jsonl") if extraction_dir else []
    nla_rows = read_jsonl(av_dir / "nla_explanations.jsonl") if av_dir else []
    ar_rows = read_jsonl(av_dir / "nla_reconstruction_scores.jsonl") if av_dir else []
    ar_by_id = {str(row.get("id")): row for row in ar_rows}
    sample_count = min(3, max(len(base_rows), len(nla_rows)))
    lines = [
        "# Day 1 NLA Smoke Test Report",
        "",
        f"Date: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This is a feasibility trial only. It should not be treated as a scientific conclusion.",
        "",
        "## Local Artifact Status",
        "",
        f"- Base model path: `{args.base_model_path}`",
        f"- NLA AV expected checkpoint: `{args.av_checkpoint}`",
        f"- NLA AR expected checkpoint: `{args.ar_checkpoint}`",
        f"- Extraction output dir: `{extraction_dir or 'not available'}`",
        f"- AV output dir: `{av_dir or 'not available'}`",
        f"- Extraction status: `{extraction_status.get('status', 'missing')}`",
        f"- AV status: `{av_status.get('status', 'missing')}`",
        f"- AV method: `{av_status.get('av_method') or av_status.get('fallback', 'sglang_or_unknown')}`",
        f"- SGLang status: `{av_status.get('sglang_status', 'not recorded')}`",
        f"- AV blocker: {av_status.get('reason', 'none recorded')}",
        f"- AR status: `{ar_status.get('status', 'missing')}`",
        "",
        "## GPU and Environment",
        "",
    ]
    env_text = ""
    if extraction_dir and (extraction_dir / "environment_extraction.txt").exists():
        env_text = (extraction_dir / "environment_extraction.txt").read_text(encoding="utf-8")
    elif extraction_dir and (extraction_dir / "environment.txt").exists():
        env_text = (extraction_dir / "environment.txt").read_text(encoding="utf-8")
    elif av_dir and (av_dir / "environment.txt").exists():
        env_text = (av_dir / "environment.txt").read_text(encoding="utf-8")
    lines.append("```text")
    lines.extend(env_text.splitlines()[:80] if env_text else ["No environment file found."])
    lines.append("```")
    lines.extend(["", "## Prompt Processing", ""])
    lines.append(f"- Base model prompt count: `{len(base_rows)}`")
    lines.append(f"- NLA explanation count: `{len(nla_rows)}`")
    lines.append(f"- Successful prompt count for AV: `{len([row for row in nla_rows if row.get('parsing_status') == 'ok'])}`")
    lines.append(f"- AR reconstruction score count: `{len(ar_rows)}`")
    lines.extend(["", "## Samples", ""])
    if sample_count == 0:
        lines.append("No completed NLA explanation samples are available yet.")
    for index in range(sample_count):
        base = base_rows[index] if index < len(base_rows) else {}
        nla = nla_rows[index] if index < len(nla_rows) else {}
        sample_id = str(base.get("id") or nla.get("id", ""))
        ar = ar_by_id.get(sample_id, {})
        lines.extend(
            [
                f"### Sample {index + 1}",
                "",
                f"- id: `{sample_id}`",
                f"- category: `{base.get('category', '')}`",
                f"- prompt: {base.get('prompt') or nla.get('prompt', '')}",
                f"- base model output: {base.get('model_answer', 'not available')}",
                f"- NLA explanation: {nla.get('nla_explanation', 'not available')}",
                f"- AR reconstruction: {ar.get('status', 'not available')}, mse_nrm={ar.get('mse_nrm', 'not available')}, cos={ar.get('cosine_similarity', 'not available')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Preliminary Qualitative Check",
            "",
            "The AV explanations for the 3-prompt smoke test are available, but this remains a feasibility trial and not a scientific conclusion.",
            "The visible examples mostly describe structured Q&A, refusal, safety, and medical-advice answer patterns, which is directionally relevant to the selected safety/refusal prompts.",
            "The mini prompt set also contains bias/stereotype, sycophancy, and evaluation-awareness categories for the next expanded run.",
            "",
            "## Blockers",
            "",
        ]
    )
    if av_status.get("status") == "blocked":
        lines.append(f"- {av_status.get('reason')}")
    if av_status.get("sglang_status") == "blocked_by_cuda_11_8_flashinfer_jit":
        lines.append("- SGLang serving was blocked by flashinfer JIT against `/usr/local/cuda-11.8`; the smoke test used a Transformers `inputs_embeds` fallback instead.")
    if not nla_rows:
        lines.append("- No NLA AV explanations are available yet.")
    if not Path(args.av_checkpoint).exists():
        lines.append("- Local NLA AV checkpoint is missing; external download requires explicit approval.")
    if not Path(args.ar_checkpoint).exists():
        lines.append("- Local NLA AR checkpoint is missing; round-trip reconstruction is skipped.")
    if ar_rows and all(row.get("status") == "ok" for row in ar_rows):
        lines.append("- No AR blocker for the 3-prompt smoke test.")
    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "- Investigate a clean SGLang runtime with CUDA 12+ toolchain or a compatible flashinfer build before scaling throughput.",
            "- Expand from 3 prompts to the full 16-prompt mini set using the Transformers fallback or a fixed SGLang server.",
            "- Add a comparison table for base answer, AV explanation, AR mse_nrm, and qualitative category tags.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--extraction-dir", type=Path, default=None)
    parser.add_argument("--av-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=PROJECT_ROOT / "reports" / "day1_nla_smoke_test_report.md")
    parser.add_argument("--base-model-path", default="/mnt/20t/xuhaoming/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--av-checkpoint", default="/mnt/20t/xuhaoming/models/nla-qwen2.5-7b-L20-av")
    parser.add_argument("--ar-checkpoint", default="/mnt/20t/xuhaoming/models/nla-qwen2.5-7b-L20-ar")
    args = parser.parse_args()

    out = make_output_dir(args.project_root, args.output_dir)
    write_common_run_files(out, args)
    logger = configure_logging(out)
    output_root = args.project_root / "outputs" / "nla"
    extraction_dir = args.extraction_dir or find_latest_with_file(output_root, "base_model_outputs.jsonl")
    av_dir = args.av_dir or find_latest_with_file(output_root, "nla_explanations.jsonl")
    report = render_report(args, extraction_dir, av_dir)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")
    (out / "day1_nla_smoke_test_report.md").write_text(report, encoding="utf-8")
    logger.info("Wrote %s", args.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
