#!/usr/bin/env python3
"""Run NLA AV smoke inference with Transformers inputs_embeds.

Attribution: this fallback follows the NLA injection recipe documented in
kitft/nla-inference (https://github.com/kitft/nla-inference), Apache License
2.0. It does not copy the full implementation; it reuses the public checkpoint
sidecar contract for a low-throughput smoke test when SGLang is unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path("/mnt/20t/xuxin/mech_interp_comparison")
DEFAULT_AV = Path("/mnt/20t/xuhaoming/models/nla-qwen2.5-7b-L20-av")
DEFAULT_ACTIVATIONS = PROJECT_ROOT / "outputs" / "nla"
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


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
        f"transformers={getattr(sys.modules.get('transformers'), '__version__', 'unknown')}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_device_count={torch.cuda.device_count()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"cuda_device_0={torch.cuda.get_device_name(0)}")
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "TMPDIR", "CUDA_VISIBLE_DEVICES", "PYTHONPATH"]:
        lines.append(f"{key}={os.environ.get(key, '')}")
    try:
        smi = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
        lines.append("nvidia_smi_full_start")
        lines.append(smi.stdout.strip())
        lines.append("nvidia_smi_full_end")
    except FileNotFoundError:
        lines.append("nvidia_smi=not found")
    (out / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_logging(out: Path) -> logging.Logger:
    logger = logging.getLogger("nla_run_av_transformers_smoke_test")
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


def normalize_activation(vector: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm = vector.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vector / (norm / target_scale).to(vector.dtype)


def load_meta(checkpoint: Path) -> dict[str, Any]:
    meta_path = checkpoint / "nla_meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing NLA sidecar: {meta_path}")
    return yaml.safe_load(meta_path.read_text(encoding="utf-8"))


def resolve_embed_scale(checkpoint: Path) -> float:
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    model_type = getattr(text_config, "model_type", "") or ""
    if model_type in {"gemma", "gemma2", "gemma3", "gemma3_text", "t5"}:
        return float(text_config.hidden_size) ** 0.5
    return 1.0


def build_injected_embeds(
    model: Any,
    tokenizer: Any,
    checkpoint: Path,
    meta: dict[str, Any],
    activation: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = meta["tokens"]
    template = meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"]
    injection_char = tokens["injection_char"]
    content = template.format(injection_char=injection_char)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    injection_id = int(tokens["injection_token_id"])
    left_id = int(tokens["injection_left_neighbor_id"])
    right_id = int(tokens["injection_right_neighbor_id"])
    positions = (input_ids[0] == injection_id).nonzero(as_tuple=False).flatten().tolist()
    valid_positions = [
        pos for pos in positions
        if 0 < pos < input_ids.shape[1] - 1
        and int(input_ids[0, pos - 1]) == left_id
        and int(input_ids[0, pos + 1]) == right_id
    ]
    if len(valid_positions) != 1:
        raise RuntimeError(f"Expected one valid injection site, found {len(valid_positions)}")
    injection_scale = float(meta["extraction"]["injection_scale"])
    vector = torch.as_tensor(activation, dtype=torch.float32, device=device).view(1, -1)
    vector = normalize_activation(vector, injection_scale)
    with torch.inference_mode():
        embeds = model.get_input_embeddings()(input_ids) * resolve_embed_scale(checkpoint)
        embeds = embeds.float()
        embeds[0, valid_positions[0], :] = vector.to(embeds.dtype)
    return embeds, attention_mask


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--av-checkpoint", type=Path, default=DEFAULT_AV)
    parser.add_argument("--activation-file", type=Path, required=True)
    parser.add_argument("--base-outputs", type=Path, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-items", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out = make_output_dir(args.project_root, args.output_dir)
    write_common_run_files(out, args)
    logger = configure_logging(out)
    if not args.av_checkpoint.exists():
        raise FileNotFoundError(f"Missing AV checkpoint: {args.av_checkpoint}")
    if not args.activation_file.exists():
        raise FileNotFoundError(f"Missing activation file: {args.activation_file}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta = load_meta(args.av_checkpoint)
    data = np.load(args.activation_file, allow_pickle=False)
    activations = data["activations"][: args.max_items]
    ids = [str(item) for item in data["ids"][: args.max_items]]
    base_rows = {str(row.get("id")): row for row in read_jsonl(args.base_outputs)} if args.base_outputs else {}

    logger.info("Loading AV tokenizer from %s", args.av_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(str(args.av_checkpoint), local_files_only=True, trust_remote_code=True)
    logger.info("Loading AV model from %s", args.av_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.av_checkpoint),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    records: list[dict[str, Any]] = []
    for row_id, activation in zip(ids, activations):
        embeds, attention_mask = build_injected_embeds(model, tokenizer, args.av_checkpoint, meta, activation, device)
        with torch.inference_mode():
            generated = model.generate(
                inputs_embeds=embeds.to(model.dtype),
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(generated[0], skip_special_tokens=False)
        match = EXPLANATION_RE.search(raw)
        explanation = match.group(1).strip() if match else raw.strip()
        records.append(
            {
                "id": row_id,
                "prompt": base_rows.get(row_id, {}).get("prompt", ""),
                "activation_source": str(args.activation_file),
                "nla_explanation": explanation,
                "raw_nla_output": raw,
                "parsing_status": "ok" if match else "missing_explanation_tags",
                "notes": "Generated with Transformers inputs_embeds fallback; SGLang was unavailable on this CUDA toolchain.",
            }
        )
        logger.info("Generated fallback AV explanation for %s", row_id)
    with (out / "nla_explanations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    status = {
        "status": "ok",
        "stage": "nla_av_transformers_smoke_test",
        "av_checkpoint": str(args.av_checkpoint),
        "processed_count": len(records),
        "fallback": "transformers_inputs_embeds",
        "sglang_status": "blocked_by_cuda_11_8_flashinfer_jit",
    }
    (out / "artifact_status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote fallback AV outputs to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
