#!/usr/bin/env python3
"""Extract small Qwen2.5 layer-20 activations for NLA feasibility checks."""

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
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path("/mnt/20t/xuxin/mech_interp_comparison")
MODEL_PATH = Path("/mnt/20t/xuhaoming/models/Qwen2.5-7B-Instruct")
PROMPTS_PATH = PROJECT_ROOT / "data" / "mini_prompts" / "nla_mini_safety_bias_sycophancy.jsonl"


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
    lines = [
        f"python={sys.executable}",
        f"python_version={platform.python_version()}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
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
    logger = logging.getLogger("nla_extract_qwen_activations")
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


def read_prompts(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit > 0 else rows


def chat_inputs(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(templated, return_tensors="pt", add_special_tokens=False)
    return {key: value.to(device) for key, value in encoded.items()}


def top_token_records(tokenizer: Any, log_probs: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    values, indices = torch.topk(log_probs, k=top_k)
    records: list[dict[str, Any]] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        records.append({"token_id": int(index), "token": tokenizer.decode([index]), "logprob": float(value)})
    return records


def best_variant_logprob(tokenizer: Any, log_probs: torch.Tensor, variants: list[str]) -> float | None:
    scores: list[float] = []
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            scores.append(float(log_probs[ids[0]].item()))
    return max(scores) if scores else None


def write_blocker(out: Path, reason: str) -> None:
    status = {"status": "blocked", "reason": reason, "stage": "activation_extraction"}
    (out / "artifact_status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out = make_output_dir(args.project_root, args.output_dir)
    write_common_run_files(out, args)
    logger = configure_logging(out)
    if not args.model_path.exists():
        reason = f"Base model path does not exist: {args.model_path}"
        logger.error(reason)
        write_blocker(out, reason)
        return 2
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        reason = "CUDA is not available for requested CUDA device"
        logger.error(reason)
        write_blocker(out, reason)
        return 2

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=True)
    logger.info("Loading model from %s", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    rows = read_prompts(args.prompts, args.limit)
    activations: list[np.ndarray] = []
    output_rows: list[dict[str, Any]] = []
    hidden_state_index = args.layer + 1
    logger.info("Processing %d prompts at layer %d", len(rows), args.layer)
    for row in rows:
        prompt = str(row["prompt"])
        encoded_inputs = chat_inputs(tokenizer, prompt, device)
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs.get("attention_mask")
        prompt_len = int(input_ids.shape[1])
        with torch.inference_mode():
            forward_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = forward_out.hidden_states[hidden_state_index][0, prompt_len - 1].float().cpu().numpy()
            logits = forward_out.logits[0, -1].float().cpu()
            log_probs = torch.log_softmax(logits, dim=-1)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, prompt_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        activations.append(hidden.astype(np.float32))
        output_rows.append(
            {
                "id": row["id"],
                "category": row["category"],
                "prompt": prompt,
                "model_answer": answer,
                "top_tokens": top_token_records(tokenizer, log_probs, args.top_k),
                "yes_logprob": best_variant_logprob(tokenizer, log_probs, [" yes", " Yes", "yes", "Yes"]),
                "no_logprob": best_variant_logprob(tokenizer, log_probs, [" no", " No", "no", "No"]),
                "extraction_layer": args.layer,
                "hidden_state_index": hidden_state_index,
                "extraction_position": prompt_len - 1,
                "prompt_token_count": prompt_len,
            }
        )
        logger.info("Processed prompt %s", row["id"])

    with (out / "base_model_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for item in output_rows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    activation_array = np.stack(activations, axis=0) if activations else np.zeros((0, 3584), dtype=np.float32)
    np.savez_compressed(
        out / "activations_layer20_last_token.npz",
        activations=activation_array,
        ids=np.array([str(row["id"]) for row in rows]),
        layer=np.array([args.layer], dtype=np.int32),
    )
    status = {
        "status": "ok",
        "stage": "activation_extraction",
        "base_model_path": str(args.model_path),
        "prompt_count": len(rows),
        "activation_file": str(out / "activations_layer20_last_token.npz"),
        "activation_shape": list(activation_array.shape),
        "full_activation_tensor_saved": False,
        "saved_small_demo_activation": True,
    }
    (out / "artifact_status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote outputs to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
