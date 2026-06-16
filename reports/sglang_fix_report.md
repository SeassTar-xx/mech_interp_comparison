# SGLang Fix Report for NLA Smoke Test

Date: 2026-06-16

## Result

SGLang was repaired enough for the NLA AV feasibility run. The server reached `/health` successfully and generated AV explanations through `kitft/nla-inference` using `input_embeds`.

- Run directory: `/mnt/20t/xuxin/mech_interp_comparison/outputs/nla/20260616_160604_sglang_av`
- SGLang status: `ok`
- AV processed prompts: `16`
- AR processed prompts: `16`
- AR `mse_nrm` mean/min/max: `0.215894` / `0.146063` / `0.310152`
- AR cosine mean/min/max: `0.892053` / `0.844924` / `0.926969`
- Server stopped after run: `True`

## Fixes Applied

- Added `scripts/setup_nla_sglang_env.sh` to force project-local runtime paths.
- Pointed SGLang/FlashInfer JIT to `/mnt/20t/xuxin/mech_interp_comparison/env/nla_pydeps/nvidia/cu13` instead of system `/usr/local/cuda-11.8`.
- Exported `NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` for FlashInfer CCCL compatibility checks.
- Added project-local CUDA library compatibility links under `env/nla_pydeps/nvidia/cu13`:
  - `lib64 -> lib`
  - `lib/libcudart.so -> libcudart.so.13`
- Added FlashInfer vendored CCCL include paths through `CPATH` and `CPLUS_INCLUDE_PATH` so SGLang tvm-ffi fused rope JIT can find `nv/target`.
- Patched local `repos/nla-inference/nla_inference.py` to normalize `apply_chat_template(..., tokenize=True)` output across Transformers versions.

## Launch Command

See `/mnt/20t/xuxin/mech_interp_comparison/outputs/nla/20260616_160604_sglang_av/sglang_command.txt` for the exact command. Required details:

```bash
source /mnt/20t/xuxin/mech_interp_comparison/scripts/setup_nla_sglang_env.sh >/dev/null
python -m sglang.launch_server \
  --model-path /mnt/20t/xuhaoming/models/nla-qwen2.5-7b-L20-av \
  --host 127.0.0.1 \
  --port 30129 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --attention-backend triton \
  --sampling-backend pytorch \
  --mem-fraction-static 0.65 \
  --context-length 512 \
  --trust-remote-code
```

## Outputs

- Base model outputs: `/mnt/20t/xuxin/mech_interp_comparison/outputs/nla/20260616_160604_sglang_av/base_model_outputs.jsonl`
- NLA AV explanations: `/mnt/20t/xuxin/mech_interp_comparison/outputs/nla/20260616_160604_sglang_av/nla_explanations.jsonl`
- NLA AR scores: `/mnt/20t/xuxin/mech_interp_comparison/outputs/nla/20260616_160604_sglang_av/nla_reconstruction_scores.jsonl`
- SGLang log: `/mnt/20t/xuxin/mech_interp_comparison/outputs/nla/20260616_160604_sglang_av/sglang_server.log`
- Day1 report: `/mnt/20t/xuxin/mech_interp_comparison/reports/day1_nla_sglang_smoke_test_report.md`

## Remaining Notes

This is an inference feasibility trial only. No model training, fine-tuning, or checkpoint modification was performed. The SGLang fixes are local environment and dependency compatibility fixes under the project directory.
