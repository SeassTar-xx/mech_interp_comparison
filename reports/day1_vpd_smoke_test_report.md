# Day 1 VPD Smoke Test Report

Date: 2026-06-16

This is an artifact inspection and feasibility smoke test only. It is not a scientific reproduction.

## Current Status

VPD is now usable for a canonical small-model comparison entry. The released Goodfire artifact can be downloaded, parsed with the `vpd-paper` tag, and loaded as a `ComponentModel` on CPU. No VPD training, fine-tuning, or full LM training was run.

## Repositories And Environment

- Main repo checkout: `/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp` at `1130c1fe66e7d526aadd169c2d2293bfa374863a`.
- Paper-tag worktree: `/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp-vpd-paper` at `74146b555f3eb3cd2bc639c3789fd3b7a099094a`.
- License: MIT License.
- Python env: `/mnt/20t/xuxin/mech_interp_comparison/env/vpd_py313`, Python 3.13.14.
- Key compatibility pins: `torch==2.8.0+cpu`, `torchvision==0.23.0+cpu`, `transformers==4.57.6`, `huggingface_hub==0.36.2`, `wandb==0.20.1`.

## Downloaded Artifacts

- VPD config: `final_config.yaml`, 3,087 bytes.
- VPD checkpoint: `model_400000.pth`, 2,908,595,703 bytes.
- Target pretrain config: `final_config.yaml`, 1,318 bytes.
- Target model config: `model_config.yaml`, 323 bytes.
- Target pretrain checkpoint: `model_step_99999.pt`, 267,736,421 bytes.

All downloaded artifacts are under `/mnt/20t/xuxin/mech_interp_comparison/outputs/vpd/param_decomp_out/`. No VPD artifact was stored in home.

## Artifact Load Smoke Test

- `ParamDecompRunInfo` parse: `True`.
- `ComponentModel` load: `True`.
- Loaded target type: `LlamaSimpleMLP`.
- State dict key count: `159`.
- Component model load elapsed: `13.39` seconds.

A local config shim was created under `outputs/vpd/param_decomp_out/localized_runs/` to point the VPD config at the locally downloaded target checkpoint. This avoids W&B API calls and does not modify source code or weights.

## Canonical Prompt Smoke Test

A CPU-only single-step next-token check was run on three short prompts. This is not a formal reproduction of attribution graphs.

- `The princess lost her crown.`: top next tokens included newline, `She`, `The`, `Her`, and `He`.
- `The prince lost his crown.`: top next tokens included newline, `He`, `The`, `His`, and `In`.
- `< u , v `: top next tokens were delimiter/newline-like tokens; this needs closer prompt matching before interpreting.

Raw output: `/mnt/20t/xuxin/mech_interp_comparison/outputs/vpd/20260616_191600_smoke/vpd_pronoun_prediction_smoke.json`.

## Comparison Entry Implications

VPD can enter the comparison framework as a method-specific canonical-case entry rather than a same-base-model safety/bias/sycophancy benchmark. Its useful evidence type is parameter-level: rank-one subcomponents, causal-importance network outputs, ablation behavior, and attribution graph metadata. The current smoke test verifies artifact availability and loader compatibility; it does not yet extract top causal components or attribution graphs.

## Remaining Blockers

- Need a focused script to compute causal-importance outputs for selected prompts and format top subcomponents.
- Cluster mapping files referenced by the app registry point to Goodfire-internal `/mnt/polished-lake/...` paths and were not available locally.
- Full attribution graph reproduction likely needs additional postprocess/harvest artifacts or a small custom extraction script.

## Files

- Artifact status: `/mnt/20t/xuxin/mech_interp_comparison/outputs/vpd/20260616_191600_smoke/artifact_status.json`
- Component load report: `/mnt/20t/xuxin/mech_interp_comparison/reports/vpd_component_model_load_smoke_test.json`
- RunInfo report: `/mnt/20t/xuxin/mech_interp_comparison/reports/vpd_paper_tag_runinfo_smoke_test.json`
- Checkpoint structure report: `/mnt/20t/xuxin/mech_interp_comparison/reports/vpd_checkpoint_structure_smoke_test.json`
