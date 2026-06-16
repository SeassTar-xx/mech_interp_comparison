# License Notes

This project is currently a clean scaffold for comparing existing interpretability methods. No external repository code has been copied into the project at day 0.

When external code or artifacts are added, record the following before running or publishing derived work:

- source repository or artifact URL;
- commit hash, release tag, or artifact version;
- original license;
- files copied or adapted;
- local modifications;
- required attribution;
- usage restrictions;
- citation or acknowledgement text.

If a dependency, model, checkpoint, or dataset has unclear licensing, treat it as blocked until its source and terms are reviewed.

## Third-Party Code: kitft/nla-inference

- Source: `https://github.com/kitft/nla-inference`
- Local path: `/mnt/20t/xuxin/mech_interp_comparison/repos/nla-inference`
- Commit inspected: `38b802a33d1d317f21b6825a9116f388c2141f86`
- License: Apache License 2.0
- Use in this project: inference-only reference implementation and wrapper dependency for Natural Language Autoencoders.
- Relevant files inspected: `README.md`, `LICENSE`, `nla_inference.py`, `examples/qwen7b_layer20_step4200.txt`
- Notes: the project wrapper imports or calls this repository instead of copying the full implementation. If any code is copied or modified later, the copied file must include source attribution at the top and retain applicable Apache-2.0 notices.

Released NLA checkpoint names documented by this repository:

- Qwen2.5-7B layer-20 AV: `kitft/nla-qwen2.5-7b-L20-av`
- Qwen2.5-7B layer-20 AR: `kitft/nla-qwen2.5-7b-L20-ar`

These checkpoint repositories were not downloaded during the day-1 setup unless explicitly approved later.

## Downloaded Third-Party Checkpoints

Approved downloads completed on 2026-06-16:

- `kitft/nla-qwen2.5-7b-L20-av`
  - Local path: `/mnt/20t/xuhaoming/models/nla-qwen2.5-7b-L20-av`
  - License reported by Hugging Face metadata: Apache-2.0
  - Approximate local size: 15G
- `kitft/nla-qwen2.5-7b-L20-ar`
  - Local path: `/mnt/20t/xuhaoming/models/nla-qwen2.5-7b-L20-ar`
  - License reported by Hugging Face metadata: Apache-2.0
  - Approximate local size: 11G

These checkpoints are used only for inference and reconstruction scoring in the comparison project. They must not be treated as project-authored artifacts.


## Local Modifications to Third-Party NLA Code

Modified on 2026-06-16 for server-local inference compatibility:

- File: `/mnt/20t/xuxin/mech_interp_comparison/repos/nla-inference/nla_inference.py`
- Source repository: `https://github.com/kitft/nla-inference`
- Source commit: `38b802a33d1d317f21b6825a9116f388c2141f86`
- License: Apache License 2.0
- Local change: added `_as_1d_token_ids(...)` and used it at both `apply_chat_template(..., tokenize=True)` call sites so the code handles Transformers versions that return a `BatchEncoding` or dict instead of a plain token-id list.
- Purpose: inference-only compatibility for the NLA AV `input_embeds` path. No model weights or checkpoints were modified.

## Third-Party Code: goodfire-ai/param-decomp

- Source: `https://github.com/goodfire-ai/param-decomp`
- Local path: `/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp`
- Commit inspected: `1130c1fe66e7d526aadd169c2d2293bfa374863a`
- License: MIT License
- Use in this project: inspection-only VPD feasibility check, artifact-loading analysis, and future comparison entry scaffolding.
- Relevant files inspected: `README.md`, `LICENSE`, `pyproject.toml`, `param_decomp_lab/pyproject.toml`, `param_decomp_lab/infra/run_files.py`, `param_decomp_lab/infra/settings.py`, `param_decomp_lab/experiments/lm/run.py`, `param_decomp_lab/component_model_io.py`, `papers/Interpreting_Language_Model_Parameters.md`.
- Local modifications: none.
- Download status: code repository cloned only; no Goodfire checkpoint, W&B run files, or decomposition artifacts were downloaded during the day-1 VPD inspection.
- Notes: the repository requires Python `==3.13.*`; the current checked research environment is Python 3.10, so only safe static inspection and top-level import checks were performed.

Canonical VPD run referenced by the repository:

- W&B run: `goodfire/spd/runs/s-55ea3f9b`
- Status: public web page was reachable during inspection, but direct artifact/checkpoint access was not confirmed and no download was attempted.

## Downloaded Third-Party VPD Artifacts

Approved downloads completed on 2026-06-16 for the Goodfire VPD feasibility check:

- `goodfire/spd/runs/s-55ea3f9b/final_config.yaml`
  - Local path: `/mnt/20t/xuxin/mech_interp_comparison/outputs/vpd/param_decomp_out/runs/s-55ea3f9b/final_config.yaml`
  - Size: 3,087 bytes
- `goodfire/spd/runs/s-55ea3f9b/model_400000.pth`
  - Local path: `/mnt/20t/xuxin/mech_interp_comparison/outputs/vpd/param_decomp_out/runs/s-55ea3f9b/model_400000.pth`
  - Size: 2,908,595,703 bytes
- `goodfire/spd/runs/t-9d2b8f02/final_config.yaml`
  - Local path: `/mnt/20t/xuxin/mech_interp_comparison/outputs/vpd/param_decomp_out/runs/spd-t-9d2b8f02/final_config.yaml`
  - Size: 1,318 bytes
- `goodfire/spd/runs/t-9d2b8f02/model_config.yaml`
  - Local path: `/mnt/20t/xuxin/mech_interp_comparison/outputs/vpd/param_decomp_out/runs/spd-t-9d2b8f02/model_config.yaml`
  - Size: 323 bytes
- `goodfire/spd/runs/t-9d2b8f02/model_step_99999.pt`
  - Local path: `/mnt/20t/xuxin/mech_interp_comparison/outputs/vpd/param_decomp_out/runs/spd-t-9d2b8f02/model_step_99999.pt`
  - Size: 267,736,421 bytes

These artifacts are used only for feasibility smoke testing and comparison scaffolding. They are not project-authored artifacts. The local shim under `outputs/vpd/param_decomp_out/localized_runs/` only rewrites the VPD config to point at the locally downloaded target checkpoint, avoiding W&B API calls; it does not modify model weights or third-party source code.

## Third-Party Code: goodfire-ai/param-decomp vpd-paper tag

- Source: `https://github.com/goodfire-ai/param-decomp`
- Local worktree: `/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp-vpd-paper`
- Tag: `vpd-paper`
- Commit inspected: `74146b555f3eb3cd2bc639c3789fd3b7a099094a`
- License: MIT License
- Use in this project: paper-release artifact loading and smoke testing.
- Local source modifications: none.

