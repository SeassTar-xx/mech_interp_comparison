# Mechanistic Interpretability Comparison

This project builds a lightweight comparison framework for existing interpretability methods inspired by "Prediction, Explanation, or Over-interpretation?".

The day-1 scope is intentionally conservative:

- compare existing explanation methods without retraining;
- focus first on the VPD and NLA lines;
- check availability, reproducibility, and integration feasibility;
- produce structured reports and reusable comparison scaffolding.

## Hard Constraints

- Do not train, fine-tune, or retrain any model, VPD component, or NLA component.
- Do not download large models unless the local model root has been checked and explicit approval has been given.
- Keep all project files, data, logs, outputs, caches, and temporary files under:

```text
/mnt/20t/xuxin/mech_interp_comparison/
```

- Prefer existing models and checkpoints from:

```text
/mnt/20t/xuhaoming/models/
```

- Do not store code, data, models, caches, logs, temporary files, or outputs in the server home directory.
- Do not use `sudo`.
- Do not print tokens, secrets, private keys, or credentials.
- Code comments must be written in English only.

## Directory Policy

The project root is:

```text
/mnt/20t/xuxin/mech_interp_comparison/
```

The repository is organized so that all local artifacts remain inside the project root:

- `configs/`: reproducible configuration files.
- `data/mini_prompts/`: small prompt fixtures only.
- `repos/`: external repositories or checked-out method code, with attribution and license notes.
- `src/mech_interp_comparison/`: local comparison framework code.
- `outputs/`: generated outputs from approved runs.
- `logs/`: logs from scripts and experiments.
- `reports/`: day-by-day notes and environment reports.
- `tmp/`: temporary files.
- `.cache/`: Hugging Face, PyTorch, XDG, and related caches.

## Model Policy

Models and checkpoints should be resolved from:

```text
/mnt/20t/xuhaoming/models/
```

The project should treat that directory as the first lookup location. It should not write to that model root unless a later task explicitly approves it.

## External Code, Licenses, and Attribution

If code is copied or adapted from an external repository:

- keep original license headers when required;
- add a source note at the top of copied or adapted files;
- document the source repository, commit, license, and modifications in `LICENSE_NOTES.md`;
- include acknowledgements in reports and derived artifacts.

## Hugging Face Upload Policy

If any dataset, model, or artifact is uploaded to Hugging Face, the upload must include a clear README, model card, or dataset card covering:

- source and provenance;
- license;
- intended use;
- acknowledgements;
- limitations and restrictions;
- whether the artifact is derived from external repositories, checkpoints, or datasets.

No upload is part of the day-0 setup.

## Day-0 Status

This repository skeleton only sets up paths, cache policy, logging utilities, and placeholder modules for VPD, NLA, and evaluation. It does not run model inference, download models, or train anything.

## NLA Day-1 Feasibility Track

The first NLA target is the released Qwen2.5-7B layer-20 pair:

- AV: `kitft/nla-qwen2.5-7b-L20-av`
- AR: `kitft/nla-qwen2.5-7b-L20-ar`
- Base model: `/mnt/20t/xuhaoming/models/Qwen2.5-7B-Instruct`
- Extraction layer: `20`
- Qwen2.5-7B hidden size: `3584`
- AV injection scale: `150`

The inference code is kept as an external repository under `repos/nla-inference` when available. It is Apache-2.0 licensed and should be treated as third-party code. Do not copy or modify external source files without recording attribution and license details in `LICENSE_NOTES.md`.

NLA smoke-test scripts:

- `scripts/nla_check_artifacts.py`
- `scripts/nla_prepare_mini_prompts.py`
- `scripts/nla_extract_qwen_activations.py`
- `scripts/nla_run_av_smoke_test.py`
- `scripts/setup_nla_sglang_env.sh`
- `scripts/nla_make_day1_report.py`

Run setup first so no cache or temporary files land in home:

```bash
source /mnt/20t/xuxin/mech_interp_comparison/scripts/setup_env.sh
export PYTHONPATH=/mnt/20t/xuxin/mech_interp_comparison/src
```

If AV inference is run through SGLang, launch it with `--disable-radix-cache` because NLA uses `input_embeds`. Stop the server after the smoke test so it does not keep a GPU occupied.

On this server, SGLang 0.5.13.post1 is installed under `env/nla_pydeps` and should be launched after `source scripts/setup_nla_sglang_env.sh`. That project-local setup points SGLang and FlashInfer JIT to the project CUDA 13 toolchain, keeps caches under the project root, and avoids the system `/usr/local/cuda-11.8` path. The Transformers `inputs_embeds` runner remains available as a low-throughput fallback.

## VPD Day-1 Feasibility Track

The VPD target is Goodfire's adVersarial Parameter Decomposition implementation from `goodfire-ai/param-decomp`. The local checkout is kept as third-party code under:

```text
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp
```

Inspected source state:

- Source: `https://github.com/goodfire-ai/param-decomp`
- Commit: `1130c1fe66e7d526aadd169c2d2293bfa374863a`
- License: MIT License
- Canonical run referenced by the project README: `goodfire/spd/runs/s-55ea3f9b`
- Package imports: `param_decomp`, `param_decomp_lab`
- Required Python version from `pyproject.toml`: `==3.13.*`

Day-1 VPD work is inspection-only. The current server environment uses Python 3.10, so the package was not installed or run beyond safe top-level import checks. No VPD model checkpoint, decomposition artifact, or W&B artifact was downloaded.

Potential training or large-job entry points must not be run during feasibility inspection without explicit approval. This includes `pd-lm`, `pd-pretrain`, `pd-clustering`, `pd-cluster-harvest`, `pd-cluster-merge`, `pd-harvest`, `pd-autointerp`, `pd-attributions`, `pd-postprocess`, and `pd-graph-interp` when they are pointed at full LM configs or canonical runs.

If VPD artifact loading is attempted later, run setup first and keep Param Decomp outputs inside the project root:

```bash
source /mnt/20t/xuxin/mech_interp_comparison/scripts/setup_env.sh
export PARAM_DECOMP_OUT_DIR=/mnt/20t/xuxin/mech_interp_comparison/outputs/vpd/param_decomp_out
```

Current blocker reports are written to:

- `reports/vpd_local_artifact_report.md`
- `reports/vpd_repo_inspection_report.md`
- `reports/day1_vpd_blocker_report.md`
- `reports/day1_vpd_smoke_test_report.md`

### VPD Day-1 Updated Result

VPD feasibility moved from blocked to artifact-load smoke-test success after creating a project-local Python 3.13 environment and checking out the `vpd-paper` tag in a separate worktree.

- Python env: `/mnt/20t/xuxin/mech_interp_comparison/env/vpd_py313`
- Main repo checkout: `repos/param-decomp` at `1130c1fe66e7d526aadd169c2d2293bfa374863a`
- Paper-tag worktree: `repos/param-decomp-vpd-paper` at `74146b555f3eb3cd2bc639c3789fd3b7a099094a` (`vpd-paper`)
- Canonical VPD artifact: `goodfire/spd/runs/s-55ea3f9b`
- Target pretrain artifact required by the VPD config: `goodfire/spd/runs/t-9d2b8f02`
- Smoke output: `outputs/vpd/20260616_191600_smoke/`

Downloaded VPD artifacts are stored under `outputs/vpd/param_decomp_out/`, not under home. The paper-tag loader can instantiate `ComponentModel` from the localized artifact layout without W&B API access. No training, fine-tuning, or full VPD training was run.

