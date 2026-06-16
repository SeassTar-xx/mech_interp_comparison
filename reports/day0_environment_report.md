# Day 0 Environment Report

Date: 2026-06-16  
Host alias: `Mistral-rev`  
Project root: `/mnt/20t/xuxin/mech_interp_comparison`

## Scope

This report records environment availability and the day-0 project scaffold only. No model inference, model download, fine-tuning, training, or explanation run was performed.

## Basic Server Check

| Item | Result |
| --- | --- |
| SSH login `pwd` | `/home/xuhaoming` |
| `hostname` | `mistral` |
| `whoami` | `xuhaoming` |
| `git --version` | `git version 2.34.1` |

The SSH session starts in `/home/xuhaoming`, but the project was created only under `/mnt/20t/xuxin/mech_interp_comparison`.

## Storage

Command: `df -h /mnt/20t`

```text
文件系统        大小  已用  可用 已用% 挂载点
/dev/sdb1        19T  7.5T  9.8T   44% /mnt/20t
```

`/mnt/20t/` is available and has about 9.8T free at the time of this check.

## GPU

Command: `nvidia-smi`

Detected GPUs:

| GPU | Name | Memory | Utilization | Notes |
| --- | --- | --- | --- | --- |
| 0 | NVIDIA A100-PCIE-40GB | 10 MiB / 40960 MiB | 0% | Only Xorg shown |
| 1 | NVIDIA A100-PCIE-40GB | 13 MiB / 40960 MiB | 0% | Only Xorg shown |

Driver and CUDA reported by `nvidia-smi`:

- NVIDIA-SMI: `590.48.01`
- Driver Version: `590.48.01`
- CUDA Version: `13.1`

## Python, Conda, and uv

| Check | Result |
| --- | --- |
| `python --version` | `python: command not found` in the default non-interactive SSH PATH |
| `which python` | no path returned |
| `conda env list` | `conda not found` in the default non-interactive SSH PATH |
| `uv --version` | `uv not found` in the default non-interactive SSH PATH |

Blocker: a usable Python environment needs to be selected or activated before running any project scripts. This setup step did not create or modify an environment.

## `/mnt/20t/` Overview

The `/mnt/20t/` mount is writable and contains user directories including:

- `/mnt/20t/xuxin/`
- `/mnt/20t/xuhaoming/`
- other shared user directories

The working directory `/mnt/20t/xuxin/` already contained EasyEdit-related directories before this scaffold was created, including:

- `EasyEdit/`
- `qwen2vl_tmp_test/`
- `qwen3_test/`

## Local Model Root Overview

Model root checked:

```text
/mnt/20t/xuhaoming/models/
```

Top-level entries observed:

```text
.qwen2-vl-7b-modelscope-tmp
Qwen2.5-1.5B-Instruct
Qwen2.5-7B-Instruct
Qwen3-30B-A3B-Instruct-2507
Qwen3-30B-A3B-Thinking-2507
Qwen3-4B-Instruct-2507
Qwen3-VL-4B-Instruct
Qwen3.5-9B
qwen2-vl-7b
```

Approximate top-level model entry count: `9`.

No model files were opened, loaded, downloaded, or modified.

## Project Scaffold

Created project root:

```text
/mnt/20t/xuxin/mech_interp_comparison/
```

Created or verified structure:

```text
README.md
LICENSE_NOTES.md
env/
configs/
data/
data/mini_prompts/
repos/
scripts/
scripts/setup_env.sh
src/
src/mech_interp_comparison/
src/mech_interp_comparison/__init__.py
src/mech_interp_comparison/paths.py
src/mech_interp_comparison/logging_utils.py
src/mech_interp_comparison/prompt_sets.py
src/mech_interp_comparison/model_utils.py
src/mech_interp_comparison/nla/__init__.py
src/mech_interp_comparison/vpd/__init__.py
src/mech_interp_comparison/evaluation/__init__.py
outputs/
logs/
reports/
tmp/
.cache/
```

`scripts/setup_env.sh` was syntax-checked with `bash -n` and executed once to verify path exports. It created only project-local cache and temporary directories.

Exported cache and temporary paths:

```text
PROJECT_ROOT=/mnt/20t/xuxin/mech_interp_comparison
HF_HOME=/mnt/20t/xuxin/mech_interp_comparison/.cache/huggingface
HF_HUB_CACHE=/mnt/20t/xuxin/mech_interp_comparison/.cache/huggingface/hub
TRANSFORMERS_CACHE=/mnt/20t/xuxin/mech_interp_comparison/.cache/huggingface/transformers
HF_DATASETS_CACHE=/mnt/20t/xuxin/mech_interp_comparison/.cache/huggingface/datasets
TORCH_HOME=/mnt/20t/xuxin/mech_interp_comparison/.cache/torch
XDG_CACHE_HOME=/mnt/20t/xuxin/mech_interp_comparison/.cache/xdg
TMPDIR=/mnt/20t/xuxin/mech_interp_comparison/tmp
```

All exported cache and temporary paths are under `/mnt/20t/xuxin/mech_interp_comparison/`.

## Current Blockers and Missing Dependencies

- `python` is not available in the default non-interactive SSH PATH.
- `conda` is not available in the default non-interactive SSH PATH.
- `uv` is not available in the default non-interactive SSH PATH.
- VPD and NLA external repositories, checkpoints, and exact licenses have not yet been inspected.
- No Python dependency file has been selected yet because the runtime environment is not confirmed.

## Next Safe Step

Identify the intended Python or Conda environment without writing to home directories, then inspect VPD and NLA upstream repositories, licenses, and reusable artifacts before adding any external code.

## Environment Recheck

Date: 2026-06-16  
Reason: VSCode sessions expose Python and Conda even though the default non-interactive SSH PATH does not.

The default SSH PATH remains minimal:

```text
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
```

In that default PATH:

- `python` is not found.
- `python3` resolves to `/usr/bin/python3`.
- `conda` is not found.
- `uv` is not found.

The server Conda installation was found at:

```text
/mnt/8t/xhm/miniconda3/bin/conda
```

Available Conda environments:

```text
base                   /mnt/8t/xhm/miniconda3
CCKS                   /mnt/8t/xhm/miniconda3/envs/CCKS
EasyEdit               /mnt/8t/xhm/miniconda3/envs/EasyEdit
EasyEdit1              /mnt/8t/xhm/miniconda3/envs/EasyEdit1
belief_training        /mnt/8t/xhm/miniconda3/envs/belief_training
cot                    /mnt/8t/xhm/miniconda3/envs/cot
easysteer              /mnt/8t/xhm/miniconda3/envs/easysteer
knowledgecircuits      /mnt/8t/xhm/miniconda3/envs/knowledgecircuits
llmee                  /mnt/8t/xhm/miniconda3/envs/llmee
semeval                /mnt/8t/xhm/miniconda3/envs/semeval
verl                   /mnt/8t/xhm/miniconda3/envs/verl
```

The `EasyEdit1` Python interpreter is:

```text
/mnt/8t/xhm/miniconda3/envs/EasyEdit1/bin/python
```

Runtime check under `EasyEdit1`:

| Package | Version |
| --- | --- |
| Python | `3.10.20` |
| pip | `26.1.2` |
| torch | `2.9.1+cu128` |
| transformers | `5.5.4` |
| accelerate | `1.13.0` |
| datasets | `4.8.5` |
| huggingface_hub | `1.17.0` |
| numpy | `2.2.6` |
| pandas | `2.3.3` |
| scikit-learn | `1.7.2` |

CUDA check under `EasyEdit1`:

```text
torch_cuda_available=True
torch_cuda_device_count=2
torch_cuda_device_0=NVIDIA A100-PCIE-40GB
```

Project import check succeeded with:

```text
PYTHONPATH=/mnt/20t/xuxin/mech_interp_comparison/src
```

The recheck used project-local cache and temporary paths:

```text
HF_HOME=/mnt/20t/xuxin/mech_interp_comparison/.cache/huggingface
HF_HUB_CACHE=/mnt/20t/xuxin/mech_interp_comparison/.cache/huggingface/hub
TRANSFORMERS_CACHE=/mnt/20t/xuxin/mech_interp_comparison/.cache/huggingface/transformers
HF_DATASETS_CACHE=/mnt/20t/xuxin/mech_interp_comparison/.cache/huggingface/datasets
TORCH_HOME=/mnt/20t/xuxin/mech_interp_comparison/.cache/torch
XDG_CACHE_HOME=/mnt/20t/xuxin/mech_interp_comparison/.cache/xdg
TMPDIR=/mnt/20t/xuxin/mech_interp_comparison/tmp
```

The previous blocker should therefore be interpreted as a PATH initialization issue, not as a missing Conda/Python installation.

## Qwen2.5-7B-Instruct Availability

The requested model directory exists locally:

```text
/mnt/20t/xuhaoming/models/Qwen2.5-7B-Instruct
```

Top-level files observed:

```text
.cache
.gitattributes
LICENSE
README.md
config.json
generation_config.json
merges.txt
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
vocab.json
```

No model download is needed for `Qwen2.5-7B-Instruct`.
