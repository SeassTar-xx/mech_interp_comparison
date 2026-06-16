#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/20t/xuxin/mech_interp_comparison"
CACHE_ROOT="${PROJECT_ROOT}/.cache"
TMP_ROOT="${PROJECT_ROOT}/tmp"

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${TMP_ROOT}"

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}"

printf 'PROJECT_ROOT=%s\n' "${PROJECT_ROOT}"
printf 'HF_HOME=%s\n' "${HF_HOME}"
printf 'HF_HUB_CACHE=%s\n' "${HF_HUB_CACHE}"
printf 'TRANSFORMERS_CACHE=%s\n' "${TRANSFORMERS_CACHE}"
printf 'HF_DATASETS_CACHE=%s\n' "${HF_DATASETS_CACHE}"
printf 'TORCH_HOME=%s\n' "${TORCH_HOME}"
printf 'XDG_CACHE_HOME=%s\n' "${XDG_CACHE_HOME}"
printf 'TMPDIR=%s\n' "${TMPDIR}"
