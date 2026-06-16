#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/20t/xuxin/mech_interp_comparison"
PYDEPS_DIR="${PROJECT_ROOT}/env/nla_pydeps"
CUDA_ROOT="${PYDEPS_DIR}/nvidia/cu13"

source "${PROJECT_ROOT}/scripts/setup_env.sh" >/dev/null

export HOME="${PROJECT_ROOT}/tmp/sglang-home"
export PYTHONPATH="${PYDEPS_DIR}:${PROJECT_ROOT}/src:${PROJECT_ROOT}/repos/nla-inference:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PYDEPS_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib:${PYDEPS_DIR}/nvidia/cudnn/lib:${PYDEPS_DIR}/nvidia/nccl/lib:${PYDEPS_DIR}/nvidia/cusparselt/lib:${PYDEPS_DIR}/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${PYDEPS_DIR}/flashinfer/data/cccl/libcudacxx/include:${PYDEPS_DIR}/flashinfer/data/cccl/cub:${PYDEPS_DIR}/flashinfer/data/cccl/thrust:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${PYDEPS_DIR}/flashinfer/data/cccl/libcudacxx/include:${CPLUS_INCLUDE_PATH:-}"
export TORCH_EXTENSIONS_DIR="${PROJECT_ROOT}/.cache/torch_extensions"
export TRITON_CACHE_DIR="${PROJECT_ROOT}/.cache/triton"
export FLASHINFER_CACHE_DIR="${PROJECT_ROOT}/.cache/flashinfer"
export SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR="${SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK}"

mkdir -p   "${HOME}"   "${TORCH_EXTENSIONS_DIR}"   "${TRITON_CACHE_DIR}"   "${FLASHINFER_CACHE_DIR}"   "${TMPDIR}"
