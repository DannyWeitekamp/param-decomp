# Source from every job and before submitting:  source vpd/env.sh
VPD_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VPD_REPO
if [[ ! -f "$VPD_REPO/vpd/local.env" ]]; then
  echo "vpd/env.sh: create $VPD_REPO/vpd/local.env from vpd/local.env.example first" >&2
  return 1 2>/dev/null || exit 1
fi
set -a; source "$VPD_REPO/vpd/local.env"; set +a
: "${VPD_ROOT:?set VPD_ROOT in vpd/local.env}"
export UV_CACHE_DIR=$VPD_ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$VPD_ROOT/cache/uv-python
export UV_LINK_MODE=copy
export HF_HOME=$VPD_ROOT/cache/hf
export WANDB_DIR=$VPD_ROOT/cache/wandb
export WANDB_CACHE_DIR=$VPD_ROOT/cache/wandb/cache
export WANDB_DATA_DIR=$VPD_ROOT/cache/wandb/data
export SPD_OUT_DIR=$VPD_ROOT/spd_out
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ -n "${VPD_NCCL_P2P_LEVEL:-}" ]] && export NCCL_P2P_LEVEL=$VPD_NCCL_P2P_LEVEL
export PATH=$VPD_ROOT/bin:$PATH
