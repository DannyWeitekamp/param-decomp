#!/bin/bash
# One SPD worker per Slurm task (launched by srun from train.sbatch). Using srun instead of
# torchrun means every worker receives SIGTERM directly and can checkpoint before exiting;
# torchrun's agent SIGKILLs workers ~30 s after it gets SIGTERM, too short for a checkpoint.
set -euo pipefail
source "$VPD_REPO/vpd/env.sh"
# Single-task runs stay non-distributed (the toy scripts never call init_distributed).
if (( SLURM_NTASKS > 1 )); then
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
  export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# NCCL flight recorder: on a collective timeout every rank dumps its recent collectives (with stack
# traces) so we can see which rank stopped where. Dumps go to the shared logs dir, not node-local /tmp.
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-$VPD_ROOT/logs/nccl_trace-${SLURM_JOB_ID:-local}-rank_}
cd "$VPD_REPO"
exec .venv/bin/python -u "$@"
