#!/bin/bash
# Submit an SPD run on one node.
#   vpd/slurm/submit.sh [--chain] <n_gpus> <config.yaml> <run_id> [extra sbatch args...]
# --chain: after a clean stop (time limit / preemption checkpoint), requeue until resume_dir/DONE.
# The experiment script is chosen from the config's task_name (lm / tms / resid_mlp).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
CHAIN=0; if [[ "${1:-}" == --chain ]]; then CHAIN=1; shift; fi
NGPU=$1; CONFIG=$(realpath "$2"); RUN_ID=$3; shift 3
read -r TASK BATCH < <("$VPD_REPO/.venv/bin/python" -c "import yaml,sys;c=yaml.safe_load(open(sys.argv[1]));print(c['task_config']['task_name'], c['batch_size'])" "$CONFIG")
case $TASK in
  lm) SCRIPT=spd/experiments/lm/lm_decomposition.py ;;
  tms) SCRIPT=spd/experiments/tms/tms_decomposition.py ;;
  resid_mlp) SCRIPT=spd/experiments/resid_mlp/resid_mlp_decomposition.py ;;
  *) echo "unknown task_name $TASK" >&2; exit 1 ;;
esac
(( BATCH % NGPU == 0 )) || { echo "batch_size $BATCH not divisible by $NGPU GPUs" >&2; exit 1; }
GRES="gpu:${VPD_GPU_TYPE:+$VPD_GPU_TYPE:}$NGPU"
set -x
CHAIN=$CHAIN sbatch --export=ALL,CHAIN=$CHAIN,VPD_REPO="$VPD_REPO" $VPD_SBATCH_ARGS \
  --nodes=1 --ntasks=$NGPU --gres=$GRES -c ${VPD_CPUS_PER_GPU:-6} --mem=$(( NGPU * ${VPD_MEM_PER_GPU_GB:-40} ))G \
  -t "${VPD_TIME:-04:00:00}" -J "vpd-$RUN_ID" -o "$VPD_ROOT/logs/$RUN_ID-%j.out" "$@" \
  "$VPD_REPO/vpd/slurm/train.sbatch" "$SCRIPT" "$CONFIG" "$RUN_ID"
