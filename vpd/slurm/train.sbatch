#!/bin/bash
# Submit with vpd/slurm/submit.sh (it sets partition, GPUs, log path, and exports VPD_REPO).
# Args: <script.py> <config.yaml> <run_id>
#SBATCH --signal=TERM@300
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
source "$VPD_REPO/vpd/env.sh"
SCRIPT=$1; CONFIG=$2; RUN_ID=$3
RESUME_DIR=$("$VPD_REPO/.venv/bin/python" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1])).get('resume_dir') or '')" "$CONFIG")
echo "=== $(date -Is) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) ntasks=$SLURM_NTASKS gpus=${CUDA_VISIBLE_DEVICES:-} config=$CONFIG run_id=$RUN_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 30 >> "$VPD_ROOT/logs/gpu-$SLURM_JOB_ID.csv" &
T0=$(date +%s)
srun --kill-on-bad-exit=1 --ntasks=$SLURM_NTASKS --cpu-bind=none "$VPD_REPO/vpd/slurm/task.sh" \
  "$SCRIPT" --config_path "$CONFIG" --run_id "$RUN_ID" &
PID=$!
# Keep the batch shell alive on SIGTERM (preemption/time limit); Slurm signals the workers itself.
trap 'echo "$(date -Is) batch shell got SIGTERM"' TERM
RC=0
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
wait $PID 2>/dev/null; R2=$?; [[ $R2 -ne 127 ]] && RC=$R2
echo "=== $(date -Is) srun exit=$RC after $(( $(date +%s) - T0 ))s"
if [[ -n "$RESUME_DIR" && -e "$RESUME_DIR/DONE" ]]; then echo "run DONE"; exit 0; fi
if [[ $RC -eq 0 && "${CHAIN:-0}" == 1 && -n "$RESUME_DIR" ]]; then
  echo "not done; requeueing"; scontrol requeue "$SLURM_JOB_ID"
fi
exit $RC
