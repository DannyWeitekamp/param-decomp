#!/bin/bash
# Submit with vpd/slurm/submit.sh (it sets partition, GPUs, log path, chaining, and exports VPD_REPO
# and VPD_RESUBMIT). Args: <script.py> <config.yaml> <run_id>
# Workers get SIGTERM 15 min before the time limit, enough to write a resumable checkpoint (~23 GB for
# the 4L run) even when scratch is slow.
#SBATCH --signal=TERM@900
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
# Keep the batch shell alive on SIGTERM (preemption/time limit); Slurm signals the workers itself.
GOT_TERM=0
trap 'GOT_TERM=1; echo "$(date -Is) batch shell got SIGTERM"' TERM
source "$VPD_REPO/vpd/env.sh"
SCRIPT=$1; CONFIG=$2; RUN_ID=$3
NODE=${SLURMD_NODENAME:-$(hostname -s)}
BAD_NODES=$VPD_ROOT/logs/bad_nodes.txt
RETRIES_FILE=$VPD_ROOT/logs/retries-$RUN_ID.txt
LOG=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'StdOut=\K\S+' || true)
LOG_START=$( [[ -f "$LOG" ]] && wc -l < "$LOG" || echo 0)
RESUME_DIR=$("$VPD_REPO/.venv/bin/python" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1])).get('resume_dir') or '')" "$CONFIG")
if [[ -n "$RESUME_DIR" && -e "$RESUME_DIR/DONE" ]]; then echo "$RESUME_DIR/DONE exists; nothing to do"; exit 0; fi
echo "=== $(date -Is) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} host=$NODE partition=$SLURM_JOB_PARTITION ntasks=$SLURM_NTASKS gpus=${CUDA_VISIBLE_DEVICES:-} config=$CONFIG run_id=$RUN_ID"

cancel_pending() {
  echo "cancelling pending segments of $SLURM_JOB_NAME"
  scancel -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING
}

# Hardware/infrastructure failure: don't kill the chain. Optionally blacklist this node (also for the
# segments already queued), then submit a replacement for this segment.
infra_failure() {  # <reason> <mark_node_bad: 0|1>
  echo "=== $(date -Is) infrastructure failure on $NODE: $1"
  if [[ $2 == 1 ]]; then
    echo "$NODE" >> "$BAD_NODES"
    EXCL=$(sort -u "$BAD_NODES" | paste -sd,)
    echo "added $NODE to $BAD_NODES; excluding $EXCL"
    for j in $(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING -o %i); do
      scontrol update JobId="$j" ExcNodeList="$EXCL" || echo "could not update exclude list of job $j"
    done
  fi
  N=$(( $(cat "$RETRIES_FILE" 2>/dev/null || echo 0) + 1 )); echo $N > "$RETRIES_FILE"
  if (( N > ${VPD_MAX_RETRIES:-5} )); then
    echo "replacement segment $N exceeds VPD_MAX_RETRIES=${VPD_MAX_RETRIES:-5}; giving up"
    cancel_pending; exit 1
  fi
  if [[ -x "${VPD_RESUBMIT:-}" ]]; then
    echo "submitted replacement segment $N: job $("$VPD_RESUBMIT")"
  else
    echo "VPD_RESUBMIT not set; no replacement submitted"
  fi
  exit 1
}

# GPU preflight: every GPU must accept a context and run a matmul before we start the workers.
if ! nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; then
  infra_failure "nvidia-smi failed" 1
fi
nvidia-smi topo -m
if ! timeout 300 "$VPD_REPO/.venv/bin/python" - "$SLURM_NTASKS" <<'EOF'
import sys, torch
need = int(sys.argv[1])
n = torch.cuda.device_count()
assert n >= need, f"only {n} CUDA devices visible, need {need}"
for i in range(n):
    x = torch.randn(1024, 1024, device=f"cuda:{i}")
    (x @ x).sum().item()
    torch.cuda.synchronize(i)
print(f"GPU preflight OK: {n} devices")
EOF
then
  infra_failure "GPU preflight failed" 1
fi

nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 30 >> "$VPD_ROOT/logs/gpu-$SLURM_JOB_ID.csv" &
T0=$(date +%s)
srun --kill-on-bad-exit=1 --ntasks=$SLURM_NTASKS --cpu-bind=none "$VPD_REPO/vpd/slurm/task.sh" \
  "$SCRIPT" --config_path "$CONFIG" --run_id "$RUN_ID" &
PID=$!
RC=0
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
wait $PID 2>/dev/null; R2=$?; [[ $R2 -ne 127 ]] && RC=$R2
echo "=== $(date -Is) srun exit=$RC after $(( $(date +%s) - T0 ))s"
if [[ -n "$RESUME_DIR" && -e "$RESUME_DIR/DONE" ]]; then echo "run DONE"; exit 0; fi
# Time limit / preemption: a normal segment end; the next singleton segment resumes.
if (( GOT_TERM )); then echo "segment ended by SIGTERM (time limit or preemption)"; exit 0; fi
(( RC == 0 )) && exit 0

SEGMENT_LOG=$( [[ -f "$LOG" ]] && tail -n +$(( LOG_START + 1 )) "$LOG" || true)
if grep -qE 'busy or unavailable|uncorrectable ECC|ECC error|GPU is lost|fallen off the bus|no CUDA-capable device|cudaErrorDevicesUnavailable|CUDA error: unknown error|NVML.*(Unknown|GPU lost)' <<< "$SEGMENT_LOG"; then
  infra_failure "GPU error in worker output" 1
elif grep -qE 'Watchdog caught collective operation timeout' <<< "$SEGMENT_LOG"; then
  infra_failure "NCCL collective timeout (often stalled shared storage); see nccl_trace-$SLURM_JOB_ID-* in logs" 0
elif grep -qE 'DUE TO NODE FAILURE|NODE_FAIL' <<< "$SEGMENT_LOG"; then
  infra_failure "Slurm node failure" 0
fi
echo "segment failed with a code error"
cancel_pending
exit $RC
