#!/bin/bash
# Submit an SPD run on one node.
#   vpd/slurm/submit.sh [--chain K] <n_gpus> <config.yaml> <run_id> [extra sbatch args...]
# --chain K: submit K identical segments named vpd-<run_id> with --dependency=singleton, so Slurm runs
#   them one at a time and each resumes from resume_dir. A segment starts the next one by ending for
#   any reason (time limit, preemption, node failure), so this works when preemption CANCELs jobs
#   instead of requeueing them. Segments exit immediately once resume_dir/DONE exists.
#   A segment that hits a hardware/infrastructure error (bad GPU, NCCL timeout, node failure) submits a
#   replacement segment (at most $VPD_MAX_RETRIES per submission, default 5); a GPU failure also adds
#   the node to $VPD_ROOT/logs/bad_nodes.txt, which every segment excludes. A segment that fails with a
#   code error cancels the remaining pending ones.
# Extra sbatch args come last and override the defaults (e.g. "-q inferno -t 1-00:00:00").
# The experiment script is chosen from the config's task_name (lm / tms / resid_mlp).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
CHAIN=1; if [[ "${1:-}" == --chain ]]; then CHAIN=$2; shift 2; fi
NGPU=$1; CONFIG=$(realpath "$2"); RUN_ID=$3; shift 3
read -r TASK BATCH RESUME_DIR < <("$VPD_REPO/.venv/bin/python" -c "import yaml,sys;c=yaml.safe_load(open(sys.argv[1]));print(c['task_config']['task_name'], c['batch_size'], c.get('resume_dir') or '')" "$CONFIG")
case $TASK in
  lm) SCRIPT=spd/experiments/lm/lm_decomposition.py ;;
  tms) SCRIPT=spd/experiments/tms/tms_decomposition.py ;;
  resid_mlp) SCRIPT=spd/experiments/resid_mlp/resid_mlp_decomposition.py ;;
  *) echo "unknown task_name $TASK" >&2; exit 1 ;;
esac
(( BATCH % NGPU == 0 )) || { echo "batch_size $BATCH not divisible by $NGPU GPUs" >&2; exit 1; }
(( CHAIN == 1 )) || [[ -n "$RESUME_DIR" ]] || { echo "--chain needs resume_dir in the config" >&2; exit 1; }
GRES="gpu:${VPD_GPU_TYPE:+$VPD_GPU_TYPE:}$NGPU"
mkdir -p "$VPD_ROOT/logs"
BAD_NODES=$VPD_ROOT/logs/bad_nodes.txt
RESUBMIT=$VPD_ROOT/logs/resubmit-$RUN_ID.sh
echo 0 > "$VPD_ROOT/logs/retries-$RUN_ID.txt"
read -ra DEFAULT_ARGS <<< "$VPD_SBATCH_ARGS"
ARGS=(--export=ALL,VPD_REPO="$VPD_REPO",VPD_RESUBMIT="$RESUBMIT" "${DEFAULT_ARGS[@]}"
  --nodes=1 --ntasks="$NGPU" --gres="$GRES" -c "${VPD_CPUS_PER_GPU:-6}" --mem=$(( NGPU * ${VPD_MEM_PER_GPU_GB:-40} ))G
  -t "${VPD_TIME:-04:00:00}" -J "vpd-$RUN_ID" --dependency=singleton -o "$VPD_ROOT/logs/$RUN_ID-%j.out" "$@"
  "$VPD_REPO/vpd/slurm/train.sbatch" "$SCRIPT" "$CONFIG" "$RUN_ID")
# One-segment submitter, also run by train.sbatch to replace a segment lost to a hardware error.
{
  echo '#!/bin/bash'
  echo "# Written by vpd/slurm/submit.sh: submits one more segment of vpd-$RUN_ID, excluding bad nodes."
  echo 'set -euo pipefail'
  echo '# Do not leak the calling job'"'"'s SLURM_* variables into the new job.'
  echo 'for v in ${!SLURM_@}; do unset "$v"; done'
  echo "BAD=$(printf %q "$BAD_NODES")"
  echo 'EXCL=$( [[ -s $BAD ]] && sort -u "$BAD" | paste -sd, || true)'
  printf 'sbatch --parsable ${EXCL:+--exclude=$EXCL} '; printf '%q ' "${ARGS[@]}"; echo
} > "$RESUBMIT"
chmod +x "$RESUBMIT"
[[ -s $BAD_NODES ]] && echo "excluding bad nodes: $(sort -u "$BAD_NODES" | paste -sd,)"
for (( i = 0; i < CHAIN; i++ )); do "$RESUBMIT"; done
