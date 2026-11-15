#!/bin/bash
# Writes the standard run configs into $VPD_ROOT/configs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$VPD_REPO"
C=$VPD_ROOT/configs; R=$VPD_ROOT/runs; PY=.venv/bin/python
# Full paper run (400k steps), resumable, checkpoint every 1000 steps.
$PY vpd/make_config.py $C/full_4L.yaml resume_dir=$R/vpd-4L/ckpt resume_save_freq=1000
# Smoke test: paper schedules, stop after 1100 steps (includes evals at 0 and 1000), no checkpoints.
$PY vpd/make_config.py $C/smoke_4L.yaml stop_after_step=1100
# Resume A/B test: A runs 200 steps straight; B checkpoints every 50 and gets killed + resumed.
AB="stop_after_step=200 train_log_freq=10 eval_freq=100 slow_eval_on_first_step=false"
$PY vpd/make_config.py $C/resA.yaml $AB
$PY vpd/make_config.py $C/resB.yaml $AB resume_dir=$R/resB/ckpt resume_save_freq=50
# Toys (local targets, no bf16 so they also run on pre-Ampere GPUs).
for x in tms/tms_5-2:tms_5-2/tms.pth resid_mlp/resid_mlp1:resid_mlp1/resid_mlp.pth; do
  src=${x%%:*}; pth=${x##*:}; name=$(basename $src)
  $PY vpd/make_config.py $C/toy_$name.yaml --base spd/experiments/${src}_config.yaml \
    pretrained_model_path=$VPD_ROOT/target/$pth autocast_bf16=false
done
