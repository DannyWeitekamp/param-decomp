#!/bin/bash
# One-time setup on a new cluster. Needs internet. Downloads ~4 GB of wheels; run it on a compute
# node if the login node forbids that (e.g. sbatch --wrap "bash vpd/setup.sh").
#   1. cp vpd/local.env.example vpd/local.env && edit it
#   2. bash vpd/setup.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$VPD_REPO"
mkdir -p "$VPD_ROOT"/{bin,cache/hf,cache/wandb,cache/uv,configs,logs,runs,target,paper_run}

if ! command -v uv >/dev/null 2>&1; then
  echo "--- installing standalone uv to $VPD_ROOT/bin"
  tmp=$(mktemp -d)
  curl -sLf -o "$tmp/uv.tgz" https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz
  tar xzf "$tmp/uv.tgz" -C "$tmp"
  cp "$tmp"/uv-x86_64-unknown-linux-gnu/{uv,uvx} "$VPD_ROOT/bin/"
  rm -rf "$tmp"
fi
uv --version

echo "--- python env (uv sync, includes dev group for tests)"
uv python install 3.13
uv sync
[[ -f spd/scripts/sweep_params.yaml ]] || cp spd/scripts/sweep_params.yaml.example spd/scripts/sweep_params.yaml
.venv/bin/python -c 'import torch, datasets; print("torch", torch.__version__, "cuda", torch.version.cuda, "datasets", datasets.__version__)'

# Public W&B run files download anonymously (no API key needed).
fetch() {  # project run file dest
  [[ -s "$4" ]] && return 0
  curl -sLf -o "$4" "https://api.wandb.ai/files/goodfire/$1/$2/$3" || { rm -f "$4"; echo "could not fetch $1/$2/$3" >&2; return 1; }
}
echo "--- target model goodfire/spd/t-9d2b8f02"
T=$VPD_ROOT/target/t-9d2b8f02; mkdir -p "$T/ckpt"
for f in final_config.yaml model_config.yaml tokenizer.json; do fetch spd t-9d2b8f02 $f "$T/$f"; done
fetch spd t-9d2b8f02 model_step_99999.pt "$T/ckpt/model_step_99999.pt"
echo "--- toy targets"
mkdir -p "$VPD_ROOT/target/tms_5-2" "$VPD_ROOT/target/resid_mlp1"
for f in tms.pth tms_train_config.yaml; do fetch spd-pre-Sep-2025 0hsp07o4 $f "$VPD_ROOT/target/tms_5-2/$f"; done
for f in resid_mlp.pth resid_mlp_train_config.yaml label_coeffs.json; do fetch spd-pre-Sep-2025 pziyck78 $f "$VPD_ROOT/target/resid_mlp1/$f"; done
echo "--- paper decomposition run goodfire/spd/s-55ea3f9b (config + stdout log for reference curves)"
P=$VPD_ROOT/paper_run/s-55ea3f9b; mkdir -p "$P"
for f in final_config.yaml wandb-metadata.json wandb-summary.json output.log; do fetch spd s-55ea3f9b $f "$P/$f"; done
.venv/bin/python vpd/tools/parse_log.py "$P/output.log" "$P/metrics.json"

echo "--- configs"
bash vpd/make_standard_configs.sh
ls -1 "$VPD_ROOT/configs"
echo "setup done. Next: run the CPU tests (make test) and vpd/slurm/nccl_bench.sbatch, then a smoke run (see vpd/README.md)."
