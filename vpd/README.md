# VPD replication toolkit

Replicating adVersarial Parameter Decomposition (Bushnaq et al., 2026, "Interpreting Language Model
Parameters") on Goodfire's 4-layer, 67M-param Pile target model `goodfire/spd/t-9d2b8f02`.

This branch is the **paper commit** `85c6b70` (PyTorch; the code the paper's run
`goodfire/spd/s-55ea3f9b` used) plus:

- **Resumable checkpoints** (`spd/resume.py`), opt-in via config fields `resume_dir`,
  `resume_save_freq`, and `stop_after_step`. Defaults leave the original behaviour unchanged.
- **`vpd/`**: cluster-portable setup, configs, a Slurm launcher, and analysis helpers.

## Quickstart on a new cluster

```bash
git clone -b vpd-replication https://github.com/DannyWeitekamp/param-decomp.git
cd param-decomp
cp vpd/local.env.example vpd/local.env   # set VPD_ROOT (scratch), VPD_SBATCH_ARGS, VPD_GPU_TYPE, ...
bash vpd/setup.sh                        # uv + python env, target model, paper logs, configs (~4 GB download)
source vpd/env.sh

# 1. CPU tests. test_sweep_creates_slurm_array fails without a W&B login; that's expected.
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest tests/ -q -m "not slow" -n 4

# 2. Interconnect check. If p2p_level_sys is much faster than default, set VPD_NCCL_P2P_LEVEL=SYS.
vpd/slurm/nccl_bench.sh 8

# 3. Toys, then the 4L smoke test (paper schedules, stops after step 1100).
vpd/slurm/submit.sh 1 $VPD_ROOT/configs/toy_tms_5-2.yaml toy-tms
vpd/slurm/submit.sh 1 $VPD_ROOT/configs/toy_resid_mlp1.yaml toy-resid
vpd/slurm/submit.sh 8 $VPD_ROOT/configs/smoke_4L.yaml smoke-4L

# 4. Resume A/B test (see below), then the full run:
vpd/slurm/submit.sh --chain 20 8 $VPD_ROOT/configs/full_4L.yaml vpd-4L
```

`submit.sh [--chain K] <n_gpus> <config> <run_id> [extra sbatch args]` runs one node with one Slurm
task per GPU. Extra sbatch args come last and override `VPD_SBATCH_ARGS`/`VPD_TIME`. Outputs:

- logs: `$VPD_ROOT/logs/<run_id>-<jobid>.out`
- metrics: `$SPD_OUT_DIR/spd/<run_id>/metrics.jsonl`
- final model: `$SPD_OUT_DIR/spd/<run_id>/model_400000.pth`

With `--chain K`, K identical segments named `vpd-<run_id>` are submitted with
`--dependency=singleton`: Slurm runs one at a time, and each resumes from `resume_dir`. A segment ends
for any reason (time limit, preemption, node failure) and the next one starts, so this also works on
clusters whose preemption CANCELs rather than requeues. Segments exit at once if `resume_dir/DONE`
exists. Stop the chain with `scancel -n vpd-<run_id>`.

Each segment runs a GPU preflight (context + matmul on every GPU) before starting the workers, and
classifies how it ended:

- `DONE`, a clean exit, or SIGTERM (time limit / preemption): normal end, the next segment resumes.
- GPU error (preflight failure, "busy or unavailable", ECC, GPU lost): the node is appended to
  `$VPD_ROOT/logs/bad_nodes.txt` and excluded from all pending and future segments, and a replacement
  segment is submitted.
- NCCL collective timeout or Slurm node failure: a replacement segment is submitted.
- Anything else is treated as a code error and cancels the pending segments.

Replacements are capped at `VPD_MAX_RETRIES` (default 5) per `submit.sh` call. `submit.sh` writes
`$VPD_ROOT/logs/resubmit-<run_id>.sh`, which submits one more segment with the same arguments. On an
NCCL timeout each rank dumps its flight-recorder trace to `$VPD_ROOT/logs/nccl_trace-<jobid>-rank_<r>`.

## Choosing GPU count

`batch_size: 64` is **global**; each rank gets `64 / n_gpus` sequences (`lm_decomposition.py:97`).
There's no gradient accumulation, so `n_gpus` must divide 64 and each rank must fit `64 / n_gpus`
sequences in memory. **Keep `n_gpus` fixed for a whole chain**: resuming into a different world
size asserts.

| GPU | seq/rank | peak mem | s/step | notes |
|---|---|---|---|---|
| 8x H200 (paper) | 8 | ? | 0.23 | 400k steps in 25.75 h |
| A40 48 GB, 1 GPU | 8 | 36 GB | 1.65 | no comm |
| 4x A40 (PCIe), NCCL default (SHM) | 8 | 36 GB | 4.15 | 2.4 GB all-reduce at 1.7 GB/s |
| 4x A40 (PCIe), `NCCL_P2P_LEVEL=SYS` | 8 | 36 GB | 1.62 | 14.9 GB/s |
| A40, 16 seq/rank | 16 | OOM | n/a | |

Memory is ~10 GB of fixed state (600M trainable params, mostly the CI transformer, plus Adam)
plus ~3.3 GB per sequence. On 80 GB cards, 16 seq/rank (4 GPUs) should fit; check with a smoke run.
Fewer GPUs means less all-reduce traffic per step but the same total compute.

## Resume details

- Checkpoint for step `s` = state at the top of loop iteration `s`:
  `resume_dir/step_XXXXXXXX/{shared.pt, rank{r}.pt, COMPLETE}`.
  - `shared.pt` (rank 0) holds the component model + CI fn, the AdamW state, and the step.
  - `rank{r}.pt` holds that rank's shard of the persistent-PGD sources and their Adam moments,
    its RNG states, and the HF streaming-dataset `state_dict()` for train and eval.
- Writes go to a `.tmp` dir with fsync, then a rename; the last 2 complete checkpoints are kept.
  A kill mid-save leaves the previous checkpoint intact.
- Saves happen every `resume_save_freq` steps and on SIGTERM/SIGUSR1. The stop flag is
  all-reduced each step, so every rank saves at the same step. `train.sbatch` uses
  `--signal=TERM@300`. Size at 4 ranks: 8.2 GB shared + 1.9 GB/rank, ~28 s to NFS.
  Restart to first step takes ~1 m 45 s.
- **Data position**: the shard and offset are restored exactly; the 1000-example shuffle buffer is
  not, so post-resume batches are a different permutation of not-yet-seen data.
- **W&B**: `wandb.init(resume="allow")` with the run id passed on the command line, so all segments
  of `vpd-4L` log to one run. Set `VPD_WANDB_PROJECT` + `WANDB_API_KEY` in `local.env`, then
  regenerate configs with `bash vpd/make_standard_configs.sh`.

**A/B test** (done on 4x A40, batch 32):

- A = 200 steps straight. B = SIGTERM at step 124 (`scancel -s TERM <job>`), then the same
  submit command again resumes it.
- Result: before the kill, A and B match to ~0.1%. After the resume, total loss is within 0.1% and
  faithfulness matches to 5 significant figures. Per-term losses are within ~4%, the size of
  batch-to-batch noise.
- To repeat:

```bash
vpd/slurm/submit.sh 8 $VPD_ROOT/configs/resA.yaml resA
vpd/slurm/submit.sh 8 $VPD_ROOT/configs/resB.yaml resB     # scancel -s TERM <jobid> around step 120
vpd/slurm/submit.sh 8 $VPD_ROOT/configs/resB.yaml resB     # resumes and finishes
.venv/bin/python vpd/tools/compare_ab.py $SPD_OUT_DIR/spd/{resA,resB}/metrics.jsonl <kill_step>
```

## Comparing with the paper

`setup.sh` downloads the paper run's stdout log and parses it into
`$VPD_ROOT/paper_run/s-55ea3f9b/metrics.json` (`{step: {metric: value}}`, train every 200 steps,
eval every 1000). Our runs log the same keys to `metrics.jsonl`.

- Paper final eval (step 400k): `eval/ce_kl/ce_difference_stoch_masked` 0.135,
  `ce_difference_rounded_masked` 0.241, `eval/loss/PGDReconLoss` 0.652.
- Paper headline targets: target CE 2.71; masked CE unmasked 2.72, stochastic 2.84, rounded
  CI>0.1 2.95; adversarial KL 0.83 @ 20 PGD steps; ~9,972 alive subcomponents; mean L0 ~205 per
  position.

## Files

| Path | What |
|---|---|
| `vpd/local.env.example` | per-cluster settings (copy to `vpd/local.env`, which is gitignored) |
| `vpd/env.sh` | env vars for jobs: caches, `SPD_OUT_DIR`, uv, NCCL |
| `vpd/setup.sh` | one-time install + downloads + configs |
| `vpd/make_config.py`, `vpd/make_standard_configs.sh` | paper config + overrides into `$VPD_ROOT/configs` |
| `vpd/slurm/submit.sh`, `train.sbatch`, `task.sh` | launcher (srun, one task per GPU, singleton-chained segments) |
| `vpd/slurm/nccl_bench.sh` | all-reduce bandwidth check |
| `vpd/tools/parse_log.py` | stdout log to `{step: metrics}` JSON |
| `vpd/tools/compare_ab.py` | resume A/B comparison |
