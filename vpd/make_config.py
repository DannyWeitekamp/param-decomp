"""Write a run config = a base experiment config + overrides.

Usage: python vpd/make_config.py <out.yaml> [--base <yaml>] key=value ...
Values are parsed as YAML. The default base is the paper's 4L Pile config, with the target model
pointed at the local download in $VPD_ROOT/target (no W&B login needed) and wandb_project taken
from $VPD_WANDB_PROJECT (empty -> null).
"""

import argparse
import os
from pathlib import Path

import yaml

from spd.utils.run_utils import parse_config

REPO = Path(__file__).resolve().parent.parent
PAPER_4L = REPO / "spd/experiments/lm/pile_llama_simple_mlp-4L.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--base", default=str(PAPER_4L))
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    root = Path(os.environ["VPD_ROOT"])
    cfg = yaml.safe_load(open(args.base))
    if Path(args.base).resolve() == PAPER_4L:
        cfg["pretrained_model_name"] = str(root / "target/t-9d2b8f02/ckpt/model_step_99999.pt")
    cfg["wandb_project"] = os.environ.get("VPD_WANDB_PROJECT") or None
    for kv in args.overrides:
        key, value = kv.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    yaml.safe_dump(cfg, open(args.out, "w"), sort_keys=False)
    parse_config(args.out, None)  # pydantic validation (extra="forbid" catches misspelled keys)
    print(f"{args.out}: overrides {[kv.split('=', 1)[0] for kv in args.overrides]}")


if __name__ == "__main__":
    main()
