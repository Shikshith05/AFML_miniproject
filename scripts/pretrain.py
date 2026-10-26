"""Entry point for a pretraining run.

    python scripts/pretrain.py --config configs/gqa_base.yaml
    python scripts/pretrain.py --preset gqa --max-steps 200 --run-name smoke

Owner: track C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from slm.model.config import Config, gqa_base, mha_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain the SLM")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=str, help="path to a YAML config")
    src.add_argument("--preset", choices=["mha", "gqa"], help="built-in preset")

    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore existing checkpoints and start from scratch",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved config and exit without training",
    )
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = {"mha": mha_baseline, "gqa": gqa_base}[args.preset]()

    # CLI overrides win over the file
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.micro_batch_size is not None:
        cfg.train.micro_batch_size = args.micro_batch_size
    if args.precision is not None:
        cfg.train.precision = args.precision
    if args.device is not None:
        cfg.train.device = args.device

    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: cuda requested but unavailable -- falling back to cpu")
        cfg.train.device = "cpu"
        cfg.train.precision = "fp32"

    return cfg


def main() -> int:
    args = parse_args()
    cfg = build_config(args)

    print(cfg.summary())
    print()

    # Record the exact config this run used, next to its results.
    cfg.to_yaml(Path(cfg.results_dir) / "config.yaml")

    if args.dry_run:
        print("dry run -- exiting before training")
        return 0

    # Imported here so --dry-run works before track A and B have landed.
    from slm.data.tinystories import make_batch_provider   # track A
    from slm.model.transformer import Transformer          # track B
    from slm.training.pretrain import pretrain

    model = Transformer(cfg.model)
    get_batch = make_batch_provider(cfg)

    pretrain(model, cfg, get_batch, resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
