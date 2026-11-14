"""Run supervised fine-tuning on a pretrained checkpoint.

    python scripts/sft.py --preset gqa --checkpoint checkpoints/gqa_base/final.pt
    python scripts/sft.py --preset mha --checkpoint checkpoints/mha_base/final.pt

    # human-written data, kept apart from the Alpaca run:
    python scripts/sft.py --preset gqa --checkpoint checkpoints/gqa_base/final.pt         --dataset oasst1 --run-name gqa_sft_oasst1

Output: checkpoints/<run-name>/{epoch_*,final}.pt and
results/<run-name>/metrics.jsonl, where --run-name defaults to <preset>_sft
(gqa_sft is the Alpaca run).

Run it on both presets: that is what shows whether GQA's quality parity
survives instruction tuning, not just pretraining.

Owner: track D (written under time pressure by track C).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from slm.model.config import Config, gqa_base, mha_baseline
from slm.tokenizer.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instruction-tune a pretrained model")
    p.add_argument("--preset", choices=["mha", "gqa"], required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="pretrained checkpoint to start from")
    p.add_argument("--run-name", type=str, default=None,
                   help="output folder name under checkpoints/ and results/ "
                        "(default: <preset>_sft). Use a new one per dataset, "
                        "e.g. gqa_sft_oasst1, so earlier runs are not overwritten")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None,
                   help="alpaca | dolly | oasst1 (or any HF name); default: config")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N Alpaca rows (smoke test)")
    p.add_argument("--dry-run", action="store_true",
                   help="build and check the data, then exit without training")
    return p.parse_args()


def warn_if_run_exists(run_name: str) -> None:
    """The metrics log appends and checkpoints are replaced, so reusing a run
    name mixes two runs. Not fatal (re-running on purpose is legitimate), but
    loud."""
    old = [p for p in (Path("results") / run_name / "metrics.jsonl",
                       Path("checkpoints") / run_name / "final.pt") if p.exists()]
    if old:
        print(f"WARNING: run '{run_name}' already has {', '.join(map(str, old))}. "
              "The log will be appended to and the checkpoints replaced; pass a "
              "different --run-name to keep the earlier run.")


def main() -> int:
    args = parse_args()

    cfg: Config = {"mha": mha_baseline, "gqa": gqa_base}[args.preset]()
    run_name = args.run_name or f"{args.preset}_sft"
    cfg.run_name = run_name
    if args.epochs is not None:
        cfg.sft.epochs = args.epochs
    if args.lr is not None:
        cfg.sft.lr = args.lr
    if args.micro_batch_size is not None:
        cfg.sft.micro_batch_size = args.micro_batch_size
    if args.dataset is not None:
        cfg.sft.dataset_name = args.dataset
    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: no CUDA -- falling back to cpu")
        cfg.train.device, cfg.train.precision = "cpu", "fp32"

    print(cfg.summary())
    print(f"  SFT data {cfg.sft.dataset_name}  ->  checkpoints/{run_name}/, results/{run_name}/")
    print()

    tok = Tokenizer.load(cfg.data.tokenizer_path)
    if tok.vocab_size != cfg.model.vocab_size:
        raise SystemExit(
            f"tokenizer vocab {tok.vocab_size} != config {cfg.model.vocab_size}"
        )

    # ---- data ------------------------------------------------------------
    from slm.data.alpaca import assert_masking_sane, load_alpaca, split_examples

    examples = load_alpaca(cfg, tok, limit=args.limit)
    assert_masking_sane(examples, tok)
    train_examples, val_examples = split_examples(examples, seed=cfg.train.seed)
    print(f"split: {len(train_examples):,} train / {len(val_examples):,} val")
    print()

    if args.dry_run:
        print("dry run -- exiting before training")
        return 0

    # ---- model -----------------------------------------------------------
    from slm.model.transformer import Transformer
    from slm.training.sft import sft
    from slm.training.utils import load_checkpoint

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    model = Transformer(cfg.model)
    state = load_checkpoint(
        ckpt, model=model, device=cfg.train.device, restore_rng=False
    )
    print(f"loaded {ckpt} (pretrained to step {state['step']})")

    warn_if_run_exists(run_name)
    sft(model, cfg, tok, train_examples, val_examples, run_name=run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
