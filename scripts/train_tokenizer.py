"""Train the project tokenizer.

Byte-level BPE on the TRAIN portion of TinyStories (the 1% val holdout is
excluded, using the same split as scripts/prepare_data.py).

    python scripts/train_tokenizer.py                       # all train stories
    python scripts/train_tokenizer.py --max-stories 200000  # subset
    python scripts/train_tokenizer.py --limit 20000         # smoke run

Owner: track A.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm.data.tinystories import (
    DATASET_NAME,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    load_tinystories,
    split_val_indices,
)
from slm.tokenizer.tokenizer import DEFAULT_TOKENIZER_PATH
from slm.tokenizer.train_bpe import DEFAULT_VOCAB_SIZE, train_on_tinystories


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the byte-level BPE tokenizer")
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--max-stories", type=int, default=None, help="cap on train stories used (default: all)")
    p.add_argument("--limit", type=int, default=None, help="only consider the first N dataset stories")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--out", type=str, default=DEFAULT_TOKENIZER_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()

    print(f"loading {DATASET_NAME} ...")
    ds = load_tinystories("train")
    print(f"{len(ds):,} stories ({time.perf_counter() - t0:.1f}s)")

    val_idx = split_val_indices(len(ds), args.val_fraction, args.seed)
    train_on_tinystories(
        ds,
        val_idx,
        args.out,
        vocab_size=args.vocab_size,
        limit=args.limit,
        max_stories=args.max_stories,
        info={
            "dataset": DATASET_NAME,
            "n_total_stories": len(ds),
            "seed": args.seed,
            "val_fraction": args.val_fraction,
        },
    )
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
