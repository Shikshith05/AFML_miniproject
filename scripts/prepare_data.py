"""Prepare project datasets.

TinyStories -> data/train.bin, data/val.bin (flat np.uint16 token streams,
each story as <bos> tokens <eos>), plus data/meta.json and the val holdout
indices in data/val_indices.npy.

Steps, in order:
  1. Hold out 1% of stories as val (seed 1337), computed over the full dataset.
  2. Train the tokenizer on the train side only -- if --train-tokenizer is
     given or no tokenizer exists yet. An existing tokenizer is checked
     against the split fingerprint so val text can't have leaked into it.
  3. Encode in batches, streaming to disk.

    python scripts/prepare_data.py --train-tokenizer      # full pipeline
    python scripts/prepare_data.py --limit 20000 --train-tokenizer   # smoke run

Owner: track A.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import chain
from pathlib import Path

# allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from tqdm import tqdm

from slm.data.tinystories import (
    DATASET_NAME,
    DEFAULT_DATA_DIR,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    TOKEN_DTYPE,
    load_tinystories,
    split_hash,
    split_val_indices,
)
from slm.tokenizer.tokenizer import DEFAULT_TOKENIZER_PATH, Tokenizer
from slm.tokenizer.train_bpe import DEFAULT_VOCAB_SIZE, load_train_info, train_on_tinystories


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenize TinyStories into train/val .bin files")
    p.add_argument("--limit", type=int, default=None, help="only use the first N stories (smoke run)")
    p.add_argument("--out-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--train-tokenizer", action="store_true", help="(re)train the tokenizer first")
    p.add_argument("--max-stories", type=int, default=None, help="tokenizer training subset (default: all)")
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--batch-size", type=int, default=10_000, help="stories per encode batch")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    # ---- load ---------------------------------------------------------------
    t0 = time.perf_counter()
    print(f"loading {DATASET_NAME} ...")
    ds = load_tinystories("train")
    n_total = len(ds)
    n = n_total if args.limit is None else min(args.limit, n_total)
    timings["load_s"] = time.perf_counter() - t0
    print(f"{n_total:,} stories, using {n:,} ({timings['load_s']:.1f}s)")

    # ---- 1. split (before tokenizer training) -------------------------------
    val_idx = split_val_indices(n_total, args.val_fraction, args.seed)
    shash = split_hash(val_idx)
    np.save(out_dir / "val_indices.npy", val_idx)
    (out_dir / "split.json").write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "method": "np.random.default_rng(seed).permutation(n_total)[:round(n_total*val_fraction)], sorted",
                "n_total_stories": n_total,
                "n_val_stories": len(val_idx),
                "seed": args.seed,
                "val_fraction": args.val_fraction,
                "split_hash": shash,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"val holdout: {len(val_idx):,} of {n_total:,} stories, split hash {shash}")

    # ---- 2. tokenizer -------------------------------------------------------
    tok_path = Path(args.tokenizer)
    if args.train_tokenizer or not tok_path.exists():
        t0 = time.perf_counter()
        train_on_tinystories(
            ds,
            val_idx,
            tok_path,
            vocab_size=args.vocab_size,
            limit=args.limit,
            max_stories=args.max_stories,
            info={
                "dataset": DATASET_NAME,
                "n_total_stories": n_total,
                "seed": args.seed,
                "val_fraction": args.val_fraction,
            },
        )
        timings["tokenizer_s"] = time.perf_counter() - t0
    else:
        info = load_train_info(tok_path)
        if info is None or info.get("split_hash") != shash:
            raise SystemExit(
                f"{tok_path} was not trained on this split (its split_hash: "
                f"{None if info is None else info.get('split_hash')}, current: {shash}). "
                "Pass --train-tokenizer to retrain it."
            )
        print(f"using existing tokenizer {tok_path} (trained on {info['n_stories_trained']:,} stories)")

    tok = Tokenizer.load(tok_path)
    assert tok.vocab_size <= np.iinfo(TOKEN_DTYPE).max + 1, "vocab does not fit in uint16"

    # ---- 3. encode, streaming to disk ---------------------------------------
    t0 = time.perf_counter()
    is_val = np.zeros(n_total, dtype=bool)
    is_val[val_idx] = True
    tmp = {s: out_dir / f"{s}.bin.tmp" for s in ("train", "val")}
    files = {s: open(p, "wb") for s, p in tmp.items()}
    n_tokens = {"train": 0, "val": 0}
    n_stories = {"train": 0, "val": 0}
    n_empty = 0
    max_id = 0
    try:
        with tqdm(total=n, desc="encoding", unit="story", unit_scale=True) as bar:
            for start in range(0, n, args.batch_size):
                stop = min(start + args.batch_size, n)
                texts = ds[start:stop]["text"]
                mask = is_val[start:stop]
                for split, want_val in (("train", False), ("val", True)):
                    batch = [t for t, v in zip(texts, mask) if v == want_val and t.strip()]
                    if not batch:
                        continue
                    ids = tok.encode_batch(batch, add_bos=True, add_eos=True)
                    arr = np.fromiter(chain.from_iterable(ids), dtype=np.int64)
                    max_id = max(max_id, int(arr.max()))
                    assert max_id < 65536, f"token id {max_id} does not fit in uint16"
                    arr.astype(TOKEN_DTYPE).tofile(files[split])
                    n_tokens[split] += arr.size
                    n_stories[split] += len(batch)
                n_empty += sum(1 for t in texts if not t.strip())
                bar.update(stop - start)
                bar.set_postfix(train_tok=f"{n_tokens['train'] / 1e6:.1f}M", val_tok=f"{n_tokens['val'] / 1e6:.2f}M")
    finally:
        for f in files.values():
            f.close()
    for split, p in tmp.items():
        os.replace(p, out_dir / f"{split}.bin")
    timings["encode_s"] = time.perf_counter() - t0
    timings["total_s"] = time.perf_counter() - t_start

    meta = {
        "vocab_size": tok.vocab_size,
        "n_train_tokens": n_tokens["train"],
        "n_val_tokens": n_tokens["val"],
        "n_train_stories": n_stories["train"],
        "n_val_stories": n_stories["val"],
        "seed": args.seed,
        "tokenizer": tok_path.as_posix(),
        "val_fraction": args.val_fraction,
        "split_hash": shash,
        "dataset": DATASET_NAME,
        "n_total_stories": n_total,
        "limit": args.limit,
        "n_empty_skipped": n_empty,
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "max_token_id": max_id,
        "special_ids": {"pad": tok.pad_id, "bos": tok.bos_id, "eos": tok.eos_id, "unk": tok.unk_id},
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"done in {timings['total_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
