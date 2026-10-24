"""Sanity-check the prepared token files.

Decodes a few random windows from val.bin (they should read as coherent
stories), then measures on the whole val set:

- fertility = content tokens / whitespace words (<bos>/<eos> excluded)
- average tokens per story
- % of tokens that are <unk> (byte-level BPE: should be 0)

and, unless --no-gpt2, the GPT-2 tokenizer's fertility on the same text.

    python scripts/inspect_data.py
    python scripts/inspect_data.py --n-windows 5 --no-gpt2

Owner: track A.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from slm.data.tinystories import DEFAULT_DATA_DIR, DEFAULT_SEQ_LEN, TOKEN_DTYPE
from slm.tokenizer.tokenizer import DEFAULT_TOKENIZER_PATH, Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect data/val.bin")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gpt2", action="store_true", help="skip the GPT-2 fertility comparison")
    return p.parse_args()


def split_stories(data: np.ndarray, bos: int, eos: int) -> list[np.ndarray]:
    """Content tokens of each story (between each <bos> and its <eos>)."""
    starts = np.flatnonzero(data == bos)
    ends = np.flatnonzero(data == eos)
    if len(starts) != len(ends) or not np.all(starts < ends):
        raise AssertionError("val.bin is not a clean sequence of <bos> ... <eos> stories")
    return [data[s + 1 : e] for s, e in zip(starts, ends)]


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    tok = Tokenizer.load(args.tokenizer)
    data = np.memmap(data_dir / "val.bin", dtype=TOKEN_DTYPE, mode="r")
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    print(f"val.bin: {len(data):,} tokens; vocab {tok.vocab_size}; max id {int(data.max())}")
    if meta and meta.get("n_val_tokens") != len(data):
        print(f"WARNING: meta.json says n_val_tokens={meta.get('n_val_tokens')}")

    # ---- random windows -----------------------------------------------------
    g = torch.Generator().manual_seed(args.seed)
    for k in range(args.n_windows):
        i = int(torch.randint(len(data) - args.seq_len, (1,), generator=g))
        window = data[i : i + args.seq_len]
        print(f"\n{'=' * 30} window {k + 1}/{args.n_windows} @ {i:,} ({args.seq_len} tokens) {'=' * 30}")
        print(tok.decode(window.tolist(), skip_special_tokens=False))
    print("=" * 80)

    # ---- stats over the whole val set ---------------------------------------
    t0 = time.perf_counter()
    stories = split_stories(np.asarray(data), tok.bos_id, tok.eos_id)
    texts = tok.decode_batch([s.tolist() for s in stories])
    n_content = sum(len(s) for s in stories)
    n_words = sum(len(t.split()) for t in texts)
    n_unk = int((np.asarray(data) == tok.unk_id).sum())
    stats = {
        "n_val_stories": len(stories),
        "n_val_tokens_total": int(len(data)),
        "n_val_tokens_content": n_content,
        "n_whitespace_words": n_words,
        "fertility_tokens_per_word": round(n_content / n_words, 4),
        "fertility_incl_bos_eos": round(len(data) / n_words, 4),
        "avg_tokens_per_story": round(len(data) / len(stories), 1),
        "avg_content_tokens_per_story": round(n_content / len(stories), 1),
        "pct_unk": round(100 * n_unk / len(data), 6),
    }

    if not args.no_gpt2:
        try:
            from tokenizers import Tokenizer as HFTokenizer

            gpt2 = HFTokenizer.from_pretrained("gpt2")
            fn = getattr(gpt2, "encode_batch_fast", gpt2.encode_batch)
            n_gpt2 = sum(len(e.ids) for e in fn(texts, add_special_tokens=False))
            stats["gpt2_vocab_size"] = gpt2.get_vocab_size()
            stats["gpt2_fertility_tokens_per_word"] = round(n_gpt2 / n_words, 4)
        except Exception as e:  # network or hub trouble shouldn't fail the sanity check
            print(f"(skipping GPT-2 comparison: {e})")

    print(json.dumps(stats, indent=2))
    print(f"stats in {time.perf_counter() - t0:.1f}s")

    # Round-trip check on one story: decode -> encode reproduces the tokens.
    s0 = stories[0].tolist()
    if tok.encode(tok.decode(s0)) != s0:
        print("WARNING: decode->encode did not reproduce the first val story's tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
