"""Build DPO preference pairs (decision S3).

    python scripts/build_preference_data.py --weak-checkpoint checkpoints/gqa_base/step_2000.pt

chosen   = real ending of a held-out TinyStories story
rejected = the weak checkpoint's ending from the same opening

Writes data/preference/pairs.jsonl (one JSON pair per line, human-readable)
and data/preference/stats.json.

Owner: track D.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from slm.model.config import Config, gqa_base, mha_baseline
from slm.tokenizer.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build chosen/rejected story pairs")
    p.add_argument("--preset", choices=["mha", "gqa"], default="gqa")
    p.add_argument("--weak-checkpoint", required=True,
                   help="an EARLY base checkpoint, e.g. checkpoints/gqa_base/step_2000.pt")
    p.add_argument("--n-pairs", type=int, default=None, help="default: cfg.dpo.n_pairs (3000)")
    p.add_argument("--prefix-tokens", type=int, default=48, help="story opening length")
    p.add_argument("--max-response-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="data/preference/pairs.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg: Config = {"mha": mha_baseline, "gqa": gqa_base}[args.preset]()
    n_pairs = args.n_pairs or cfg.dpo.n_pairs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from slm.data.preference import (build_pairs, generate_rejected, load_story_prefixes,
                                     pair_length_stats, save_pairs, tokenize_pairs)
    from slm.model.transformer import Transformer
    from slm.training.utils import load_checkpoint

    tok = Tokenizer.load(cfg.data.tokenizer_path)

    # Ask for 10% extra: a few pairs drop out (identical text, too long once templated).
    stories = load_story_prefixes(tok, int(n_pairs * 1.1), args.prefix_tokens,
                                  args.max_response_tokens, seed=cfg.train.seed)
    print(f"{len(stories):,} held-out story openings")

    model = Transformer(cfg.model)
    state = load_checkpoint(args.weak_checkpoint, model=model, device=device, restore_rng=False)
    print(f"weak model: {args.weak_checkpoint} (step {state['step']})")

    rejected = generate_rejected(model, tok, [p for p, _ in stories],
                                 max_new_tokens=args.max_response_tokens,
                                 temperature=args.temperature, top_k=args.top_k,
                                 batch_size=args.batch_size, seed=cfg.train.seed, device=device)

    pairs = build_pairs(tok, stories, rejected)[:n_pairs]
    save_pairs(pairs, args.out)

    stats = pair_length_stats(tokenize_pairs(pairs, tok, cfg.model.max_seq_len))
    stats.update(weak_checkpoint=args.weak_checkpoint, weak_step=state["step"],
                 temperature=args.temperature, top_k=args.top_k,
                 prefix_tokens=args.prefix_tokens)
    stats_path = Path(args.out).with_name("stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\nwrote {len(pairs):,} pairs -> {args.out}")
    print(json.dumps(stats, indent=2))
    p = pairs[0]
    print(f"\nexample\n  opening : {p.input!r}\n  chosen  : {p.chosen[:200]!r}\n"
          f"  rejected: {p.rejected[:200]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
