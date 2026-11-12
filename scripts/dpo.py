"""Run DPO on the SFT model.

    python scripts/dpo.py --sft-checkpoint checkpoints/gqa_sft/final.pt

    # from the oasst1 SFT model, kept apart from the Alpaca run:
    python scripts/dpo.py --sft-checkpoint checkpoints/gqa_sft_oasst1/final.pt         --run-name gqa_dpo_oasst1

The SFT model is both the starting policy and the frozen reference. The
reference log-probs are computed first (and cached), then the same model is
trained -- so only one copy is ever in GPU memory.

Output: checkpoints/<run-name>/final.pt, results/<run-name>/metrics.jsonl
(--run-name defaults to <preset>_dpo, i.e. gqa_dpo, the Alpaca run).

Reference cache: data/ref_logprobs_{train,test}.npz for the default run name,
data/ref_logprobs_<run-name>_{train,test}.npz otherwise (or --ref-cache). Each
file is fingerprinted with both the pairs and the reference model's weights,
so a cache from another SFT checkpoint is recomputed, never reused.

Owner: track D.
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
    p = argparse.ArgumentParser(description="Direct preference optimization")
    p.add_argument("--preset", choices=["mha", "gqa"], default="gqa")
    p.add_argument("--sft-checkpoint", required=True)
    p.add_argument("--pairs", default="data/preference/pairs.jsonl")
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-test", type=int, default=200, help="held-out pairs for reward accuracy")
    p.add_argument("--limit", type=int, default=None, help="use only N pairs (smoke test)")
    p.add_argument("--run-name", default=None,
                   help="output folder name under checkpoints/ and results/ "
                        "(default: <preset>_dpo). Use a new one per SFT checkpoint, "
                        "e.g. gqa_dpo_oasst1")
    p.add_argument("--ref-cache", default=None,
                   help="reference log-prob cache stem; _train.npz/_test.npz are "
                        "appended (default: data/ref_logprobs for the default run "
                        "name, data/ref_logprobs_<run-name> otherwise)")
    return p.parse_args()


def ref_cache_stem(explicit: str | None, run_name: str, preset: str, default: str) -> str:
    """Where the reference log-probs go. The default run keeps the original
    path; any other run gets its own file, so two runs never share a cache
    (the fingerprint would catch it anyway, but recomputing in place would
    throw away the other run's cache)."""
    if explicit:
        return explicit
    if run_name == f"{preset}_dpo":
        return default
    d = Path(default)
    return str(d.with_name(f"{d.stem}_{run_name}{d.suffix}"))


def warn_if_run_exists(run_name: str) -> None:
    old = [p for p in (Path("results") / run_name / "metrics.jsonl",
                       Path("checkpoints") / run_name / "final.pt") if p.exists()]
    if old:
        print(f"WARNING: run '{run_name}' already has {', '.join(map(str, old))}. "
              "The log will be appended to and the checkpoint replaced; pass a "
              "different --run-name to keep the earlier run.")


def main() -> int:
    args = parse_args()
    cfg: Config = {"mha": mha_baseline, "gqa": gqa_base}[args.preset]()
    if args.beta is not None:
        cfg.dpo.beta = args.beta
    if args.lr is not None:
        cfg.dpo.lr = args.lr
    if args.epochs is not None:
        cfg.dpo.epochs = args.epochs
    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: no CUDA -- falling back to cpu")
        cfg.train.device, cfg.train.precision = "cpu", "fp32"
    run_name = args.run_name or f"{args.preset}_dpo"
    cfg.run_name = run_name

    from slm.data.preference import load_pairs, pair_length_stats, split_pairs, tokenize_pairs
    from slm.model.transformer import Transformer
    from slm.training.dpo import dpo_train, load_or_compute_reference
    from slm.training.utils import load_checkpoint

    tok = Tokenizer.load(cfg.data.tokenizer_path)
    pairs = load_pairs(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    train_text, test_text = split_pairs(pairs, n_test=args.n_test, seed=cfg.train.seed)
    train = tokenize_pairs(train_text, tok, cfg.model.max_seq_len)
    test = tokenize_pairs(test_text, tok, cfg.model.max_seq_len)
    print(f"pairs: {len(train):,} train / {len(test):,} test   {pair_length_stats(train)}")

    model = Transformer(cfg.model)
    state = load_checkpoint(args.sft_checkpoint, model=model, device=cfg.train.device,
                            restore_rng=False)
    print(f"loaded {args.sft_checkpoint} (step {state['step']}) as policy AND reference")
    # Carry the SFT stage's settings (dataset name in particular) into the DPO
    # checkpoint's config, so it records which SFT model it was built on.
    if isinstance(state["config"].get("sft"), dict):
        cfg.sft = type(cfg.sft)(**{k: v for k, v in state["config"]["sft"].items()
                                   if k in cfg.sft.__dataclass_fields__})
    print(f"  SFT data {cfg.sft.dataset_name}  ->  checkpoints/{run_name}/, results/{run_name}/")
    warn_if_run_exists(run_name)

    cache = Path(ref_cache_stem(args.ref_cache, run_name, args.preset,
                                cfg.dpo.ref_logprobs_cache))
    print(f"  reference cache {cache.with_name(cache.stem + '_{train,test}.npz')}")
    kw = dict(pad_id=tok.pad_id, batch_size=cfg.dpo.micro_batch_size * 2,
              precision=cfg.train.precision, device=cfg.train.device)
    ref_train = load_or_compute_reference(cache.with_name(cache.stem + "_train.npz"), model, train, **kw)
    ref_test = load_or_compute_reference(cache.with_name(cache.stem + "_test.npz"), model, test, **kw)

    dpo_train(model, cfg, tok, train, test, ref_train, ref_test, run_name=run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
