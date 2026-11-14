"""Evaluate base -> SFT -> DPO side by side.

    python scripts/evaluate.py \
        --base checkpoints/gqa_base/final.pt \
        --sft  checkpoints/gqa_sft/final.pt \
        --dpo  checkpoints/gqa_dpo/final.pt

Any subset of --base/--sft/--dpo works. For every model:
  1. perplexity on held-out TinyStories   (did fine-tuning hurt fluency?)
  2. samples on a fixed prompt set        (what did each stage change?)
  3. preference accuracy on held-out pairs (does it prefer real story endings?)
plus, when both --sft and --dpo are given, DPO's implicit-reward accuracy.

Writes <out>/metrics.json, samples.json and samples.md (default out:
results/eval, the Alpaca chain). For another SFT dataset, point it at that
chain's checkpoints and a separate folder:

    python scripts/evaluate.py --base checkpoints/gqa_base/final.pt         --sft checkpoints/gqa_sft_oasst1/final.pt         --dpo checkpoints/gqa_dpo_oasst1/final.pt --out results/eval_oasst1

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
    p = argparse.ArgumentParser(description="Evaluate base / SFT / DPO checkpoints")
    p.add_argument("--preset", choices=["mha", "gqa"], default="gqa")
    p.add_argument("--base")
    p.add_argument("--sft")
    p.add_argument("--dpo")
    p.add_argument("--pairs", default="data/preference/pairs.jsonl")
    p.add_argument("--n-test", type=int, default=200, help="must match scripts/dpo.py")
    p.add_argument("--n-ppl-stories", type=int, default=500)
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--out", default="results/eval",
                   help="output folder; use a new one per chain, e.g. results/eval_oasst1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg: Config = {"mha": mha_baseline, "gqa": gqa_base}[args.preset]()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = cfg.train.precision if device == "cuda" else "fp32"

    from slm.data.preference import load_pairs, split_pairs, tokenize_pairs
    from slm.evaluation.dpo_eval import implicit_reward_accuracy, pair_scores, preference_accuracy
    from slm.evaluation.instruction_eval import EVAL_PROMPTS, run_prompts
    from slm.evaluation.perplexity import load_heldout_stories, perplexity
    from slm.model.transformer import Transformer
    from slm.training.utils import load_checkpoint

    models = {k: v for k, v in [("base", args.base), ("sft", args.sft), ("dpo", args.dpo)] if v}
    if not models:
        raise SystemExit("give at least one of --base / --sft / --dpo")

    tok = Tokenizer.load(cfg.data.tokenizer_path)
    test_pairs = []
    if Path(args.pairs).exists():
        # Same split as scripts/dpo.py, so these pairs were never trained on.
        _, test_text = split_pairs(load_pairs(args.pairs), n_test=args.n_test, seed=cfg.train.seed)
        test_pairs = tokenize_pairs(test_text, tok, cfg.model.max_seq_len)
        print(f"{len(test_pairs)} held-out preference pairs")
    else:
        print(f"no {args.pairs}: skipping preference accuracy")
    stories = [] if args.skip_ppl else load_heldout_stories(tok, args.n_ppl_stories,
                                                            cfg.model.max_seq_len, cfg.train.seed)

    e = cfg.eval
    metrics, samples, scores = {}, {}, {}
    for name, path in models.items():
        print(f"\n== {name}: {path}")
        model = Transformer(cfg.model)
        state = load_checkpoint(path, model=model, device=device, restore_rng=False)
        model.to(device)
        m = {"checkpoint": path, "step": state["step"]}
        if name != "base":
            # which instruction data this model was tuned on (from its own config)
            m["sft_dataset"] = state["config"].get("sft", {}).get("dataset_name")

        if stories:
            m["perplexity"] = perplexity(model, stories, pad_id=tok.pad_id,
                                         precision=precision, device=device)
            print(f"  ppl {m['perplexity']['ppl']:.3f}")

        gen = run_prompts(model, tok, max_new_tokens=e.max_new_tokens,
                          temperature=e.temperature, top_k=e.top_k, seed=e.gen_seed, device=device)
        samples[name] = gen["samples"]
        m["generation"] = gen["scores"]
        print(f"  generation {gen['scores']}")

        if test_pairs:
            scores[name] = pair_scores(model, test_pairs, pad_id=tok.pad_id,
                                       precision=precision, device=device)
            m["preference"] = preference_accuracy(scores[name])
            print(f"  preference {m['preference']}")

        metrics[name] = m
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if "sft" in scores and "dpo" in scores:
        metrics["dpo"]["implicit_reward_vs_sft"] = implicit_reward_accuracy(
            scores["dpo"], scores["sft"], cfg.dpo.beta)
        print(f"\nDPO implicit reward vs SFT: {metrics['dpo']['implicit_reward_vs_sft']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    lines = ["# Samples: same prompts, same seed, every model", "",
             f"temperature {e.temperature}, top-k {e.top_k}, max {e.max_new_tokens} new tokens", ""]
    for k, (instruction, input_text) in enumerate(EVAL_PROMPTS):
        lines += [f"## {k + 1}. {instruction}", ""]
        if input_text:
            lines += [f"> {input_text}", ""]
        for name in samples:
            lines += [f"**{name}:** {samples[name][k]['response'] or '(empty)'}", ""]
    (out / "samples.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + summary_table(metrics))
    print(f"\nwrote {out}/metrics.json, samples.json, samples.md")
    return 0


def summary_table(metrics: dict) -> str:
    rows = ["| model | ppl | pref acc (sum) | pref acc (per-token) | eos rate | distinct-2 |",
            "|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        ppl = f"{m['perplexity']['ppl']:.3f}" if "perplexity" in m else "-"
        pa = m.get("preference", {})
        g = m["generation"]
        rows.append(f"| {name} | {ppl} | {pa.get('acc_sum', float('nan')):.3f} | "
                    f"{pa.get('acc_mean', float('nan')):.3f} | {g['eos_rate']:.2f} | {g['distinct_2']:.3f} |")
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
