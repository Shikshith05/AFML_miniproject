"""Phase 4: controlled MHA vs GQA benchmark.

Two experiments, both in a single process on a single GPU.

  A. training step throughput
     Settles the open question: the batch-size probe measured GQA ~3% faster,
     while the two pretraining runs suggested ~24%. Those runs were on
     different Kaggle sessions, so the gap may be machine variance rather
     than architecture. Here both variants are measured back to back on the
     same device, interleaved, with the spread reported.

  B. inference sweep
     KV-cache size, decode throughput and peak memory across context lengths
     and batch sizes. This is where GQA's saving actually lives: during
     generation the cache dominates memory, and it is 4x smaller.

Why interleave rather than run all of A then all of B: GPU clocks drift with
temperature and a shared Kaggle T4 may have other work on it. Alternating
variant order spreads any drift across both rather than loading it onto
whichever ran second.

Note on context lengths beyond 512: the models were TRAINED at 512, so
anything longer measures cost, not quality. Perplexity comparisons come from
the pretraining runs; this script measures only time and memory, for which
randomly initialised weights are equivalent.

    python scripts/benchmark_attention.py --out results/benchmark

Owner: tracks B + C.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from slm.model.config import ModelConfig, gqa_base, mha_baseline
from slm.model.kv_cache import KVCache
from slm.model.transformer import Transformer
from slm.training.pretrain import compute_loss
from slm.training.utils import (
    configure_optimizer,
    get_autocast_context,
    make_scaler,
)

VARIANTS = ("mha", "gqa")
DECODE_TOKENS = 32          # tokens generated per inference measurement


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def model_config(variant: str, max_seq_len: int) -> ModelConfig:
    """Config for a variant, with the context length overridden.

    Everything else comes from the presets, so the two stay parameter-matched
    (27,795,968 each) exactly as the pretraining runs were.
    """
    cfg = {"mha": mha_baseline, "gqa": gqa_base}[variant]().model
    cfg.max_seq_len = max_seq_len
    return cfg


# --------------------------------------------------------------------------
# A. training step throughput
# --------------------------------------------------------------------------

def time_training_steps(
    variant: str,
    *,
    batch_size: int,
    seq_len: int,
    device: str,
    precision: str,
    steps: int,
    warmup: int,
) -> dict:
    cfg = model_config(variant, seq_len)
    free_gpu()

    model = Transformer(cfg).to(device)
    optimizer = configure_optimizer(
        model, lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95), device=device
    )
    scaler = make_scaler(precision)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

    per_step = []
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with get_autocast_context(precision, device):
            loss = compute_loss(model, x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        if i >= warmup:
            per_step.append(time.perf_counter() - t0)

    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, optimizer
    free_gpu()

    mean = statistics.mean(per_step)
    return {
        "variant": variant,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "s_per_step_mean": mean,
        "s_per_step_std": statistics.stdev(per_step) if len(per_step) > 1 else 0.0,
        "tokens_per_s": batch_size * seq_len / mean,
        "peak_gb": peak,
        "n_samples": len(per_step),
    }


# --------------------------------------------------------------------------
# B. inference sweep
# --------------------------------------------------------------------------

@torch.no_grad()
def time_generation(
    variant: str,
    *,
    batch_size: int,
    context_len: int,
    device: str,
    precision: str,
    reps: int,
    warmup: int,
) -> dict | None:
    """Prefill `context_len` tokens, then decode DECODE_TOKENS one at a time.

    Decode is timed separately from prefill: prefill is a single parallel
    pass, while decode is the sequential part the KV cache exists for, and
    it is where cache size affects speed.
    """
    total = context_len + DECODE_TOKENS
    cfg = model_config(variant, total)
    cache_dtype = torch.float16 if precision == "fp16" else torch.float32
    free_gpu()

    try:
        model = Transformer(cfg).to(device).eval()
        prompt = torch.randint(0, cfg.vocab_size, (batch_size, context_len), device=device)

        prefill_times, decode_times = [], []
        cache_bytes = None

        for i in range(warmup + reps):
            cache = KVCache(
                cfg, batch_size=batch_size, max_seq_len=total,
                device=device, dtype=cache_dtype,
            )
            cache_bytes = cache.nbytes

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with get_autocast_context(precision, device):
                logits = model(prompt, kv_cache=cache)
            torch.cuda.synchronize()
            t_prefill = time.perf_counter() - t0

            nxt = logits[:, -1:, :].argmax(dim=-1)
            t0 = time.perf_counter()
            for _ in range(DECODE_TOKENS):
                with get_autocast_context(precision, device):
                    logits = model(nxt, kv_cache=cache)
                nxt = logits[:, -1:, :].argmax(dim=-1)
            torch.cuda.synchronize()
            t_decode = time.perf_counter() - t0

            if i >= warmup:
                prefill_times.append(t_prefill)
                decode_times.append(t_decode)
            del cache

        peak = torch.cuda.max_memory_allocated() / 1e9
        del model
        free_gpu()

        decode_mean = statistics.mean(decode_times)
        analytic = cfg.kv_cache_bytes(
            seq_len=total, batch_size=batch_size,
            dtype_bytes=2 if precision == "fp16" else 4,
        )
        return {
            "variant": variant,
            "batch_size": batch_size,
            "context_len": context_len,
            "kv_cache_bytes_measured": cache_bytes,
            "kv_cache_bytes_analytic": analytic,
            "kv_cache_match": cache_bytes == analytic,
            "prefill_s_mean": statistics.mean(prefill_times),
            "decode_s_mean": decode_mean,
            "decode_s_std": statistics.stdev(decode_times) if len(decode_times) > 1 else 0.0,
            "decode_tokens_per_s": batch_size * DECODE_TOKENS / decode_mean,
            "peak_gb": peak,
            "n_samples": len(decode_times),
        }

    except torch.cuda.OutOfMemoryError:
        free_gpu()
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            free_gpu()
            return None
        raise


# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="MHA vs GQA, same GPU, same session")
    p.add_argument("--out", type=str, default="results/benchmark")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    p.add_argument("--train-steps", type=int, default=20)
    p.add_argument("--train-reps", type=int, default=3,
                   help="interleaved A/B rounds for the training benchmark")
    p.add_argument("--train-batch", type=int, default=85)
    p.add_argument("--gen-reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--context-lens", type=int, nargs="+",
                   default=[128, 256, 512, 1024, 2048])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("No GPU. This benchmark is only meaningful on the GPU you trained on.")
        return 1

    device = "cuda"
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {name} ({total_gb:.1f} GB), precision {args.precision}")
    for v in VARIANTS:
        c = model_config(v, 512)
        print(f"  {v:>4}: {c.param_count_str()}")
    print()

    # ---- A. training throughput ------------------------------------------
    print("=" * 70)
    print(f"A. training step throughput  (batch {args.train_batch}, seq 512)")
    print("=" * 70)

    train_runs: list[dict] = []
    for rep in range(args.train_reps):
        # alternate which variant goes first, so order cannot favour one
        order = VARIANTS if rep % 2 == 0 else tuple(reversed(VARIANTS))
        for v in order:
            r = time_training_steps(
                v, batch_size=args.train_batch, seq_len=512, device=device,
                precision=args.precision, steps=args.train_steps,
                warmup=args.warmup,
            )
            r["rep"] = rep
            train_runs.append(r)
            print(f"  rep {rep}  {v:>4}  {r['s_per_step_mean']:.4f} s/step "
                  f"(+/- {r['s_per_step_std']:.4f})  "
                  f"{r['tokens_per_s']:>9,.0f} tok/s  {r['peak_gb']:.2f} GB")

    print()
    summary = {}
    for v in VARIANTS:
        rs = [r["tokens_per_s"] for r in train_runs if r["variant"] == v]
        summary[v] = {
            "tokens_per_s_mean": statistics.mean(rs),
            "tokens_per_s_std": statistics.stdev(rs) if len(rs) > 1 else 0.0,
        }
        print(f"  {v:>4}: {summary[v]['tokens_per_s_mean']:>9,.0f} tok/s "
              f"+/- {summary[v]['tokens_per_s_std']:,.0f}")

    speedup = (summary["gqa"]["tokens_per_s_mean"]
               / summary["mha"]["tokens_per_s_mean"] - 1) * 100
    spread = max(summary[v]["tokens_per_s_std"] /
                 summary[v]["tokens_per_s_mean"] for v in VARIANTS) * 100
    print()
    print(f"  GQA vs MHA: {speedup:+.1f}%   (run-to-run spread {spread:.1f}%)")
    if abs(speedup) < 2 * spread:
        print("  -> within noise: no training-speed difference demonstrated")
    else:
        print("  -> larger than the spread: a real difference")
    print("  Both models have identical FLOPs by construction (parameter-matched),")
    print("  so any real gap is memory traffic or kernel efficiency, not arithmetic.")

    # ---- B. inference sweep ----------------------------------------------
    print()
    print("=" * 70)
    print("B. inference: KV cache, decode throughput, peak memory")
    print("=" * 70)
    print(f"{'ctx':>6} {'batch':>6} {'variant':>8} {'cache MB':>10} "
          f"{'decode tok/s':>13} {'peak GB':>9}")
    print("-" * 70)

    gen_runs: list[dict] = []
    for ctx in args.context_lens:
        for bs in args.batch_sizes:
            for v in VARIANTS:
                r = time_generation(
                    v, batch_size=bs, context_len=ctx, device=device,
                    precision=args.precision, reps=args.gen_reps,
                    warmup=max(1, args.warmup // 2),
                )
                if r is None:
                    print(f"{ctx:>6} {bs:>6} {v:>8}        OOM")
                    continue
                if not r["kv_cache_match"]:
                    print(f"  WARNING: cache size mismatch for {v} "
                          f"ctx={ctx} batch={bs}: measured "
                          f"{r['kv_cache_bytes_measured']} vs analytic "
                          f"{r['kv_cache_bytes_analytic']}")
                gen_runs.append(r)
                print(f"{ctx:>6} {bs:>6} {v:>8} "
                      f"{r['kv_cache_bytes_measured'] / 1e6:>10.1f} "
                      f"{r['decode_tokens_per_s']:>13,.1f} "
                      f"{r['peak_gb']:>9.2f}")
            # saving at this (ctx, batch), if both fitted
            pair = {r["variant"]: r for r in gen_runs
                    if r["context_len"] == ctx and r["batch_size"] == bs}
            if len(pair) == 2:
                saved = (pair["mha"]["kv_cache_bytes_measured"]
                         - pair["gqa"]["kv_cache_bytes_measured"]) / 1e6
                ratio = (pair["mha"]["kv_cache_bytes_measured"]
                         / pair["gqa"]["kv_cache_bytes_measured"])
                faster = (pair["gqa"]["decode_tokens_per_s"]
                          / pair["mha"]["decode_tokens_per_s"] - 1) * 100
                print(f"{'':>6} {'':>6} {'saved':>8} {saved:>10.1f} "
                      f"{faster:>12.1f}% {ratio:>8.2f}x")

    # ---- write -----------------------------------------------------------
    payload = {
        "gpu": name,
        "total_memory_gb": round(total_gb, 2),
        "precision": args.precision,
        "decode_tokens": DECODE_TOKENS,
        "params": {v: model_config(v, 512).param_counts()["total"] for v in VARIANTS},
        "training": {"runs": train_runs, "summary": summary,
                     "gqa_vs_mha_pct": speedup, "spread_pct": spread},
        "inference": gen_runs,
    }
    (out / "benchmark.json").write_text(json.dumps(payload, indent=2))
    print()
    print(f"written to {out / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
