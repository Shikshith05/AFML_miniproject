"""Find the largest batch size that fits, measure throughput, size the run.

Three jobs, in one pass on the GPU you will actually train on:

  1. Binary-search micro_batch_size up to the OOM boundary
  2. Time real training steps at that size
  3. Convert a wall-clock budget into max_steps

Run this once per GPU type before launching anything long. The numbers it
prints go straight into the config and into docs/decisions.md.

    python scripts/probe_batch_size.py --preset gqa --budget-hours 3

Owner: track C.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from slm.model.config import Config, gqa_base, mha_baseline
from slm.model.transformer import Transformer
from slm.training.pretrain import compute_loss
from slm.training.utils import (
    apply_lr,
    configure_optimizer,
    get_autocast_context,
    make_scaler,
)

TARGET_EFFECTIVE_BATCH = 128


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def try_batch_size(cfg: Config, micro_bs: int, *, steps: int = 3) -> dict | None:
    """One real training step at this batch size. None if it OOMs.

    Uses the actual forward/backward/step path rather than a forward pass
    alone -- gradients and optimizer state are most of the memory, so a
    forward-only probe reports a batch size that will not survive training.
    """
    device, seq = cfg.train.device, cfg.model.max_seq_len
    free_gpu()

    model = optimizer = None
    try:
        model = Transformer(cfg.model).to(device)
        optimizer = configure_optimizer(
            model,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=(cfg.train.beta1, cfg.train.beta2),
            device=device,
        )
        scaler = make_scaler(cfg.train.precision)
        apply_lr(optimizer, cfg.train.lr)
        model.train()

        x = torch.randint(0, cfg.model.vocab_size, (micro_bs, seq), device=device)
        y = torch.randint(0, cfg.model.vocab_size, (micro_bs, seq), device=device)

        for i in range(steps):
            if i == 1:                      # discard step 0: allocator warmup
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
            with get_autocast_context(cfg.train.precision, device):
                loss = compute_loss(model, x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) / (steps - 1)

        peak = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else 0.0
        )
        return {
            "micro_batch_size": micro_bs,
            "s_per_micro_step": elapsed,
            "tokens_per_s": micro_bs * seq / elapsed,
            "peak_gb": peak,
        }

    except torch.cuda.OutOfMemoryError:
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return None
        raise
    finally:
        del model, optimizer
        free_gpu()


def main() -> int:
    p = argparse.ArgumentParser(description="Probe batch size and throughput")
    p.add_argument("--preset", choices=["mha", "gqa"], default="gqa")
    p.add_argument("--budget-hours", type=float, default=3.0)
    p.add_argument("--max-try", type=int, default=128)
    p.add_argument("--safety", type=float, default=0.9,
                   help="back off from the OOM boundary by this factor")
    p.add_argument("--out", type=str, default="results/probe.json")
    args = p.parse_args()

    cfg = {"mha": mha_baseline, "gqa": gqa_base}[args.preset]()

    if not torch.cuda.is_available():
        print("No GPU. This probe only means something on the GPU you will")
        print("train on -- run it on Kaggle, not locally.")
        return 1

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # T4 is Turing and has no bf16 support
    if cfg.train.precision == "bf16" and not torch.cuda.is_bf16_supported():
        print(f"note: {name} has no bf16 support -- switching to fp16")
        cfg.train.precision = "fp16"

    print(f"GPU: {name} ({total_gb:.1f} GB)")
    print(f"model: {cfg.model.param_count_str()}")
    print(f"precision: {cfg.train.precision}, seq {cfg.model.max_seq_len}")
    print()

    # ---- find the ceiling: double until OOM, then bisect ------------------
    print("probing batch size ...")
    lo, hi, best = 1, None, None

    bs = 1
    while bs <= args.max_try:
        r = try_batch_size(cfg, bs)
        if r is None:
            print(f"  {bs:>4}  OOM")
            hi = bs
            break
        print(f"  {bs:>4}  ok   {r['peak_gb']:.2f} GB   "
              f"{r['tokens_per_s']:>8,.0f} tok/s")
        best, lo = r, bs
        bs *= 2

    if hi is not None:
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            r = try_batch_size(cfg, mid)
            if r is None:
                print(f"  {mid:>4}  OOM")
                hi = mid
            else:
                print(f"  {mid:>4}  ok   {r['peak_gb']:.2f} GB   "
                      f"{r['tokens_per_s']:>8,.0f} tok/s")
                best, lo = r, mid

    if best is None:
        print("\nnothing fits, not even batch size 1 -- shrink the model")
        return 1

    # Back off from the boundary: fragmentation and the occasional longer
    # batch make the exact maximum a bad place to sit for hours.
    safe_bs = max(1, int(best["micro_batch_size"] * args.safety))
    if safe_bs != best["micro_batch_size"]:
        r = try_batch_size(cfg, safe_bs)
        if r is not None:
            best = r

    # ---- turn it into settings -------------------------------------------
    micro = best["micro_batch_size"]
    accum = max(1, round(TARGET_EFFECTIVE_BATCH / micro))
    tokens_per_step = micro * accum * cfg.model.max_seq_len
    s_per_step = best["s_per_micro_step"] * accum
    budget_s = args.budget_hours * 3600
    max_steps = int(budget_s / s_per_step)

    print()
    print("=" * 58)
    print(f"  micro_batch_size      {micro}")
    print(f"  grad_accum_steps      {accum}   (effective batch "
          f"{micro * accum})")
    print(f"  peak memory           {best['peak_gb']:.2f} GB of {total_gb:.1f}")
    print(f"  throughput            {best['tokens_per_s']:,.0f} tok/s")
    print(f"  time per step         {s_per_step:.2f} s")
    print(f"  tokens per step       {tokens_per_step:,}")
    print("-" * 58)
    print(f"  for a {args.budget_hours:g}-hour run:")
    print(f"    max_steps           {max_steps:,}")
    print(f"    total tokens        {max_steps * tokens_per_step / 1e9:.2f}B")
    print("=" * 58)
    print()
    print("Use the SAME values for both the MHA and GQA runs, even if MHA")
    print("could take a smaller batch -- matched conditions are what make")
    print("the comparison valid.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gpu": name,
        "total_memory_gb": round(total_gb, 2),
        "preset": args.preset,
        "precision": cfg.train.precision,
        "micro_batch_size": micro,
        "grad_accum_steps": accum,
        "peak_gb": round(best["peak_gb"], 3),
        "tokens_per_s": round(best["tokens_per_s"], 1),
        "s_per_step": round(s_per_step, 3),
        "tokens_per_step": tokens_per_step,
        "budget_hours": args.budget_hours,
        "max_steps": max_steps,
    }, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
