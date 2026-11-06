"""Supervised fine-tuning loop.

Owner: track D (written under time pressure by track C).

Deliberately close to training/pretrain.py -- same optimizer, same schedule
shape, same checkpointing -- so there is one set of training machinery to
trust rather than two. The differences are all small and all deliberate:

  * learning rate 2e-5 rather than 3e-4. We are adjusting a model that
    already works; too large and it suffers catastrophic forgetting, picking
    up the answer format while losing the fluency pretraining built.
  * epochs over a small dataset rather than a step budget over a huge one.
  * the loss is masked (see data/alpaca.py) so only response tokens count.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from slm.data.alpaca import Example, make_sft_batches
from slm.tokenizer.tokenizer import Tokenizer
from slm.training.pretrain import compute_loss
from slm.training.utils import (
    MetricsLogger,
    apply_lr,
    configure_optimizer,
    format_time,
    get_autocast_context,
    get_lr,
    make_scaler,
    save_checkpoint,
)
from slm.utils.seed import set_seed


@torch.no_grad()
def evaluate(model: nn.Module, batches, *, precision: str, device: str) -> float:
    """Mean loss over held-out examples. Masked exactly as in training, so
    this measures response quality, not prompt reconstruction."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for x, y in batches():
        with get_autocast_context(precision, device):
            total += compute_loss(model, x, y).item()
        n += 1
    if was_training:
        model.train()
    return total / max(1, n)


def sft(
    model: nn.Module,
    cfg,
    tok: Tokenizer,
    train_examples: list[Example],
    val_examples: list[Example],
    *,
    run_name: str | None = None,
) -> nn.Module:
    s, t = cfg.sft, cfg.train
    device = t.device
    run_name = run_name or f"{cfg.run_name}_sft"

    set_seed(t.seed, deterministic=t.deterministic_data_order)
    model.to(device)

    optimizer = configure_optimizer(
        model, lr=s.lr, weight_decay=t.weight_decay,
        betas=(t.beta1, t.beta2), device=device,
    )
    scaler = make_scaler(t.precision)

    train_batches = make_sft_batches(train_examples, cfg, tok, shuffle=True)
    val_batches = make_sft_batches(val_examples, cfg, tok, shuffle=False)

    steps_per_epoch = len(train_examples) // s.micro_batch_size // s.grad_accum_steps
    max_steps = steps_per_epoch * s.epochs
    warmup = max(1, int(max_steps * s.warmup_ratio))

    ckpt_dir = Path("checkpoints") / run_name
    logger = MetricsLogger(Path("results") / run_name / "metrics.jsonl", run_name=run_name)
    logger.log(event="start", config=cfg.to_dict(),
               n_train=len(train_examples), n_val=len(val_examples),
               max_steps=max_steps)

    print(f"SFT {run_name}: {len(train_examples):,} train, {len(val_examples):,} val")
    print(f"  {s.epochs} epochs x {steps_per_epoch:,} steps = {max_steps:,} steps, "
          f"lr {s.lr}, warmup {warmup}")

    baseline = evaluate(model, val_batches, precision=t.precision, device=device)
    print(f"  before SFT: val_loss {baseline:.4f}  ppl {math.exp(min(baseline, 20)):.2f}")
    logger.log(event="eval", step=0, val_loss=round(baseline, 5),
               val_ppl=round(math.exp(min(baseline, 20)), 3))

    model.train()
    t_start = time.time()
    step = 0

    for ep in range(s.epochs):
        it = train_batches()
        exhausted = False
        while not exhausted:
            lr = get_lr(step, lr=s.lr, warmup_steps=warmup,
                        max_steps=max_steps, min_lr_ratio=t.min_lr_ratio)
            apply_lr(optimizer, lr)

            loss_sum = 0.0
            for _ in range(s.grad_accum_steps):
                try:
                    x, y = next(it)
                except StopIteration:
                    exhausted = True
                    break
                with get_autocast_context(t.precision, device):
                    loss = compute_loss(model, x, y) / s.grad_accum_steps
                scaler.scale(loss).backward()
                loss_sum += loss.item()

            if exhausted:
                break

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if step % t.log_interval == 0:
                print(f"  ep {ep} step {step:>5}/{max_steps}  loss {loss_sum:.4f}  "
                      f"ppl {math.exp(min(loss_sum, 20)):>7.2f}  lr {lr:.2e}  "
                      f"gn {grad_norm:.2f}")
                logger.log(event="train", step=step, epoch=ep,
                           loss=round(loss_sum, 5), lr=lr,
                           grad_norm=round(float(grad_norm), 4))
            step += 1

        val = evaluate(model, val_batches, precision=t.precision, device=device)
        print(f"  >> epoch {ep} done  val_loss {val:.4f}  "
              f"ppl {math.exp(min(val, 20)):.2f}")
        logger.log(event="eval", step=step, epoch=ep, val_loss=round(val, 5),
                   val_ppl=round(math.exp(min(val, 20)), 3))

        save_checkpoint(
            ckpt_dir / f"epoch_{ep}.pt",
            model=model, optimizer=optimizer, scaler=scaler, step=step,
            config=cfg.to_dict(), extra={"epoch": ep, "val_loss": val},
        )

    final = evaluate(model, val_batches, precision=t.precision, device=device)
    save_checkpoint(
        ckpt_dir / "final.pt",
        model=model, optimizer=optimizer, scaler=scaler, step=step,
        config=cfg.to_dict(),
        extra={"final_val_loss": final, "baseline_val_loss": baseline},
    )

    print(f"\ndone in {format_time(time.time() - t_start)}")
    print(f"  val_loss {baseline:.4f} -> {final:.4f}   "
          f"ppl {math.exp(min(baseline, 20)):.2f} -> {math.exp(min(final, 20)):.2f}")
    logger.log(event="done", step=step, val_loss=round(final, 5),
               val_ppl=round(math.exp(min(final, 20)), 3),
               baseline_val_loss=round(baseline, 5))
    logger.close()
    return model
