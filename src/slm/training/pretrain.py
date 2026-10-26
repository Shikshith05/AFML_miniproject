"""Pretraining loop.

Owner: track C.

Decoupled from the model (track B) and the data pipeline (track A) by two
contracts:

  model(x) -> logits  [B, S, vocab]        or (logits, loss) if it takes targets
  get_batch(split)    -> (x, y) on device, both [B, S] int64

So this file can be written, read and reviewed before either exists.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.training.utils import (
    MetricsLogger,
    apply_lr,
    configure_optimizer,
    find_latest_checkpoint,
    format_time,
    get_autocast_context,
    get_lr,
    load_checkpoint,
    make_scaler,
    save_checkpoint,
)
from slm.utils.seed import set_seed


class BatchProvider(Protocol):
    def __call__(self, split: str) -> tuple[torch.Tensor, torch.Tensor]: ...


def compute_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy.

    ignore_index=-100 matters later: SFT and DPO mask prompt tokens with it so
    only response tokens contribute. Pretraining has nothing masked, but
    sharing the code path keeps the three stages consistent.
    """
    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    get_batch: BatchProvider,
    *,
    n_batches: int,
    precision: str,
    device: str,
    split: str = "val",
) -> float:
    """Mean loss over n_batches. Averaging matters -- a single validation
    batch is noisy enough to hide real differences between runs."""
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(split)
        with get_autocast_context(precision, device):
            total += compute_loss(model, x, y).item()
    if was_training:
        model.train()
    return total / n_batches


def pretrain(
    model: nn.Module,
    cfg,
    get_batch: BatchProvider,
    *,
    resume: bool = True,
    on_eval: Callable[[int, float], None] | None = None,
) -> nn.Module:
    """Train `model` according to `cfg` (a Config from slm.model.config).

    Resumes from the latest checkpoint in cfg.checkpoint_dir unless
    resume=False.
    """
    t, m = cfg.train, cfg.model
    device = t.device

    set_seed(t.seed, deterministic=t.deterministic_data_order)
    model.to(device)

    optimizer = configure_optimizer(
        model,
        lr=t.lr,
        weight_decay=t.weight_decay,
        betas=(t.beta1, t.beta2),
        device=device,
    )
    scaler = make_scaler(t.precision)

    start_step = 0
    if resume:
        latest = find_latest_checkpoint(cfg.checkpoint_dir)
        if latest is not None:
            state = load_checkpoint(
                latest,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )
            start_step = state["step"] + 1
            print(f"resumed from {latest} at step {start_step}")

    if t.compile_model:
        model = torch.compile(model)

    logger = MetricsLogger(cfg.metrics_path, run_name=cfg.run_name)
    logger.log(event="start", step=start_step, config=cfg.to_dict())

    tokens_per_step = t.tokens_per_step(m.max_seq_len)
    model.train()
    t_start = time.time()
    t_window = time.time()

    try:
        for step in range(start_step, t.max_steps):
            lr = get_lr(
                step,
                lr=t.lr,
                warmup_steps=t.warmup_steps,
                max_steps=t.max_steps,
                min_lr_ratio=t.min_lr_ratio,
            )
            apply_lr(optimizer, lr)

            # ---- gradient accumulation -----------------------------------
            loss_sum = 0.0
            for _ in range(t.grad_accum_steps):
                x, y = get_batch("train")
                with get_autocast_context(t.precision, device):
                    loss = compute_loss(model, x, y)
                    # Divide so the accumulated sum is a mean. Forgetting this
                    # makes the effective learning rate grad_accum_steps times
                    # too large.
                    loss = loss / t.grad_accum_steps
                scaler.scale(loss).backward()
                loss_sum += loss.item()

            # ---- step ----------------------------------------------------
            # unscale before clipping, or the clip threshold is applied to
            # scaled gradients and means nothing.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), t.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # ---- logging -------------------------------------------------
            if step % t.log_interval == 0:
                dt = time.time() - t_window
                t_window = time.time()
                tok_per_s = (
                    tokens_per_step * t.log_interval / dt if dt > 0 else 0.0
                )
                elapsed = time.time() - t_start
                eta = (
                    elapsed / (step - start_step + 1) * (t.max_steps - step)
                    if step > start_step
                    else 0.0
                )
                print(
                    f"step {step:>6}/{t.max_steps}  "
                    f"loss {loss_sum:.4f}  ppl {math.exp(min(loss_sum, 20)):>8.2f}  "
                    f"lr {lr:.2e}  gn {grad_norm:.2f}  "
                    f"{tok_per_s:,.0f} tok/s  eta {format_time(eta)}"
                )
                logger.log(
                    event="train",
                    step=step,
                    loss=round(loss_sum, 5),
                    lr=lr,
                    grad_norm=round(float(grad_norm), 4),
                    tokens=step * tokens_per_step,
                    tokens_per_s=round(tok_per_s, 1),
                )

            # ---- evaluation ----------------------------------------------
            if step > 0 and step % t.eval_interval == 0:
                val_loss = estimate_loss(
                    model,
                    get_batch,
                    n_batches=cfg.eval.eval_batches,
                    precision=t.precision,
                    device=device,
                )
                print(
                    f"  >> step {step}  val_loss {val_loss:.4f}  "
                    f"val_ppl {math.exp(min(val_loss, 20)):.2f}"
                )
                logger.log(
                    event="eval",
                    step=step,
                    val_loss=round(val_loss, 5),
                    val_ppl=round(math.exp(min(val_loss, 20)), 3),
                    tokens=step * tokens_per_step,
                )
                if on_eval is not None:
                    on_eval(step, val_loss)

            # ---- checkpoint ----------------------------------------------
            if step > 0 and step % t.checkpoint_interval == 0:
                path = Path(cfg.checkpoint_dir) / f"step_{step:06d}.pt"
                save_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    config=cfg.to_dict(),
                )
                print(f"  >> checkpoint {path}")
                logger.log(event="checkpoint", step=step, path=str(path))

    except KeyboardInterrupt:
        print("\ninterrupted -- saving before exit")
        save_checkpoint(
            Path(cfg.checkpoint_dir) / f"step_{step:06d}_interrupt.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            config=cfg.to_dict(),
        )
        raise

    # ---- final -------------------------------------------------------------
    final_loss = estimate_loss(
        model,
        get_batch,
        n_batches=cfg.eval.eval_batches,
        precision=t.precision,
        device=device,
    )
    final_path = Path(cfg.checkpoint_dir) / "final.pt"
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=t.max_steps - 1,
        config=cfg.to_dict(),
        extra={"final_val_loss": final_loss},
    )

    print(
        f"\ndone in {format_time(time.time() - t_start)}  "
        f"final val_loss {final_loss:.4f}  "
        f"val_ppl {math.exp(min(final_loss, 20)):.2f}"
    )
    logger.log(
        event="done",
        step=t.max_steps,
        val_loss=round(final_loss, 5),
        val_ppl=round(math.exp(min(final_loss, 20)), 3),
    )
    logger.close()
    return model
