"""Optimizers, learning-rate schedule, checkpointing and metrics logging.

Owner: track C.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn

from slm.utils.seed import get_rng_state, set_rng_state


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------

def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    device: str = "cuda",
) -> torch.optim.AdamW:
    """AdamW with weight decay applied only to matrices.

    Decaying 1-D parameters (norm gains, biases) pulls them toward zero for no
    benefit -- they have no redundant directions to regularize. Standard
    practice is to split them into their own group with decay 0.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    # fused AdamW is noticeably faster but CUDA-only
    use_fused = device.startswith("cuda") and torch.cuda.is_available()
    opt = torch.optim.AdamW(groups, lr=lr, betas=betas, fused=use_fused)

    n_decay = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    print(
        f"optimizer: {len(decay)} decayed tensors ({n_decay:,} params), "
        f"{len(no_decay)} undecayed ({n_no_decay:,} params), fused={use_fused}"
    )
    return opt


# --------------------------------------------------------------------------
# Learning-rate schedule
# --------------------------------------------------------------------------

def get_lr(
    step: int,
    *,
    lr: float,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup, then cosine decay to lr * min_lr_ratio.

    Computed from `step` rather than held as scheduler state, so resuming is
    exact -- there is nothing to restore and get wrong.
    """
    min_lr = lr * min_lr_ratio

    if step < warmup_steps:
        # step+1 so the first step isn't exactly zero
        return lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
    return min_lr + coeff * (lr - min_lr)


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# --------------------------------------------------------------------------
# Precision
# --------------------------------------------------------------------------

def get_autocast_context(precision: str, device: str):
    """Autocast context for the forward pass.

    A Kaggle T4 is Turing and has no bf16 support -- use fp16 there, which
    needs a GradScaler (see make_scaler).
    """
    if precision == "fp32" or not device.startswith("cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_scaler(precision: str) -> torch.amp.GradScaler:
    """Gradient scaler. Only fp16 needs one: its range bottoms out around
    6e-8, so small gradients silently flush to zero without scaling. bf16 has
    fp32's exponent range and needs none.
    """
    return torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    step: int,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write everything needed to resume as if nothing happened.

    The optimizer state is not optional: Adam's m and v buffers take hundreds
    of steps to re-estimate, and a resume without them shows up as a visible
    jump in the loss curve.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "config": config,
        "rng": get_rng_state(),
        "extra": extra or {},
    }

    # Write to a temp file and rename: a crash mid-write otherwise leaves a
    # corrupt checkpoint where the last good one used to be.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: str = "cuda",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint in place. Returns the payload minus the tensors."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng and ckpt.get("rng") is not None:
        set_rng_state(ckpt["rng"])

    return {
        "step": ckpt["step"],
        "config": ckpt.get("config", {}),
        "extra": ckpt.get("extra", {}),
    }


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    """Most recent step_*.pt in a directory, by step number."""
    directory = Path(directory)
    if not directory.exists():
        return None
    ckpts = list(directory.glob("step_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

class MetricsLogger:
    """Append-only JSONL log.

    One line per event, flushed immediately -- a killed session keeps
    everything written up to that point.
    """

    def __init__(self, path: str | Path, run_name: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self._file = self.path.open("a", buffering=1)
        self._t0 = time.time()

    def log(self, **kwargs: Any) -> None:
        record = {
            "run": self.run_name,
            "wall_s": round(time.time() - self._t0, 2),
            **kwargs,
        }
        self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_metrics(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL metrics file back, skipping any truncated final line."""
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad or not trainable_only
    )


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"
