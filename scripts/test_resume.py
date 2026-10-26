"""Resume-fidelity test.

The question this answers: if Kaggle kills a run at hour 12 and we restart
from the last checkpoint, does training continue exactly as if nothing
happened -- or does it silently diverge?

Method:
  Phase A  train 0..40 from scratch, checkpointing at step 20, record losses
  Phase B  delete everything after step 20, resume, train 20..40, record losses
  Compare  losses for steps 21..40 must match to float tolerance

A divergence means one of these is not being restored:
  - optimizer state (Adam's m/v buffers)   -> loss jumps, then recovers
  - RNG state                              -> different batches, noisy mismatch
  - GradScaler state (fp16)                -> occasional large mismatch
  - LR schedule position                   -> smooth but wrong trajectory

Runs on CPU in about a minute; no data files or GPU needed.

    python scripts/test_resume.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from slm.model.config import Config
from slm.model.transformer import Transformer
from slm.training.pretrain import pretrain
from slm.training.utils import read_metrics

STEPS_TOTAL = 40
STEPS_BREAK = 20
TOLERANCE = 1e-5


def tiny_config(run_name: str) -> Config:
    """Smallest config that still exercises every code path."""
    cfg = Config(run_name=run_name)
    cfg.model.n_layers = 2
    cfg.model.d_model = 64
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2        # exercise the GQA path, not just MHA
    cfg.model.ffn_hidden = 128
    cfg.model.vocab_size = 256
    cfg.model.max_seq_len = 32

    cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.train.precision = "fp16" if torch.cuda.is_available() else "fp32"
    cfg.train.micro_batch_size = 4
    cfg.train.grad_accum_steps = 2
    cfg.train.max_steps = STEPS_TOTAL
    cfg.train.warmup_ratio = 0.1
    cfg.train.log_interval = 1          # need every step
    cfg.train.eval_interval = 10_000    # skip eval; it is not what we test
    cfg.train.checkpoint_interval = STEPS_BREAK
    cfg.train.seed = 1337

    cfg.eval.eval_batches = 2
    return cfg


def synthetic_batches(cfg: Config):
    """Random-token batches drawn from torch's global RNG.

    Using the global RNG is the point: it is the same stream the real data
    loader draws from, so if the checkpoint fails to restore RNG state, this
    test sees different batches after resume and the losses diverge.
    """
    device = cfg.train.device
    bs = cfg.train.micro_batch_size
    seq = cfg.model.max_seq_len
    vocab = cfg.model.vocab_size

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        buf = torch.randint(0, vocab, (bs, seq + 1))
        x, y = buf[:, :-1].contiguous(), buf[:, 1:].contiguous()
        return x.to(device), y.to(device)

    return get_batch


def train_losses(cfg: Config, *, resume: bool) -> dict[int, float]:
    torch.manual_seed(cfg.train.seed)
    model = Transformer(cfg.model)
    pretrain(model, cfg, synthetic_batches(cfg), resume=resume)
    return {
        r["step"]: r["loss"]
        for r in read_metrics(cfg.metrics_path)
        if r.get("event") == "train"
    }


def main() -> int:
    run = "resume_test"
    cfg = tiny_config(run)

    # clean slate
    for d in (cfg.checkpoint_dir, cfg.results_dir):
        shutil.rmtree(d, ignore_errors=True)

    print("=" * 62)
    print(f"PHASE A -- train 0..{STEPS_TOTAL} uninterrupted")
    print("=" * 62)
    losses_a = train_losses(cfg, resume=False)

    # Keep only the step-20 checkpoint, and move phase A's metrics aside, so
    # phase B starts from exactly the state a killed session would leave.
    keep = Path(cfg.checkpoint_dir) / f"step_{STEPS_BREAK:06d}.pt"
    if not keep.exists():
        print(f"\nFAIL: expected checkpoint {keep} was never written")
        return 1
    for p in Path(cfg.checkpoint_dir).glob("*.pt"):
        if p != keep:
            p.unlink()
    Path(cfg.metrics_path).rename(Path(cfg.metrics_path).with_suffix(".phase_a.jsonl"))

    print()
    print("=" * 62)
    print(f"PHASE B -- resume from step {STEPS_BREAK}, train to {STEPS_TOTAL}")
    print("=" * 62)
    losses_b = train_losses(cfg, resume=True)

    # ---- compare ----------------------------------------------------------
    steps = sorted(s for s in losses_b if s > STEPS_BREAK)
    if not steps:
        print("\nFAIL: phase B logged no steps after the break")
        return 1

    print()
    print("=" * 62)
    print(f"{'step':>6} {'phase A':>14} {'phase B':>14} {'|diff|':>12}")
    print("-" * 62)

    worst = 0.0
    for s in steps:
        if s not in losses_a:
            print(f"{s:>6}  missing from phase A")
            return 1
        a, b = losses_a[s], losses_b[s]
        d = abs(a - b)
        worst = max(worst, d)
        flag = "" if d <= TOLERANCE else "  <-- MISMATCH"
        print(f"{s:>6} {a:>14.8f} {b:>14.8f} {d:>12.2e}{flag}")

    print("-" * 62)
    print(f"worst difference: {worst:.3e}   tolerance: {TOLERANCE:.0e}")
    print()

    if worst <= TOLERANCE:
        print("PASS -- resume reproduces the loss curve.")
        print("Safe to run long jobs under Kaggle's 12-hour session limit.")
        return 0

    print("FAIL -- the resumed run diverges. Check, in this order:")
    print("  1. optimizer state_dict saved and loaded (Adam m/v)")
    print("  2. RNG state saved and loaded (same batch stream)")
    print("  3. GradScaler state, if running fp16")
    print("  4. LR schedule computed from step, not from scheduler state")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
