"""Direct Preference Optimization (Rafailov et al., 2023), from scratch.

Owner: track D.

Idea in one paragraph: for each prompt we have a chosen and a rejected
response. We want the policy to raise the probability of chosen relative to
rejected -- but measured *relative to a frozen reference model* (the SFT
model), so it cannot simply drift off and forget how to write. The loss is

    margin = beta * [ (log pi(chosen) - log ref(chosen))
                    - (log pi(rejected) - log ref(rejected)) ]
    loss   = -log sigmoid(margin)

log pi(y) is the SUM of log-probabilities of the response tokens (prompt
tokens are masked with -100, exactly as in SFT). beta sets how far the policy
may move from the reference: small beta = weak pull back to the reference.

"Implicit reward" of a response = beta * (log pi(y) - log ref(y)). Reward
accuracy = fraction of pairs where chosen's reward > rejected's -- the main
number to watch during training.

Reference log-probs never change (the reference is frozen), so they are
computed once and cached to disk. Training then needs only one model in
GPU memory.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.data.alpaca import IGNORE_INDEX
from slm.data.preference import TokenizedPair, collate_pairs
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


# -- the maths ------------------------------------------------------------------

def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log p(label) over supervised positions, one number per row.

    logits [B, T, V], labels [B, T] (already shifted: labels[t] is the token
    that follows input[t]); -100 positions contribute nothing.
    """
    logp = F.log_softmax(logits.float(), dim=-1)                 # fp32: fp16 log-probs lose precision
    mask = labels != IGNORE_INDEX
    safe = labels.masked_fill(~mask, 0)                          # any valid index; masked out below
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)   # [B, T]
    return (tok_logp * mask).sum(dim=-1)                         # [B]


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All inputs [B] summed log-probs. Returns (loss, chosen_reward, rejected_reward).

    The rewards are detached: they are for logging, not for the gradient.
    """
    chosen_logratio = policy_chosen - ref_chosen
    rejected_logratio = policy_rejected - ref_rejected
    margin = beta * (chosen_logratio - rejected_logratio)
    loss = -F.logsigmoid(margin).mean()      # logsigmoid, not log(sigmoid()): stable for large |margin|
    return loss, (beta * chosen_logratio).detach(), (beta * rejected_logratio).detach()


def pair_logprobs(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward over the stacked [chosen; rejected] batch -> (chosen [B], rejected [B])."""
    logits = model(x)
    return sequence_logprobs(logits, y).chunk(2)


# -- reference log-probs (computed once) -----------------------------------------

@torch.no_grad()
def compute_reference_logprobs(
    ref_model: nn.Module,
    pairs: list[TokenizedPair],
    *,
    pad_id: int,
    batch_size: int = 16,
    precision: str = "fp16",
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    ref_model.eval().to(device)
    chosen, rejected = [], []
    for i in range(0, len(pairs), batch_size):
        x, y = collate_pairs(pairs[i : i + batch_size], pad_id, device)
        with get_autocast_context(precision, device):
            c, r = pair_logprobs(ref_model, x, y)
        chosen.append(c.float().cpu())
        rejected.append(r.float().cpu())
    return torch.cat(chosen).numpy(), torch.cat(rejected).numpy()


def pairs_fingerprint(pairs: list[TokenizedPair]) -> str:
    """Hash of every token in every pair: the cache is only valid for these exact pairs."""
    import hashlib

    h = hashlib.sha256()
    for p in pairs:
        for ex in (p.chosen, p.rejected):
            h.update(np.asarray(ex.input_ids, dtype=np.int32).tobytes())
            h.update(np.asarray(ex.labels, dtype=np.int32).tobytes())
    return h.hexdigest()[:16]


def model_fingerprint(model: nn.Module) -> str:
    """Hash of every weight: the cache is only valid for this exact reference model.

    Pairs alone are not enough -- DPO from a different SFT checkpoint (say
    oasst1 instead of Alpaca) on the same pairs would otherwise reuse the old
    model's log-probs and train against the wrong reference without any error.
    """
    import hashlib

    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def load_or_compute_reference(path, ref_model, pairs, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Load cached reference log-probs, or compute and save them.

    A cache built from different pairs, or from a different reference model,
    would silently give wrong rewards, so the file stores a fingerprint of
    both and is recomputed on any mismatch. (Caches written before the model
    fingerprint existed have no `model_fingerprint` and are recomputed once.)
    """
    path = Path(path)
    fp = pairs_fingerprint(pairs)
    mfp = model_fingerprint(ref_model)
    if path.exists():
        d = np.load(path)
        cached_mfp = str(d["model_fingerprint"]) if "model_fingerprint" in d.files else None
        if str(d["fingerprint"]) != fp:
            print(f"reference log-probs: {path} is for different pairs, recomputing")
        elif cached_mfp != mfp:
            print(f"reference log-probs: {path} is for a different reference model "
                  f"({cached_mfp} != {mfp}), recomputing")
        else:
            print(f"reference log-probs: loaded {path}")
            return d["chosen"], d["rejected"]
    print(f"reference log-probs: computing for {len(pairs):,} pairs ...")
    c, r = compute_reference_logprobs(ref_model, pairs, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, chosen=c, rejected=r, fingerprint=fp, model_fingerprint=mfp)
    return c, r


# -- evaluation ------------------------------------------------------------------

@torch.no_grad()
def evaluate_dpo(model, pairs, ref_c, ref_r, *, beta, pad_id, batch_size, precision, device) -> dict:
    """Held-out pairs: loss, reward accuracy, mean margin."""
    was_training = model.training
    model.eval()
    losses, acc, margins = [], [], []
    for i in range(0, len(pairs), batch_size):
        x, y = collate_pairs(pairs[i : i + batch_size], pad_id, device)
        with get_autocast_context(precision, device):
            pc, pr = pair_logprobs(model, x, y)
        rc = torch.as_tensor(ref_c[i : i + batch_size], device=device)
        rr = torch.as_tensor(ref_r[i : i + batch_size], device=device)
        loss, cw, rw = dpo_loss(pc, pr, rc, rr, beta)
        losses.append(loss.item() * len(pc))
        acc += (cw > rw).tolist()
        margins += (cw - rw).tolist()
    if was_training:
        model.train()
    n = max(1, len(acc))
    return {"loss": sum(losses) / n, "reward_acc": float(np.mean(acc)) if acc else 0.0,
            "margin": float(np.mean(margins)) if margins else 0.0}


# -- training loop ----------------------------------------------------------------

def dpo_train(
    model: nn.Module,
    cfg,
    tok,
    train_pairs: list[TokenizedPair],
    test_pairs: list[TokenizedPair],
    ref_train: tuple[np.ndarray, np.ndarray],
    ref_test: tuple[np.ndarray, np.ndarray],
    *,
    run_name: str = "gqa_dpo",
) -> nn.Module:
    """Same machinery as sft.py (AdamW, warmup+cosine, fp16 scaler, grad clip)."""
    d, t = cfg.dpo, cfg.train
    device = t.device
    set_seed(t.seed, deterministic=t.deterministic_data_order)
    model.to(device)

    optimizer = configure_optimizer(model, lr=d.lr, weight_decay=0.0,   # no decay: stay near the reference
                                    betas=(t.beta1, t.beta2), device=device)
    scaler = make_scaler(t.precision)

    bs, accum = d.micro_batch_size, d.grad_accum_steps
    steps_per_epoch = len(train_pairs) // bs // accum
    max_steps = max(1, steps_per_epoch * d.epochs)
    warmup = max(1, int(0.1 * max_steps))
    eval_kw = dict(beta=d.beta, pad_id=tok.pad_id, batch_size=bs * 2,
                   precision=t.precision, device=device)

    ckpt_dir = Path("checkpoints") / run_name
    logger = MetricsLogger(Path("results") / run_name / "metrics.jsonl", run_name=run_name)
    logger.log(event="start", config=cfg.to_dict(), n_train=len(train_pairs),
               n_test=len(test_pairs), max_steps=max_steps)
    print(f"DPO {run_name}: {len(train_pairs):,} train pairs, {len(test_pairs):,} test, "
          f"{max_steps} steps, beta {d.beta}, lr {d.lr}")

    before = evaluate_dpo(model, test_pairs, *ref_test, **eval_kw)
    print(f"  before: test loss {before['loss']:.4f}  reward_acc {before['reward_acc']:.3f}  "
          f"(policy == reference, so loss = log 2 = 0.693)")
    logger.log(event="eval", step=0, **before)

    ref_c_all = torch.as_tensor(ref_train[0])
    ref_r_all = torch.as_tensor(ref_train[1])
    model.train()
    t_start = time.time()
    step = 0
    log_every = max(1, min(t.log_interval, max_steps // 20))

    for ep in range(d.epochs):
        order = torch.randperm(len(train_pairs)).tolist()
        for s in range(steps_per_epoch):
            lr = get_lr(step, lr=d.lr, warmup_steps=warmup, max_steps=max_steps,
                        min_lr_ratio=t.min_lr_ratio)
            apply_lr(optimizer, lr)

            loss_sum, acc_sum, margin_sum = 0.0, 0.0, 0.0
            for a in range(accum):
                idx = order[(s * accum + a) * bs : (s * accum + a + 1) * bs]
                x, y = collate_pairs([train_pairs[i] for i in idx], tok.pad_id, device)
                rc, rr = ref_c_all[idx].to(device), ref_r_all[idx].to(device)
                with get_autocast_context(t.precision, device):
                    pc, pr = pair_logprobs(model, x, y)
                loss, cw, rw = dpo_loss(pc, pr, rc, rr, d.beta)
                scaler.scale(loss / accum).backward()
                loss_sum += loss.item() / accum
                acc_sum += (cw > rw).float().mean().item() / accum
                margin_sum += (cw - rw).mean().item() / accum

            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                print(f"  ep {ep} step {step:>4}/{max_steps}  loss {loss_sum:.4f}  "
                      f"acc {acc_sum:.3f}  margin {margin_sum:+.3f}  lr {lr:.2e}  gn {gn:.2f}")
                logger.log(event="train", step=step, epoch=ep, loss=round(loss_sum, 5),
                           reward_acc=round(acc_sum, 4), margin=round(margin_sum, 4),
                           lr=lr, grad_norm=round(float(gn), 4))
            step += 1

        res = evaluate_dpo(model, test_pairs, *ref_test, **eval_kw)
        print(f"  >> epoch {ep}: test loss {res['loss']:.4f}  reward_acc {res['reward_acc']:.3f}  "
              f"margin {res['margin']:+.3f}")
        logger.log(event="eval", step=step, epoch=ep, **res)

    save_checkpoint(ckpt_dir / "final.pt", model=model, optimizer=optimizer, scaler=scaler,
                    step=step, config=cfg.to_dict(), extra={"test": res, "before": before})
    print(f"\ndone in {format_time(time.time() - t_start)} -> {ckpt_dir / 'final.pt'}")
    logger.log(event="done", step=step, **res)
    logger.close()
    return model
