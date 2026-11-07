"""Evaluate preference accuracy on held-out pairs.

Owner: track D.

For each model and each held-out (chosen, rejected) pair, does the model
assign the chosen response higher log-probability? Two versions:

  * sum      -- total log p(response). Biased toward SHORT responses
                (every extra token adds a negative number).
  * mean     -- log p per token. Removes the length effect.

Both are reported because they can disagree, and the disagreement itself
tells you whether length is driving the result.

For the DPO model there is also the implicit-reward accuracy against the
SFT reference (the quantity DPO directly optimises); that comes from
training/dpo.evaluate_dpo.
"""

from __future__ import annotations

import numpy as np
import torch

from slm.data.preference import TokenizedPair, collate_pairs
from slm.training.dpo import sequence_logprobs
from slm.training.utils import get_autocast_context


@torch.no_grad()
def pair_scores(model, pairs: list[TokenizedPair], *, pad_id: int, batch_size: int = 16,
                precision: str = "fp16", device: str = "cuda") -> dict[str, np.ndarray]:
    """Per pair: summed log-probs and token counts for chosen and rejected."""
    model.eval().to(device)
    lc, lr = [], []
    for i in range(0, len(pairs), batch_size):
        x, y = collate_pairs(pairs[i : i + batch_size], pad_id, device)
        with get_autocast_context(precision, device):
            logits = model(x)
        c, r = sequence_logprobs(logits, y).float().chunk(2)
        lc.append(c.cpu())
        lr.append(r.cpu())
    return {
        "chosen": torch.cat(lc).numpy(),
        "rejected": torch.cat(lr).numpy(),
        "n_chosen": np.array([p.chosen.n_supervised for p in pairs]),
        "n_rejected": np.array([p.rejected.n_supervised for p in pairs]),
    }


def preference_accuracy(scores: dict[str, np.ndarray]) -> dict:
    s = scores
    mean_c, mean_r = s["chosen"] / s["n_chosen"], s["rejected"] / s["n_rejected"]
    return {
        "acc_sum": float(np.mean(s["chosen"] > s["rejected"])),
        "acc_mean": float(np.mean(mean_c > mean_r)),
        "mean_logp_per_token_chosen": float(mean_c.mean()),
        "mean_logp_per_token_rejected": float(mean_r.mean()),
        "n_pairs": int(len(s["chosen"])),
    }


def implicit_reward_accuracy(policy: dict, reference: dict, beta: float) -> dict:
    """DPO's own metric: beta * (log pi - log ref), chosen vs rejected."""
    rc = beta * (policy["chosen"] - reference["chosen"])
    rr = beta * (policy["rejected"] - reference["rejected"])
    return {"reward_acc": float(np.mean(rc > rr)), "reward_margin": float(np.mean(rc - rr))}
