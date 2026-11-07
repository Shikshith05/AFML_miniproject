"""Evaluate language-model perplexity on held-out stories.

Owner: track D.

Used as a forgetting check: SFT and DPO change the model, and perplexity on
plain TinyStories shows whether story-writing ability survived. A large rise
from base -> SFT -> DPO would mean the fine-tuning stages traded fluency for
format.

Stories come from the TAIL of the shuffled HF validation split. The DPO
pairs (data/preference.py) take stories from the HEAD of the same shuffle,
so the two never overlap.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from slm.training.utils import get_autocast_context


def load_heldout_stories(tok, n: int = 500, max_len: int = 512, seed: int = 1337) -> list[list[int]]:
    from slm.data.tinystories import clean_text, load_tinystories

    ds = load_tinystories("validation")
    order = np.random.default_rng(seed).permutation(len(ds))[::-1]   # tail first
    out = []
    for i in order:
        ids = tok.encode(clean_text(ds[int(i)]["text"]).strip(), add_bos=True, add_eos=True)
        if 8 < len(ids) <= max_len:
            out.append(ids)
        if len(out) == n:
            break
    return out


@torch.no_grad()
def perplexity(model, stories: list[list[int]], *, pad_id: int, batch_size: int = 16,
               precision: str = "fp16", device: str = "cuda") -> dict:
    """Token-weighted perplexity: exp(total NLL / total predicted tokens)."""
    model.eval().to(device)
    nll, n_tok = 0.0, 0
    for i in range(0, len(stories), batch_size):
        batch = stories[i : i + batch_size]
        width = max(len(s) for s in batch)
        x = torch.full((len(batch), width), pad_id, dtype=torch.long)
        y = torch.full((len(batch), width), -100, dtype=torch.long)
        for j, s in enumerate(batch):
            x[j, : len(s)] = torch.tensor(s)
            y[j, : len(s)] = torch.tensor(s)
        x, y = x[:, :-1].to(device), y[:, 1:].to(device)
        with get_autocast_context(precision, device):
            logits = model(x)
        nll += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-100, reduction="sum").item()
        n_tok += (y != -100).sum().item()
    loss = nll / max(1, n_tok)
    return {"loss": loss, "ppl": math.exp(min(loss, 20)), "n_stories": len(stories), "n_tokens": n_tok}
