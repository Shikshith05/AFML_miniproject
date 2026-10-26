"""Count model parameters and compare matched models.

Owner: track B.

Counts come from the built model (model.named_parameters()), so they are an
independent check on the formula in ModelConfig.param_counts().

    python -m slm.utils.params      # MHA vs GQA table
"""

from __future__ import annotations

import dataclasses

import torch.nn as nn

from slm.model.config import ModelConfig

EMBEDDING_PREFIXES = ("tok_emb.", "lm_head.")


def count_params(model: nn.Module) -> dict[str, int]:
    """Total, embedding and non-embedding parameter counts.

    named_parameters() yields a tied matrix once, so the shared
    embedding/lm_head weight is counted once. "Non-embedding" excludes the
    token embedding (and an untied lm_head), matching
    ModelConfig.param_counts() -- it is the number scaling laws use.
    """
    total = embedding = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if name.startswith(EMBEDDING_PREFIXES):
            embedding += p.numel()
    return {"total": total, "embedding": embedding, "non_embedding": total - embedding}


def param_breakdown(model: nn.Module) -> dict[str, int]:
    """Parameters per component, summed over all layers."""
    groups = dict.fromkeys(("embedding", "attention", "ffn", "norms", "lm_head"), 0)
    for name, p in model.named_parameters():
        if name.startswith("tok_emb."):
            groups["embedding"] += p.numel()
        elif name.startswith("lm_head."):       # only appears when untied
            groups["lm_head"] += p.numel()
        elif "norm" in name:                    # before "attn": attn_norm is a norm
            groups["norms"] += p.numel()
        elif ".attn." in name:
            groups["attention"] += p.numel()
        elif ".ffn." in name:
            groups["ffn"] += p.numel()
        else:
            raise ValueError(f"unclassified parameter: {name}")
    return {**groups, **count_params(model)}


def matched_ffn_hidden(cfg: ModelConfig, reference: ModelConfig) -> int:
    """ffn_hidden that gives `cfg` the same total parameters as `reference`.

    For the ablation's option O2: GQA's smaller K/V projections leave it with
    fewer parameters than MHA; spend the difference on the FFN. Each extra
    hidden unit adds 3 * d_model weights per layer (w_gate, w_up, w_down).
    Rounded to the nearest unit, so the match is exact only when the gap
    divides evenly -- it does for the project's MHA vs GQA-2 presets.
    """
    gap = reference.param_counts()["total"] - cfg.param_counts()["total"]
    per_unit = 3 * cfg.d_model * cfg.n_layers
    return cfg.ffn_hidden + round(gap / per_unit)


def compare(models: dict[str, nn.Module]) -> str:
    """Side-by-side breakdown table of several models."""
    rows = {name: param_breakdown(m) for name, m in models.items()}
    width = max(14, *(len(n) + 2 for n in rows))
    keys = next(iter(rows.values())).keys()
    lines = [" " * 14 + "".join(f"{n:>{width}}" for n in rows)]
    for k in keys:
        lines.append(f"{k:<14}" + "".join(f"{r[k]:>{width},}" for r in rows.values()))
    return "\n".join(lines)


if __name__ == "__main__":
    from slm.model.config import gqa_base, mha_baseline
    from slm.model.transformer import Transformer

    mha, gqa = mha_baseline().model, gqa_base().model
    matched = dataclasses.replace(gqa, ffn_hidden=matched_ffn_hidden(gqa, mha))
    print(compare({
        "MHA": Transformer(mha),
        "GQA-2": Transformer(gqa),
        f"GQA-2 ffn={matched.ffn_hidden}": Transformer(matched),
    }))
