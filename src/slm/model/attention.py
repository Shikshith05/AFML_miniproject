"""Causal self-attention covering MHA, GQA and MQA with one class.

Owner: track B.

The variant is set by ModelConfig.n_kv_heads alone:

    n_kv_heads == n_heads      -> MHA  (every query head has its own K/V)
    1 < n_kv_heads < n_heads   -> GQA  (groups of n_rep query heads share K/V)
    n_kv_heads == 1            -> MQA  (all query heads share one K/V)

Q is projected to n_heads * head_dim, but K and V only to
n_kv_heads * head_dim. Each K/V head then serves n_rep = n_heads // n_kv_heads
query heads: by default K and V are repeated n_rep times (repeat_interleave)
just before SDPA. SDPA's own enable_gqa grouping is available as an opt-in
(SLM_SDPA_ENABLE_GQA=1), see decision S14 for why it is not the default. The
saving is in the K/V projections and, more importantly, in the KV cache, which
stores K and V at n_kv_heads on either path.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from slm.model.config import ModelConfig
from slm.model.layers import RotaryEmbedding

if TYPE_CHECKING:
    from slm.model.kv_cache import KVCache


def _probe_sdpa_gqa() -> bool:
    """Does this torch's SDPA broadcast K/V across query groups itself?

    torch >= 2.5 takes enable_gqa and does the grouping internally, with no
    full-size copy of K/V. Probed by making the call rather than by checking
    the version or the signature -- SDPA is a C builtin, so
    inspect.signature() raises on it.
    """
    try:
        q = torch.zeros(1, 2, 1, 8)   # 2 query heads
        kv = torch.zeros(1, 1, 1, 8)  # 1 KV head -> needs grouping
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except (TypeError, RuntimeError):
        return False


_SDPA_HAS_GQA = _probe_sdpa_gqa()

ENABLE_GQA_ENV = "SLM_SDPA_ENABLE_GQA"


def _resolve_use_enable_gqa(env: dict | None = None, sdpa_has_gqa: bool | None = None) -> bool:
    """Should GQA layers pass K/V to SDPA at n_kv_heads with enable_gqa=True?

    Opt-in only: True when SLM_SDPA_ENABLE_GQA=1 AND this torch supports the
    keyword. Default (unset, or any other value) is the repeat_interleave path.

    Why not the default: on a Kaggle T4 (sm_75, torch 2.11) enable_gqa=True
    drops SDPA to the math backend, and it measured worse than
    repeat_interleave on every count except the cache (which is the same 4x
    smaller on both paths):
      - training throughput: GQA -39.1 % vs MHA (spread 3.5 %, batch 48)
      - training peak memory: +58 % vs MHA
      - decode: slower at all 20 configs, -17 % to -71.3 % (ctx 1024, batch 64)
      - OOM at ctx 2048, batch 64, where repeat_interleave completed.
    Newer GPUs (Ampere+, FlashAttention) may keep a fused kernel with
    enable_gqa; that is untested here.
    """
    env = os.environ if env is None else env
    has = _SDPA_HAS_GQA if sdpa_has_gqa is None else sdpa_has_gqa
    return env.get(ENABLE_GQA_ENV, "") == "1" and has


# Read once at import. Tests monkeypatch this flag to force either path.
_USE_ENABLE_GQA = _resolve_use_enable_gqa()


def kv_path() -> str:
    """Name of the K/V expansion path GQA layers use in this process."""
    return "sdpa_enable_gqa" if _USE_ENABLE_GQA else "repeat_interleave"


class Attention(nn.Module):
    """Multi-head causal self-attention with grouped K/V heads and RoPE.

    Projection names match ModelConfig.param_counts() (q_proj, k_proj,
    v_proj, o_proj). No biases, per decision S1.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep  # query heads per K/V head
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """x [B, S, D] -> [B, S, D].

        start_pos: absolute position of x's first token (0 unless a KV cache
            already holds earlier tokens).
        kv_cache / layer_idx: during generation, this layer's new K/V are
            written into the cache and attention runs over everything stored
            so far.
        """
        B, S, _ = x.shape

        # Project, then split the last dim into heads and move heads ahead of
        # the sequence dim, the layout scaled_dot_product_attention expects.
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)     # [B, H, S, hd]
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, H_kv, S, hd]
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, H_kv, S, hd]

        # Position enters here, and only into q and k.
        q, k = self.rope(q, k, start_pos)

        if kv_cache is not None:
            # Cache stores K/V at n_kv_heads, never expanded to n_heads, which
            # is where GQA's memory saving comes from. Returns every K/V so far.
            k, v = kv_cache.update(layer_idx, start_pos, k, v)                         # [B, H_kv, T, hd]
            k, v = k.to(q.dtype), v.to(q.dtype)

        # Every query head needs a K/V head. Default: repeat_interleave copies
        # each K/V head n_rep times to [B, n_heads, T, hd] right before SDPA,
        # so SDPA sees plain MHA and keeps its fused (mem-efficient) kernel.
        # The copy is transient; the cache above still holds n_kv_heads.
        # Opt-in (SLM_SDPA_ENABLE_GQA=1, torch >= 2.5): pass K/V at n_kv_heads
        # with enable_gqa=True and let SDPA group them. Tried as the default
        # and reverted: on the T4 it falls back to the math backend -- GQA
        # training -39.1 % vs MHA, +58 % peak memory, decode slower at all 20
        # configs (down to -71.3 %), OOM at ctx 2048 batch 64. See S14.
        # Both paths give the same output (tests/test_attention.py).
        T_pre = k.size(2)
        if self.n_rep > 1 and not _USE_ENABLE_GQA:
            k = k.repeat_interleave(self.n_rep, dim=1)                                  # [B, H, T, hd]
            v = v.repeat_interleave(self.n_rep, dim=1)

        gqa_kw = {"enable_gqa": True} if (self.n_rep > 1 and _USE_ENABLE_GQA) else {}

        T = T_pre  # number of keys: S, plus anything already cached
        dropout_p = self.dropout if self.training else 0.0
        if T == S:
            # Plain forward pass: standard lower-triangular causal mask.
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p, **gqa_kw
            )
        else:
            # Cached step: the S queries are the *last* S of T positions, so
            # query i may see keys 0 .. (T - S + i). is_causal=True would align
            # the mask to the top-left corner instead, which is wrong here.
            mask = torch.ones(S, T, dtype=torch.bool, device=x.device).tril(diagonal=T - S)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p, **gqa_kw
            )

        # Merge heads back: [B, H, S, hd] -> [B, S, H * hd] -> [B, S, D]
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.resid_dropout(self.o_proj(out))
