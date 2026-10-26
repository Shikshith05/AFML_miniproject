"""RMSNorm, SwiGLU feed-forward and rotary position embeddings (RoPE).

Owner: track B.

The Llama-style building blocks around attention (decision S1 in
docs/decisions.md). Attention itself lives in attention.py.

Shape letters used in comments:
    B = batch, S = sequence length, D = d_model,
    H = query heads, H_kv = key/value heads, hd = head_dim
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# RMSNorm
# --------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Rescale each token's vector to root-mean-square 1, then apply a learned
    per-feature gain.

    LayerNorm without the mean subtraction or the bias: cheaper, and in
    practice just as good. `eps` stops a division by zero on an all-zero
    vector.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Starts at 1, so at init the layer is a pure normalisation.
        # It is 1-D, so configure_optimizer leaves it out of weight decay.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Statistics in fp32: under fp16 autocast, x**2 overflows past 65504
        # and a mean over 512 small squares loses precision.
        x32 = x.float()
        inv_rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * inv_rms * self.weight).type_as(x)


# --------------------------------------------------------------------------
# SwiGLU feed-forward
# --------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """Feed-forward block: w_down( silu(w_gate(x)) * w_up(x) ).

    The silu(w_gate(x)) branch is a soft per-unit gate on w_up(x). Three
    matrices instead of the classic two, which is why ffn_hidden is
    8/3 * d_model rather than 4 * d_model -- it keeps the parameter count
    about the same.

    Attribute names match ModelConfig.param_counts() (w_gate, w_up, w_down).
    No biases, per decision S1.
    """

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, S, D] -> [B, S, hidden] -> [B, S, D]
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# --------------------------------------------------------------------------
# Rotary position embeddings
# --------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1], where x1, x2 are the two halves of the last dim.

    Feature i is paired with feature i + hd/2. Combined with the cos/sin
    tables below, this rotates every pair (x1[i], x2[i]) by its own angle:

        x1' = x1 * cos - x2 * sin
        x2' = x2 * cos + x1 * sin
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x [..., S, hd] by per-position angles cos/sin [S, hd]."""
    # fp32 for the same reason as RMSNorm, then back to x's dtype.
    x32 = x.float()
    return (x32 * cos + rotate_half(x32) * sin).type_as(x)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings, applied to queries and keys (not values).

    Each head vector of size hd is treated as hd/2 pairs. Pair i of the token
    at position p is rotated by the angle p * freq_i, with

        freq_i = 1 / theta ** (2i / hd),    i = 0 .. hd/2 - 1

    so early pairs spin fast (fine position) and late pairs spin slowly
    (coarse position). Rotating both q and k means their dot product only
    sees the *difference* of the angles: the attention score between
    positions m and n depends on m - n alone.

    Layout is the "half split" pairing (i with i + hd/2) used by
    HuggingFace's Llama, not the adjacent pairs of the original paper. The
    two are equivalent up to a fixed reordering of features.

    cos/sin are precomputed for every position up to max_seq_len and kept as
    non-persistent buffers: they follow .to(device) but stay out of
    state_dict, so checkpoints hold only learned weights.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even -- RoPE rotates pairs")
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [hd/2]
        positions = torch.arange(max_seq_len).float()                                   # [max_seq_len]
        angles = torch.outer(positions, inv_freq)                                       # [max_seq_len, hd/2]
        # Both members of a pair share an angle, so repeat for the second half.
        angles = torch.cat((angles, angles), dim=-1)                                    # [max_seq_len, hd]

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate q [B, H, S, hd] and k [B, H_kv, S, hd].

        `start_pos` is the absolute position of the first token in this call:
        0 for a normal forward pass, and the number of tokens already in the
        KV cache during generation, so each new token gets its true position.
        """
        end = start_pos + q.size(-2)
        if end > self.max_seq_len:
            raise ValueError(
                f"positions up to {end} requested but max_seq_len is {self.max_seq_len}"
            )
        # [S, hd] -- broadcasts over the batch and head dims
        cos = self.cos[start_pos:end]
        sin = self.sin[start_pos:end]
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
