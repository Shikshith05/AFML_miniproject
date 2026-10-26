"""Test multi-head and grouped-query attention."""

import math

import pytest
import torch

from slm.model.attention import Attention
from slm.model.config import ModelConfig
from slm.model.layers import RotaryEmbedding, apply_rotary


def reference_mha(attn: Attention, x: torch.Tensor) -> torch.Tensor:
    """Textbook causal MHA, one head at a time, using attn's own weights.

    Deliberately naive -- explicit mask, explicit softmax, no fused kernel,
    no repeat_interleave -- so it can only agree with Attention if the
    projections, head split, RoPE, masking and merge are all right.
    """
    B, S, _ = x.shape
    hd = attn.head_dim
    cos, sin = attn.rope.cos[:S], attn.rope.sin[:S]
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool))

    heads = []
    for h in range(attn.n_heads):
        rows = slice(h * hd, (h + 1) * hd)  # this head's slice of each projection
        q = apply_rotary(x @ attn.q_proj.weight[rows].T, cos, sin)  # [B, S, hd]
        k = apply_rotary(x @ attn.k_proj.weight[rows].T, cos, sin)
        v = x @ attn.v_proj.weight[rows].T

        scores = q @ k.transpose(-1, -2) / math.sqrt(hd)            # [B, S, S]
        scores = scores.masked_fill(~causal, float("-inf"))
        heads.append(scores.softmax(dim=-1) @ v)                    # [B, S, hd]

    return torch.cat(heads, dim=-1) @ attn.o_proj.weight.T


@torch.no_grad()
def test_mha_matches_reference():
    torch.manual_seed(0)
    cfg = ModelConfig(n_kv_heads=8)  # n_kv_heads == n_heads -> plain MHA
    attn = Attention(cfg).eval()
    x = torch.randn(2, 16, cfg.d_model)

    assert attn.n_rep == 1
    torch.testing.assert_close(attn(x), reference_mha(attn, x), atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_gqa_equals_mha_with_shared_kv_weights():
    """GQA is MHA whose K/V heads come in identical groups of n_rep.

    Build an MHA layer whose K/V weights are the GQA weights with each head
    copied n_rep times; the outputs must match. This checks that query head h
    really uses K/V head h // n_rep.
    """
    torch.manual_seed(0)
    gqa = Attention(ModelConfig(n_kv_heads=2)).eval()
    mha = Attention(ModelConfig(n_kv_heads=8)).eval()

    def expand(w: torch.Tensor) -> torch.Tensor:
        # [H_kv * hd, D] -> [H_kv, hd, D] -> repeat heads -> [H * hd, D]
        per_head = w.view(gqa.n_kv_heads, gqa.head_dim, -1)
        return per_head.repeat_interleave(gqa.n_rep, dim=0).reshape(-1, w.size(1))

    mha.q_proj.weight.copy_(gqa.q_proj.weight)
    mha.o_proj.weight.copy_(gqa.o_proj.weight)
    mha.k_proj.weight.copy_(expand(gqa.k_proj.weight))
    mha.v_proj.weight.copy_(expand(gqa.v_proj.weight))

    x = torch.randn(2, 16, 512)
    torch.testing.assert_close(gqa(x), mha(x), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("n_kv_heads, variant", [(8, "MHA"), (2, "GQA-2"), (1, "MQA")])
@torch.no_grad()
def test_variants_shapes(n_kv_heads, variant):
    cfg = ModelConfig(n_kv_heads=n_kv_heads)
    attn = Attention(cfg)
    x = torch.randn(2, 16, cfg.d_model)

    assert cfg.attention_variant == variant
    assert attn.q_proj.out_features == 8 * 64
    assert attn.k_proj.out_features == n_kv_heads * 64
    assert attn.v_proj.out_features == n_kv_heads * 64
    assert attn(x).shape == (2, 16, cfg.d_model)


@torch.no_grad()
def test_rope_score_depends_only_on_relative_position():
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=64, max_seq_len=512)
    q = torch.randn(1, 1, 1, 64)
    k = torch.randn(1, 1, 1, 64)

    def score(m: int, n: int) -> torch.Tensor:
        qm, _ = rope(q, q, start_pos=m)
        _, kn = rope(k, k, start_pos=n)
        return (qm * kn).sum()

    # same distance, different absolute positions -> same score
    torch.testing.assert_close(score(5, 2), score(105, 102), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(score(5, 2), score(400, 397), atol=1e-4, rtol=1e-4)
    # different distance -> different score
    assert not torch.allclose(score(5, 2), score(5, 4), atol=1e-3)
    # a rotation never changes a vector's length
    torch.testing.assert_close(rope(q, q, start_pos=300)[0].norm(), q.norm())
