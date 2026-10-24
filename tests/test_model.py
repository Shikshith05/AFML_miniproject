"""Test the language model."""

import pytest
import torch

from slm.model.config import ModelConfig
from slm.model.kv_cache import KVCache
from slm.model.transformer import Transformer

VOCAB = 4096


def make_model(n_kv_heads: int) -> Transformer:
    torch.manual_seed(0)
    return Transformer(ModelConfig(n_kv_heads=n_kv_heads)).eval()


# --------------------------------------------------------------------------
# Forward pass
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_kv_heads", [8, 2, 1])
@torch.no_grad()
def test_output_shape(n_kv_heads):
    model = make_model(n_kv_heads)
    x = torch.randint(0, VOCAB, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, VOCAB)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("t", [1, 7, 15])
@torch.no_grad()
def test_causality(t):
    """Changing token t must leave logits at positions < t unchanged."""
    model = make_model(n_kv_heads=2)
    x = torch.randint(0, VOCAB, (2, 16))
    x_changed = x.clone()
    x_changed[:, t] = (x[:, t] + 1) % VOCAB

    a, b = model(x), model(x_changed)

    torch.testing.assert_close(a[:, :t], b[:, :t])
    # ...and the change must be visible from t on, or the test proves nothing
    assert not torch.allclose(a[:, t:], b[:, t:])


def test_embeddings_are_tied():
    model = make_model(n_kv_heads=2)
    assert model.lm_head.weight is model.tok_emb.weight
    # parameters() yields the shared matrix once
    n_matrices = sum(1 for p in model.parameters() if p.shape == (VOCAB, 512))
    assert n_matrices == 1


@torch.no_grad()
def test_state_dict_round_trip():
    """What save_checkpoint writes, load_checkpoint must restore exactly --
    including the tie, which a naive copy would break."""
    a, b = make_model(n_kv_heads=2), Transformer(ModelConfig(n_kv_heads=2)).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, VOCAB, (2, 16))
    torch.testing.assert_close(a(x), b(x))
    assert b.lm_head.weight is b.tok_emb.weight
    # RoPE tables are recomputed, not stored
    assert not any("rope" in k for k in a.state_dict())


# --------------------------------------------------------------------------
# KV cache
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_kv_heads", [8, 2])
@torch.no_grad()
def test_cached_generation_matches_uncached(n_kv_heads):
    """Greedy generation with and without a KV cache gives the same logits."""
    model = make_model(n_kv_heads)
    prompt = torch.randint(0, VOCAB, (2, 8))
    n_new = 8

    # Uncached: re-run the whole sequence for every new token.
    seq = prompt
    uncached = []
    for _ in range(n_new):
        logits = model(seq)[:, -1]
        uncached.append(logits)
        seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)

    # Cached: prefill the prompt once, then feed one token per step. Feed the
    # tokens the uncached run chose, so both runs see identical inputs.
    cache = KVCache(model.cfg, batch_size=2)
    cached = [model(prompt, kv_cache=cache)[:, -1]]
    for i in range(prompt.size(1), seq.size(1) - 1):
        cached.append(model(seq[:, i : i + 1], kv_cache=cache)[:, -1])

    assert cache.seq_len == seq.size(1) - 1
    for a, b in zip(uncached, cached):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    # same logits -> same greedy choices
    assert torch.equal(
        torch.stack([a.argmax(-1) for a in uncached]),
        torch.stack([b.argmax(-1) for b in cached]),
    )


@torch.no_grad()
def test_chunked_prefill_matches_full_forward():
    """Feeding a sequence in uneven chunks through the cache must reproduce
    the full forward pass. Chunks longer than one token on top of a non-empty
    cache exercise the offset causal mask in Attention."""
    model = make_model(n_kv_heads=2)
    x = torch.randint(0, VOCAB, (2, 16))
    full = model(x)

    cache = KVCache(model.cfg, batch_size=2)
    pieces, start = [], 0
    for size in (5, 3, 1, 4, 1, 2):
        pieces.append(model(x[:, start : start + size], kv_cache=cache))
        start += size
    assert start == 16

    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("n_kv_heads", [8, 2, 1])
def test_kv_cache_size_matches_config(n_kv_heads):
    """The cache holds K/V at n_kv_heads, so its size is exactly what
    ModelConfig.kv_cache_bytes() predicts."""
    cfg = ModelConfig(n_kv_heads=n_kv_heads)
    cache = KVCache(cfg, batch_size=4, max_seq_len=256, dtype=torch.float16)
    assert cache.k.shape == (8, 4, n_kv_heads, 256, 64)
    assert cache.nbytes == cfg.kv_cache_bytes(seq_len=256, batch_size=4, dtype_bytes=2)


def test_gqa_cache_is_4x_smaller_than_mha():
    mha = KVCache(ModelConfig(n_kv_heads=8), batch_size=1, dtype=torch.float16)
    gqa = KVCache(ModelConfig(n_kv_heads=2), batch_size=1, dtype=torch.float16)
    assert mha.nbytes == 8_388_608   # 2 * 8 layers * 8 heads * 64 * 512 * 2 bytes
    assert gqa.nbytes == 2_097_152   # same with 2 heads
    assert mha.nbytes == 4 * gqa.nbytes


def test_kv_cache_rejects_overflow():
    model = make_model(n_kv_heads=2)
    cache = KVCache(model.cfg, batch_size=1, max_seq_len=8)
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 8)), kv_cache=cache)
        with pytest.raises(ValueError, match="cache full"):
            model(torch.randint(0, VOCAB, (1, 1)), kv_cache=cache)
