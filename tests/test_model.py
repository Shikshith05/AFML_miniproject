"""Test the language model."""

import pytest
import torch

from slm.model.config import ModelConfig
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
