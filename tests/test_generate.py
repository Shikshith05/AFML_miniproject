"""Test generation: sampling rules and that the KV cache changes nothing.

Owner: track D.
"""

import pytest
import torch

from slm.generation.generate import generate, sample_next_token, strip_after_eos
from slm.model.config import ModelConfig
from slm.model.transformer import Transformer


def tiny_model(n_kv_heads: int = 2) -> Transformer:
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, n_kv_heads=n_kv_heads,
                      max_seq_len=64, vocab_size=97)
    return Transformer(cfg).eval()


# -- sample_next_token --------------------------------------------------------

def test_greedy_picks_argmax():
    logits = torch.tensor([[0.1, 3.0, -1.0], [2.0, 0.0, 5.0]])
    assert sample_next_token(logits, temperature=0).tolist() == [[1], [2]]


def test_top_k_never_samples_outside_k():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    g = torch.Generator().manual_seed(0)
    picks = {sample_next_token(logits, 1.0, top_k=2, generator=g).item() for _ in range(500)}
    assert picks <= {0, 1}


def test_low_temperature_concentrates_on_top_token():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    g = torch.Generator().manual_seed(0)
    hot = sum(sample_next_token(logits, 2.0, generator=g).item() == 0 for _ in range(1000))
    cold = sum(sample_next_token(logits, 0.2, generator=g).item() == 0 for _ in range(1000))
    assert cold > hot


def test_seeded_sampling_is_reproducible():
    logits = torch.randn(3, 50)
    a = sample_next_token(logits, 1.0, 10, torch.Generator().manual_seed(7))
    b = sample_next_token(logits, 1.0, 10, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


# -- generate -----------------------------------------------------------------

@pytest.mark.parametrize("n_kv_heads", [4, 2, 1], ids=["MHA", "GQA", "MQA"])
def test_cache_matches_no_cache_greedy(n_kv_heads):
    """The KV cache is a speed trick only: greedy output must be identical."""
    model = tiny_model(n_kv_heads)
    prompt = torch.randint(0, 97, (2, 5))
    with_cache = generate(model, prompt, max_new_tokens=20, temperature=0, use_cache=True)
    without = generate(model, prompt, max_new_tokens=20, temperature=0, use_cache=False)
    assert torch.equal(with_cache, without)


def test_output_starts_with_prompt_and_has_right_length():
    model = tiny_model()
    prompt = torch.randint(0, 97, (3, 4))
    out = generate(model, prompt, max_new_tokens=10, temperature=1.0)
    assert out.shape == (3, 14)
    assert torch.equal(out[:, :4], prompt)


def test_stops_at_max_seq_len():
    model = tiny_model()                         # max_seq_len 64
    prompt = torch.randint(0, 97, (1, 60))
    out = generate(model, prompt, max_new_tokens=100, temperature=0)
    assert out.shape[1] == 64


def test_eos_stops_and_pads_finished_rows():
    model = tiny_model()
    prompt = torch.randint(0, 97, (2, 3))
    # Whatever greedy picks first, call that token "eos": both rows stop at once.
    first = generate(model, prompt, max_new_tokens=1, temperature=0)[:, -1]
    eos = int(first[0])
    out = generate(model, prompt, max_new_tokens=15, temperature=0, eos_id=eos)
    row0 = out[0, 3:].tolist()
    assert row0[0] == eos and all(t == eos for t in row0)


def test_generate_restores_training_mode():
    model = tiny_model().train()
    generate(model, torch.randint(0, 97, (1, 3)), max_new_tokens=2)
    assert model.training


def test_strip_after_eos():
    assert strip_after_eos([5, 6, 2, 7], eos_id=2) == [5, 6]
    assert strip_after_eos([5, 6], eos_id=2) == [5, 6]
