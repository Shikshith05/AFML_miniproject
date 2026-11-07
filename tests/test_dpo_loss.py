"""Test direct preference optimization loss.

Owner: track D.
"""

import math

import pytest
import torch

from slm.data.alpaca import IGNORE_INDEX
from slm.model.config import ModelConfig
from slm.model.transformer import Transformer
from slm.training.dpo import dpo_loss, pair_logprobs, sequence_logprobs


def test_loss_is_log2_when_policy_equals_reference():
    """At the start of DPO the policy IS the reference: margin 0, loss log 2."""
    lp = torch.tensor([-10.0, -20.0])
    loss, cw, rw = dpo_loss(lp, lp - 3, lp, lp - 3, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2))
    assert torch.all(cw == 0) and torch.all(rw == 0)


def test_hand_computed_value():
    # chosen log-ratio = -5 - (-6) = +1, rejected log-ratio = -9 - (-8) = -1
    # margin = 0.5 * (1 - (-1)) = 1.0 ; loss = -log sigmoid(1) = log(1 + e^-1)
    loss, cw, rw = dpo_loss(torch.tensor([-5.0]), torch.tensor([-9.0]),
                            torch.tensor([-6.0]), torch.tensor([-8.0]), beta=0.5)
    assert loss.item() == pytest.approx(math.log(1 + math.exp(-1)), rel=1e-6)
    assert cw.item() == pytest.approx(0.5) and rw.item() == pytest.approx(-0.5)


def test_loss_falls_as_chosen_gains_over_rejected():
    ref = torch.tensor([-10.0])
    losses = [dpo_loss(ref + d, ref - d, ref, ref, beta=0.1)[0].item() for d in [0, 1, 5, 20]]
    assert losses == sorted(losses, reverse=True)
    assert losses[-1] < 0.1


def test_only_the_difference_matters():
    """Shifting both responses' log-probs by the same amount changes nothing."""
    a = dpo_loss(torch.tensor([-5.0]), torch.tensor([-7.0]), torch.tensor([-6.0]), torch.tensor([-6.0]))[0]
    b = dpo_loss(torch.tensor([-1.0]), torch.tensor([-3.0]), torch.tensor([-2.0]), torch.tensor([-2.0]))[0]
    assert a.item() == pytest.approx(b.item())


def test_gradient_pushes_chosen_up_and_rejected_down():
    pc = torch.tensor([-5.0], requires_grad=True)
    pr = torch.tensor([-5.0], requires_grad=True)
    loss, _, _ = dpo_loss(pc, pr, torch.tensor([-5.0]), torch.tensor([-5.0]), beta=0.1)
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()   # gradient descent raises pc, lowers pr


def test_sequence_logprobs_ignores_masked_positions():
    logits = torch.zeros(1, 4, 10)                         # uniform: every token log(1/10)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 7]])
    lp = sequence_logprobs(logits, labels)
    assert lp.item() == pytest.approx(2 * math.log(0.1))


def test_sequence_logprobs_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.randint(0, 11, (2, 6))
    labels[:, :2] = IGNORE_INDEX
    per_row = sequence_logprobs(logits, labels)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1),
                                           ignore_index=IGNORE_INDEX, reduction="sum")
    assert (-per_row.sum()).item() == pytest.approx(ce.item(), rel=1e-5)


def test_one_dpo_step_increases_reward_margin():
    """End to end on a tiny model: a few optimizer steps make chosen beat rejected."""
    torch.manual_seed(0)
    model = Transformer(ModelConfig(d_model=32, n_layers=1, n_heads=2, n_kv_heads=1,
                                    max_seq_len=32, vocab_size=50, dropout=0.0))
    x = torch.randint(0, 50, (2, 8))                      # row 0 = chosen, row 1 = rejected
    y = torch.randint(0, 50, (2, 8))
    y[:, :3] = IGNORE_INDEX
    with torch.no_grad():
        rc, rr = pair_logprobs(model, x, y)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(10):
        pc, pr = pair_logprobs(model, x, y)
        loss, cw, rw = dpo_loss(pc, pr, rc, rr, beta=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (cw - rw).item() > 0
    assert loss.item() < math.log(2)
