"""Reference log-prob cache: never reused for other pairs or another model.

Owner: track D. CPU only, tiny random model, no data files.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from slm.data.alpaca import IGNORE_INDEX, Example
from slm.data.preference import TokenizedPair
from slm.model.config import ModelConfig
from slm.model.transformer import Transformer
from slm.training.dpo import load_or_compute_reference, model_fingerprint, pairs_fingerprint

REPO = Path(__file__).resolve().parents[1]
KW = dict(pad_id=0, batch_size=4, precision="fp32", device="cpu")


def tiny_model(seed: int) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(ModelConfig(d_model=32, n_layers=1, n_heads=4, n_kv_heads=2,
                                   max_seq_len=32, vocab_size=50))


def make_pairs(offset: int = 0, n: int = 6) -> list[TokenizedPair]:
    def ex(ids, n_prompt):
        return Example(input_ids=ids, labels=[IGNORE_INDEX] * n_prompt + ids[n_prompt:])
    g = np.random.default_rng(offset)
    return [TokenizedPair(chosen=ex(list(g.integers(1, 50, 10)), 4),
                          rejected=ex(list(g.integers(1, 50, 10)), 4)) for _ in range(n)]


def test_same_pairs_and_model_load_from_cache(tmp_path, capsys):
    path, model, pairs = tmp_path / "ref.npz", tiny_model(0), make_pairs()
    c1, r1 = load_or_compute_reference(path, model, pairs, **KW)
    c2, r2 = load_or_compute_reference(path, model, pairs, **KW)
    assert "loaded" in capsys.readouterr().out
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(r1, r2)


def test_different_reference_model_is_recomputed(tmp_path, capsys):
    """The oasst1 case: same pairs, different SFT checkpoint."""
    path, pairs = tmp_path / "ref.npz", make_pairs()
    alpaca_c, _ = load_or_compute_reference(path, tiny_model(0), pairs, **KW)
    capsys.readouterr()
    oasst_c, _ = load_or_compute_reference(path, tiny_model(1), pairs, **KW)
    assert "different reference model" in capsys.readouterr().out
    assert not np.allclose(alpaca_c, oasst_c)
    assert str(np.load(path)["model_fingerprint"]) == model_fingerprint(tiny_model(1))


def test_different_pairs_are_recomputed(tmp_path, capsys):
    path, model = tmp_path / "ref.npz", tiny_model(0)
    load_or_compute_reference(path, model, make_pairs(0), **KW)
    capsys.readouterr()
    load_or_compute_reference(path, model, make_pairs(1), **KW)
    assert "different pairs" in capsys.readouterr().out


def test_cache_without_model_fingerprint_is_recomputed(tmp_path, capsys):
    """Caches written before the model fingerprint existed are not trusted."""
    path, model, pairs = tmp_path / "ref.npz", tiny_model(0), make_pairs()
    np.savez(path, chosen=np.zeros(6), rejected=np.zeros(6), fingerprint=pairs_fingerprint(pairs))
    c, _ = load_or_compute_reference(path, model, pairs, **KW)
    assert "different reference model" in capsys.readouterr().out
    assert not np.allclose(c, 0)


def test_model_fingerprint_tracks_weights():
    a, b = tiny_model(0), tiny_model(0)
    assert model_fingerprint(a) == model_fingerprint(b)
    with torch.no_grad():
        b.tok_emb.weight[0, 0] += 1e-3
    assert model_fingerprint(a) != model_fingerprint(b)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dpo_cache_path_is_separate_per_run():
    stem = load_script("dpo").ref_cache_stem
    default = "data/ref_logprobs.npz"
    assert stem(None, "gqa_dpo", "gqa", default) == default            # Alpaca run unchanged
    oasst = Path(stem(None, "gqa_dpo_oasst1", "gqa", default))
    assert oasst == Path("data/ref_logprobs_gqa_dpo_oasst1.npz")
    assert stem("data/mine", "gqa_dpo_oasst1", "gqa", default) == "data/mine"
