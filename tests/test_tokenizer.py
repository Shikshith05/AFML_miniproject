"""Test tokenizer behavior.

Needs no dataset: a tiny tokenizer is trained in a tmp dir, and get_batch is
tested on a small fake .bin. If the committed tokenizer exists it is checked
too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm.data.tinystories import (  # noqa: E402
    clean_batch,
    clean_text,
    iter_chunks,
    make_batch_provider,
    make_get_batch,
    split_val_indices,
)
from slm.tokenizer.tokenizer import Tokenizer  # noqa: E402
from slm.tokenizer.train_bpe import SPECIAL_TOKENS, save_tokenizer, train_bpe  # noqa: E402

COMMITTED = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"

CORPUS = [
    "Once upon a time, there was a little girl named Lily. She loved to play outside.",
    "Tom had a big red ball. He said, \"Can we play?\" His mom smiled and said yes!",
    "The cat sat on the mat.\nThe dog ran in the park.\n\nThey were happy friends.",
    "One day, a bird flew to the tree. It sang a song: la, la, la... The end.",
] * 50

ROUND_TRIP = [
    "Once upon a time, there was a little girl named Lily.",
    "  leading spaces, trailing spaces   ",
    "Punctuation!? \"quotes\" 'single' (parens) [brackets] {braces} -- dashes; colons: ...",
    "Line one\nLine two\n\n\tTabbed line\r\nWindows newline",
    "Unicode: café, naïve, Straße, 日本語, emoji 🐱🎈, curly ‘quotes’ — em dash",
    "",
    "<bos> literal special-looking text <eos>",
]


@pytest.fixture(scope="module")
def tiny_tok(tmp_path_factory) -> Tokenizer:
    hf = train_bpe(CORPUS, vocab_size=300, show_progress=False)
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    save_tokenizer(hf, path)
    return Tokenizer.load(path.parent)  # directory form, as track C's config passes it


def _tokenizers():
    yield "tiny"
    if COMMITTED.exists():
        yield "committed"


@pytest.fixture(params=list(_tokenizers()))
def tok(request, tiny_tok) -> Tokenizer:
    return tiny_tok if request.param == "tiny" else Tokenizer.load(COMMITTED)


# ---- tokenizer ---------------------------------------------------------------


def test_special_token_ids(tok):
    assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id) == (0, 1, 2, 3)
    for i, s in enumerate(SPECIAL_TOKENS):
        assert tok.hf.token_to_id(s) == i


def test_committed_vocab_size():
    if not COMMITTED.exists():
        pytest.skip("no committed tokenizer yet")
    assert Tokenizer.load(COMMITTED).vocab_size == 4096


@pytest.mark.parametrize("text", ROUND_TRIP)
def test_round_trip(tok, text):
    ids = tok.encode(text)
    assert tok.unk_id not in ids
    assert tok.decode(ids) == text


def test_bos_eos(tok):
    plain = tok.encode("hello world")
    wrapped = tok.encode("hello world", add_bos=True, add_eos=True)
    assert wrapped == [tok.bos_id] + plain + [tok.eos_id]
    assert tok.decode(wrapped) == "hello world"


def test_encode_batch_matches_encode(tok):
    texts = ROUND_TRIP[:4]
    assert tok.encode_batch(texts, add_bos=True) == [tok.encode(t, add_bos=True) for t in texts]


# ---- cleaning ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, clean",
    [
        ("Tom said, â€œHi!â€ Then he left.", 'Tom said, "Hi!" Then he left.'),
        ("â€œWait!â€\nShe stopped.", '"Wait!"\nShe stopped.'),
        ("I donâ€™t knowâ€.", "I don't know\"."),
        ("CafÃ© and a naÃ¯ve cat", "Café and a naïve cat"),
        ("Lily’s “big” day", "Lily's \"big\" day"),
        ("Plain ASCII story.\n\nThe end.", "Plain ASCII story.\n\nThe end."),
    ],
)
def test_clean_text(raw, clean):
    assert clean_text(raw) == clean


def test_clean_batch_counts_changes():
    out, n = clean_batch(["fine", "donâ€™t", "also fine"])
    assert out == ["fine", "don't", "also fine"] and n == 1


def test_iter_chunks_parallel_matches_serial():
    ds = _FakeDataset(["ok", "donâ€™t", "x â€ y"] * 7)
    serial = list(iter_chunks(ds, batch_size=4, num_proc=1))
    parallel = list(iter_chunks(ds, batch_size=4, num_proc=2))
    assert serial == parallel
    assert sum(n for _, _, n in serial) == 14


class _FakeDataset:
    """Just enough of a HF Dataset: len() and ds[a:b]["text"]."""

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, sl):
        return {"text": self.texts[sl]}


# ---- split -------------------------------------------------------------------


def test_split_is_deterministic():
    a = split_val_indices(100_000, 0.01, 1337)
    b = split_val_indices(100_000, 0.01, 1337)
    assert len(a) == 1000 and np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)
    assert not np.array_equal(a, split_val_indices(100_000, 0.01, 1338))


# ---- get_batch -----------------------------------------------------------------


@pytest.fixture
def fake_data(tmp_path) -> Path:
    # Consecutive values make "y is x shifted by one" checkable from values alone.
    np.arange(5000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    (np.arange(300, dtype=np.uint16) + 10_000).tofile(tmp_path / "val.bin")
    return tmp_path


@pytest.mark.parametrize("split", ["train", "val"])
def test_get_batch_shapes_and_shift(fake_data, split):
    get_batch = make_get_batch(fake_data, batch_size=4, seq_len=32, device="cpu")
    x, y = get_batch(split)
    assert x.shape == y.shape == (4, 32)
    assert x.dtype == y.dtype == torch.int64
    assert x.device.type == "cpu"
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y, x + 1)  # the fake data is consecutive integers
    if split == "val":
        assert int(x.min()) >= 10_000


def test_get_batch_covers_last_window(fake_data):
    # seq_len+1 == file length: the only window is the whole file.
    np.arange(33, dtype=np.uint16).tofile(fake_data / "val.bin")
    x, y = make_get_batch(fake_data, batch_size=2, seq_len=32)("val")
    assert int(x[0, 0]) == 0 and int(y[0, -1]) == 32


def test_get_batch_too_short(fake_data):
    with pytest.raises(ValueError):
        make_get_batch(fake_data, batch_size=2, seq_len=300)


def test_same_seed_same_batches(fake_data):
    get_batch = make_get_batch(fake_data, batch_size=8, seq_len=16)
    torch.manual_seed(1337)
    a = [get_batch("train") for _ in range(3)]
    torch.manual_seed(1337)
    b = [get_batch("train") for _ in range(3)]
    for (xa, ya), (xb, yb) in zip(a, b):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    torch.manual_seed(1338)
    assert not torch.equal(get_batch("train")[0], a[0][0])


def test_rng_state_restore_resumes_stream(fake_data):
    get_batch = make_get_batch(fake_data, batch_size=8, seq_len=16)
    torch.manual_seed(0)
    get_batch("train")
    state = torch.get_rng_state()
    expected = get_batch("train")[0]
    get_batch("train")
    torch.set_rng_state(state)
    assert torch.equal(get_batch("train")[0], expected)


def test_make_batch_provider_adapter(fake_data):
    from types import SimpleNamespace as NS

    cfg = NS(
        data=NS(train_bin=str(fake_data / "train.bin"), val_bin=str(fake_data / "val.bin"), max_train_tokens=100),
        train=NS(micro_batch_size=3, device="cpu"),
        model=NS(max_seq_len=20),
    )
    x, y = make_batch_provider(cfg)("train")
    assert x.shape == (3, 20)
    assert int(y.max()) < 100  # sampling restricted to the max_train_tokens prefix


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_get_batch_cuda(fake_data):
    x, y = make_get_batch(fake_data, batch_size=2, seq_len=16, device="cuda")("train")
    assert x.is_cuda and y.is_cuda and x.dtype == torch.int64
