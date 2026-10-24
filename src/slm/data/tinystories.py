"""Provide the pretraining TinyStories dataset.

Two halves:

- Corpus side (used by scripts/prepare_data.py and scripts/train_tokenizer.py):
  the deterministic train/val story split and a story iterator.
- Training side (used by track C): ``make_get_batch`` returns
  ``get_batch(split) -> (x, y)``, both int64 [B, S] on the target device,
  with y = x shifted left by one token.

Token files are flat np.uint16 arrays (``data/train.bin``, ``data/val.bin``):
every story is ``<bos> tokens <eos>`` and stories are concatenated.

Owner: track A.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

DATASET_NAME = "roneneldan/TinyStories"
DEFAULT_DATA_DIR = "data"
DEFAULT_SEED = 1337
DEFAULT_VAL_FRACTION = 0.01
DEFAULT_SEQ_LEN = 512
TOKEN_DTYPE = np.uint16

BatchFn = Callable[[str], "tuple[torch.Tensor, torch.Tensor]"]


# --------------------------------------------------------------------------
# Corpus split
# --------------------------------------------------------------------------


def split_val_indices(
    n_total: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Sorted indices (into the full HF train split) of the held-out val stories.

    Always computed over the *full* dataset size, so a ``--limit`` smoke run
    uses a subset of the real val set rather than a different one, and a
    tokenizer trained on the full train portion never saw any smoke-run val
    story either.
    """
    n_val = max(1, round(n_total * val_fraction))
    rng = np.random.default_rng(seed)
    return np.sort(rng.permutation(n_total)[:n_val]).astype(np.int64)


def split_hash(val_indices: np.ndarray) -> str:
    """Short fingerprint of a split, recorded in meta.json and the tokenizer info."""
    return hashlib.sha256(np.ascontiguousarray(val_indices, dtype=np.int64).tobytes()).hexdigest()[:16]


def load_tinystories(split: str = "train"):
    """The HF dataset (imported lazily so get_batch never pays for ``datasets``)."""
    from datasets import load_dataset

    return load_dataset(DATASET_NAME, split=split)


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------

# TinyStories v1 has UTF-8 punctuation that was decoded as cp1252 ("â€™" for
# "’"). The closing quote ” is E2 80 9D, and 0x9D has no cp1252 character,
# so that byte was lost, leaving a bare "â€". ftfy can't recover a missing
# byte (before a space it guesses E2 80 A0 and gives "†"). A "â€" not followed
# by a cp1252 continuation character can only be a broken ”, so restore it
# first and let ftfy handle the rest.
_CP1252_CONTINUATION = "".join(
    bytes([b]).decode("cp1252") for b in range(0x80, 0xC0) if b not in (0x81, 0x8D, 0x8F, 0x90, 0x9D)
)
_BROKEN_RDQUO = re.compile("â€(?![" + re.escape(_CP1252_CONTINUATION) + "])")


def cleaning_version() -> str:
    """Identifies the cleaning applied, recorded beside the tokenizer and data."""
    import ftfy

    return f"ftfy-{ftfy.__version__}+rdquo-v1"


def clean_text(text: str) -> str:
    """Fix mojibake (and ftfy's defaults: curly quotes -> ASCII, NFC, line breaks)."""
    import ftfy

    return ftfy.fix_text(_BROKEN_RDQUO.sub("”", text))


def clean_batch(texts: list[str]) -> tuple[list[str], int]:
    """Cleaned texts and how many of them changed."""
    out = [clean_text(t) for t in texts]
    return out, sum(a != b for a, b in zip(texts, out))


def iter_chunks(
    ds,
    limit: int | None = None,
    batch_size: int = 10_000,
    num_proc: int | None = None,
) -> Iterator[tuple[int, list[str], int]]:
    """Yield ``(start, cleaned_texts, n_changed)`` over the dataset in order.

    This is the one text-loading path: tokenizer training and encoding both
    go through it, so they see identical cleaned text. Cleaning runs in a
    process pool, a bounded number of chunks at a time, so memory stays flat.
    Callers must be under ``if __name__ == "__main__":`` on Windows.
    """
    n = len(ds) if limit is None else min(limit, len(ds))
    starts = list(range(0, n, batch_size))
    num_proc = num_proc or min(8, os.cpu_count() or 1)

    def raw(start: int) -> list[str]:
        return ds[start : min(start + batch_size, n)]["text"]

    if num_proc <= 1:
        for s in starts:
            yield (s, *clean_batch(raw(s)))
        return

    from multiprocessing import Pool

    wave = 2 * num_proc
    with Pool(num_proc) as pool:
        for i in range(0, len(starts), wave):
            group = starts[i : i + wave]
            for s, (texts, n_changed) in zip(group, pool.map(clean_batch, [raw(s) for s in group])):
                yield s, texts, n_changed


def iter_stories(
    ds,
    val_indices: np.ndarray,
    which: str,
    limit: int | None = None,
    batch_size: int = 10_000,
    num_proc: int | None = None,
) -> Iterator[list[str]]:
    """Yield batches of cleaned story texts from the train or val side of the split.

    Reads the dataset sequentially (fast) and routes by a boolean mask rather
    than ``ds.select``, which would turn every read into a random access.
    ``limit`` restricts to the first ``limit`` stories of the dataset.
    """
    if which not in ("train", "val"):
        raise ValueError(f"which must be 'train' or 'val', got {which!r}")
    is_val = np.zeros(len(ds), dtype=bool)
    is_val[val_indices] = True
    want_val = which == "val"

    for start, texts, _ in iter_chunks(ds, limit, batch_size, num_proc):
        mask = is_val[start : start + len(texts)]
        yield [t for t, v in zip(texts, mask) if v == want_val]


# --------------------------------------------------------------------------
# Batch sampling
# --------------------------------------------------------------------------


def _open(path: Path) -> np.memmap:
    # Re-opened on every call: a memmap is cheap to create, and holding one
    # open for the whole run makes its pages count as process memory.
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def _make(
    paths: dict[str, Path],
    batch_size: int,
    seq_len: int,
    device: str | torch.device,
    max_train_tokens: int | None = None,
) -> BatchFn:
    for split, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{split} tokens not found at {p} -- run scripts/prepare_data.py")

    # Number of usable tokens per split, checked once up front.
    lengths: dict[str, int] = {}
    for split, p in paths.items():
        n = os.path.getsize(p) // np.dtype(TOKEN_DTYPE).itemsize
        if split == "train" and max_train_tokens is not None:
            n = min(n, int(max_train_tokens))
        if n < seq_len + 1:
            raise ValueError(f"{split} has {n} tokens, need at least seq_len+1={seq_len + 1}")
        lengths[split] = n

    device = torch.device(device)
    pin = device.type == "cuda"

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        if split not in paths:
            raise KeyError(f"unknown split {split!r}; expected one of {sorted(paths)}")
        data = _open(paths[split])
        # torch's global CPU RNG, so the training loop's RNG save/restore makes
        # the batch stream resumable. Largest valid start leaves room for y.
        ix = torch.randint(lengths[split] - seq_len, (batch_size,)).tolist()
        buf = np.stack([data[i : i + seq_len + 1] for i in ix]).astype(np.int64)
        xy = torch.from_numpy(buf)
        x, y = xy[:, :-1], xy[:, 1:]
        if pin:
            x = x.contiguous().pin_memory().to(device, non_blocking=True)
            y = y.contiguous().pin_memory().to(device, non_blocking=True)
        else:
            x = x.contiguous().to(device)
            y = y.contiguous().to(device)
        return x, y

    return get_batch


def make_get_batch(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    batch_size: int = 16,
    seq_len: int = DEFAULT_SEQ_LEN,
    device: str | torch.device = "cpu",
    *,
    max_train_tokens: int | None = None,
) -> BatchFn:
    """Return ``get_batch(split)`` over ``data_dir/{train,val}.bin``.

    Windows are sampled uniformly at random (with replacement) and may span
    story boundaries; the ``<eos><bos>`` separators tell the model where a
    story ends. ``max_train_tokens`` caps sampling to a prefix of train.bin.
    """
    data_dir = Path(data_dir)
    paths = {"train": data_dir / "train.bin", "val": data_dir / "val.bin"}
    return _make(paths, batch_size, seq_len, device, max_train_tokens)


def make_batch_provider(cfg: Any) -> BatchFn:
    """Adapter for track C's ``Config`` (read by attribute; no import of it).

    Uses ``cfg.data.train_bin``, ``cfg.data.val_bin``, ``cfg.data.max_train_tokens``,
    ``cfg.train.micro_batch_size``, ``cfg.model.max_seq_len`` and ``cfg.train.device``.
    """
    paths = {"train": Path(cfg.data.train_bin), "val": Path(cfg.data.val_bin)}
    return _make(
        paths,
        batch_size=cfg.train.micro_batch_size,
        seq_len=cfg.model.max_seq_len,
        device=cfg.train.device,
        max_train_tokens=getattr(cfg.data, "max_train_tokens", None),
    )
