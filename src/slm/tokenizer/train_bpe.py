"""Train a byte-pair encoding tokenizer.

Byte-level BPE (GPT-2 style): text is mapped to bytes before merging, so any
string is encodable and ``<unk>`` is never produced. ``<unk>`` still gets an
id so the special-token layout is fixed for every track:

    <pad>=0  <bos>=1  <eos>=2  <unk>=3

Owner: track A.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
DEFAULT_VOCAB_SIZE = 4096
INFO_FILENAME = "train_info.json"


def build_tokenizer() -> Tokenizer:
    """An untrained byte-level BPE tokenizer with no normalizer (lossless)."""
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_bpe(
    texts: Iterable[str] | Iterable[list[str]],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = 2,
    length: int | None = None,
    show_progress: bool = True,
) -> Tokenizer:
    """Train from an iterator of strings (or of batches of strings).

    ``length`` is only used for the progress bar.
    """
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        # all 256 byte symbols, so rare bytes are never missing from the base vocab
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    check_special_ids(tok)
    return tok


def check_special_ids(tok: Tokenizer) -> None:
    for i, s in enumerate(SPECIAL_TOKENS):
        got = tok.token_to_id(s)
        if got != i:
            raise AssertionError(f"special token {s} has id {got}, expected {i}")


def save_tokenizer(tok: Tokenizer, path: str | Path, info: dict[str, Any] | None = None) -> Path:
    """Write ``tokenizer.json`` and, if given, a ``train_info.json`` beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    if info is not None:
        (path.parent / INFO_FILENAME).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return path


def load_train_info(tokenizer_path: str | Path) -> dict[str, Any] | None:
    p = Path(tokenizer_path).parent / INFO_FILENAME
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def train_on_tinystories(
    ds,
    val_indices,
    out_path: str | Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    limit: int | None = None,
    max_stories: int | None = None,
    info: dict[str, Any] | None = None,
) -> Tokenizer:
    """Train on the (cleaned) TRAIN side of the split only, then save with a train_info.json.

    ``limit`` truncates the dataset (smoke runs); ``max_stories`` caps how many
    train stories the tokenizer sees.
    """
    from slm.data.tinystories import cleaning_version, iter_stories, split_hash

    n_lim = len(ds) if limit is None else min(limit, len(ds))
    n_avail = n_lim - int(np.searchsorted(val_indices, n_lim))  # val_indices is sorted
    n_used = n_avail if max_stories is None else min(max_stories, n_avail)
    counter = {"n": 0}

    def texts() -> Iterator[list[str]]:
        for batch in take(iter_stories(ds, val_indices, "train", limit=limit), n_used):
            counter["n"] += len(batch)
            yield batch

    print(f"training byte-level BPE (vocab {vocab_size}) on {n_used:,} train stories")
    t0 = time.perf_counter()
    tok = train_bpe(texts(), vocab_size=vocab_size, length=n_used)
    dt = time.perf_counter() - t0
    print(f"tokenizer trained in {dt:.1f}s; vocab size {tok.get_vocab_size()}")

    full_info = {
        "type": "byte-level BPE",
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": {s: i for i, s in enumerate(SPECIAL_TOKENS)},
        "n_stories_trained": counter["n"],
        "limit": limit,
        "max_stories": max_stories,
        "split_hash": split_hash(val_indices),
        "text_cleaning": cleaning_version(),
        "train_seconds": round(dt, 1),
        **(info or {}),
    }
    save_tokenizer(tok, out_path, full_info)
    print(f"saved {out_path}")
    return tok


def take(batches: Iterable[list[str]], max_items: int | None) -> Iterator[list[str]]:
    """Truncate an iterator of batches to at most ``max_items`` strings in total."""
    seen = 0
    for batch in batches:
        if max_items is not None and seen + len(batch) > max_items:
            batch = batch[: max_items - seen]
        if batch:
            yield batch
        seen += len(batch)
        if max_items is not None and seen >= max_items:
            return
