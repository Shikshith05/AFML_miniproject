"""Alpaca instruction data for SFT.

Owner: track D (written under time pressure by track C).

The pretrained model continues text; it does not answer. SFT teaches the
answer format using the same next-token objective, on (instruction, response)
pairs, with the loss masked so only response tokens count.

Two things here are easy to get wrong and invisible when wrong:

  1. Loss masking. Prompt tokens get label -100 so cross_entropy ignores
     them. Without it the model also learns to generate instructions, and the
     loss curve looks completely normal either way. `assert_masking_sane`
     exists so a bug fails loudly instead.

  2. The prompt template. It must be byte-identical at SFT, DPO and eval.
     It lives in exactly one place -- config.PROMPT_TEMPLATE -- and is
     imported, never retyped.

Filtering: our model was pretrained only on TinyStories, so it has a
children's-story vocabulary and a 512-token context. Alpaca examples that
need world knowledge or long answers cannot be learned from here, and
training on them only adds noise, so they are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import torch

from slm.model.config import PROMPT_TEMPLATE, Config
from slm.tokenizer.tokenizer import Tokenizer

IGNORE_INDEX = -100     # torch's cross_entropy default ignore_index


@dataclass
class Example:
    """One tokenized instruction/response pair, already masked."""
    input_ids: list[int]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def n_supervised(self) -> int:
        return sum(1 for l in self.labels if l != IGNORE_INDEX)


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Render the prompt. The ONLY place the template is applied.

    Alpaca rows may carry an extra `input` field (e.g. a passage to
    summarise). We fold it into the instruction rather than inventing a
    second template, so SFT, DPO and eval all see one format.
    """
    instruction = instruction.strip()
    if input_text and input_text.strip():
        instruction = f"{instruction}\n\n{input_text.strip()}"
    return PROMPT_TEMPLATE.format(instruction=instruction)


def build_example(
    tok: Tokenizer,
    instruction: str,
    response: str,
    input_text: str = "",
    max_len: int = 512,
) -> Example | None:
    """Tokenize one pair and mask the prompt. None if it does not fit."""
    prompt_ids = tok.encode(format_prompt(instruction, input_text), add_bos=True)
    response_ids = tok.encode(response.strip(), add_eos=True)

    if len(prompt_ids) + len(response_ids) > max_len:
        return None
    if not response_ids:
        return None

    input_ids = prompt_ids + response_ids
    # -100 over the prompt: the model learns to PRODUCE responses, not to
    # produce instructions.
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    return Example(input_ids=input_ids, labels=labels)


def load_alpaca(cfg: Config, tok: Tokenizer, limit: int | None = None) -> list[Example]:
    """Load, filter and tokenize Alpaca.

    Filtering keeps short, simple examples -- the only ones a
    TinyStories-pretrained model has any chance of learning.
    """
    from datasets import load_dataset

    s = cfg.sft
    raw = load_dataset(s.dataset_name, split="train")
    if limit:
        raw = raw.select(range(min(limit, len(raw))))

    kept: list[Example] = []
    n_too_long = n_no_fit = 0

    for row in raw:
        instruction = row.get("instruction", "")
        response = row.get("output", "")
        input_text = row.get("input", "")
        if not instruction or not response:
            continue

        # cheap length filter before tokenizing the whole thing
        if len(tok.encode(instruction)) > s.max_instruction_tokens:
            n_too_long += 1
            continue
        if len(tok.encode(response)) > s.max_response_tokens:
            n_too_long += 1
            continue

        ex = build_example(
            tok, instruction, response, input_text, max_len=cfg.model.max_seq_len
        )
        if ex is None:
            n_no_fit += 1
            continue
        kept.append(ex)

    print(
        f"alpaca: {len(kept):,} kept of {len(raw):,}  "
        f"({n_too_long:,} too long, {n_no_fit:,} did not fit {cfg.model.max_seq_len})"
    )
    if kept:
        lens = np.array([len(e) for e in kept])
        sup = np.array([e.n_supervised for e in kept])
        print(
            f"  tokens/example: mean {lens.mean():.0f}  max {lens.max()}   "
            f"supervised: mean {sup.mean():.0f} ({100 * sup.sum() / lens.sum():.0f}% of tokens)"
        )
    return kept


def assert_masking_sane(examples: list[Example], tok: Tokenizer) -> None:
    """Fail loudly if masking is wrong.

    A masking bug trains happily and produces a normal-looking loss curve,
    so this is checked rather than eyeballed.
    """
    if not examples:
        raise ValueError("no examples after filtering -- loosen the filters")

    for ex in examples[:200]:
        if len(ex.input_ids) != len(ex.labels):
            raise AssertionError("input_ids and labels differ in length")
        if ex.n_supervised == 0:
            raise AssertionError("an example has no supervised tokens")
        if ex.n_supervised == len(ex):
            raise AssertionError("nothing masked -- the prompt is being trained on")
        # the mask must be a prefix: all -100s first, none after
        first_sup = next(i for i, l in enumerate(ex.labels) if l != IGNORE_INDEX)
        if any(l == IGNORE_INDEX for l in ex.labels[first_sup:]):
            raise AssertionError("mask is not a contiguous prefix")
        # supervised labels must equal the matching input ids
        for i in range(first_sup, len(ex)):
            if ex.labels[i] != ex.input_ids[i]:
                raise AssertionError("supervised label does not match input id")

    ex = examples[0]
    first_sup = next(i for i, l in enumerate(ex.labels) if l != IGNORE_INDEX)
    print("masking check passed. first example:")
    print(f"  prompt   ({first_sup} tokens, ignored): "
          f"{tok.decode(ex.input_ids[:first_sup])!r}")
    print(f"  response ({ex.n_supervised} tokens, learned): "
          f"{tok.decode(ex.input_ids[first_sup:])!r}")


def collate(batch: list[Example], pad_id: int, device: str):
    """Pad a batch to equal length. Padding is masked out of the loss too."""
    width = max(len(e) for e in batch)
    x = torch.full((len(batch), width), pad_id, dtype=torch.long)
    y = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
    for i, e in enumerate(batch):
        x[i, : len(e)] = torch.tensor(e.input_ids, dtype=torch.long)
        y[i, : len(e)] = torch.tensor(e.labels, dtype=torch.long)
    # next-token: predict position t+1 from position t
    return x[:, :-1].contiguous().to(device), y[:, 1:].contiguous().to(device)


def make_sft_batches(
    examples: list[Example],
    cfg: Config,
    tok: Tokenizer,
    *,
    shuffle: bool = True,
) -> Callable[[], Iterator]:
    """Return a function yielding one epoch of batches.

    Uses torch's global RNG for shuffling, so the training loop's RNG
    save/restore makes SFT resumable too.
    """
    bs = cfg.sft.micro_batch_size
    device = cfg.train.device

    def epoch() -> Iterator:
        order = (
            torch.randperm(len(examples)).tolist()
            if shuffle
            else list(range(len(examples)))
        )
        for i in range(0, len(order) - bs + 1, bs):
            yield collate([examples[j] for j in order[i : i + bs]], tok.pad_id, device)

    return epoch


def split_examples(
    examples: list[Example], val_fraction: float = 0.02, seed: int = 1337
) -> tuple[list[Example], list[Example]]:
    """Hold out a validation slice, deterministically."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(examples))
    n_val = max(1, int(len(examples) * val_fraction))
    val = [examples[i] for i in idx[:n_val]]
    train = [examples[i] for i in idx[n_val:]]
    return train, val
