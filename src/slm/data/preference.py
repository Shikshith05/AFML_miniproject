"""Build chosen and rejected preference pairs for DPO.

Owner: track D.

Decision S3: chosen = the real ending of a held-out TinyStories story,
rejected = what a weaker checkpoint of our own model writes from the same
opening. Both are put in the SFT instruction format, so DPO trains the model
in the same format it was fine-tuned on (this resolves the SFT/DPO mismatch):

    ### Instruction:
    Continue the story.

    <first ~48 tokens of the story>

    ### Response:
    <chosen: the real rest of the story>   or   <rejected: weak model's text>

Held-out source: the HF "validation" split. Pretraining carved its val set out
of the HF *train* split (data/tinystories.py), so no story here was seen by
the model or the tokenizer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from slm.data.alpaca import IGNORE_INDEX, Example, build_example, collate

STORY_INSTRUCTION = "Continue the story."


@dataclass
class PreferencePair:
    """One pair, stored as text so it can be read and inspected."""
    instruction: str
    input: str          # the story opening
    chosen: str         # real continuation
    rejected: str       # weak model's continuation


@dataclass
class TokenizedPair:
    chosen: Example     # prompt masked with -100, response supervised
    rejected: Example


# -- story openings -----------------------------------------------------------

def load_story_prefixes(
    tok,
    n: int,
    prefix_tokens: int = 48,
    max_response_tokens: int = 192,
    seed: int = 1337,
    split: str = "validation",
) -> list[tuple[list[int], list[int]]]:
    """Return n (prefix_ids, rest_ids) from held-out stories.

    Only stories that END within max_response_tokens after the prefix are
    kept, so every chosen response is a complete story and the rejected
    generation gets the same token budget to finish its own.
    """
    from slm.data.tinystories import clean_text, load_tinystories

    ds = load_tinystories(split)
    order = np.random.default_rng(seed).permutation(len(ds))

    out: list[tuple[list[int], list[int]]] = []
    for i in order:
        text = clean_text(ds[int(i)]["text"]).strip()
        if not text:
            continue
        ids = tok.encode(text)
        rest = ids[prefix_tokens:]
        if len(ids) <= prefix_tokens + 16 or len(rest) > max_response_tokens:
            continue    # too short to continue, or too long to finish
        out.append((ids[:prefix_tokens], rest))
        if len(out) == n:
            break
    if len(out) < n:
        print(f"warning: only {len(out)} stories met the length filters (asked for {n})")
    return out


# -- rejected responses -------------------------------------------------------

@torch.no_grad()
def generate_rejected(
    model,
    tok,
    prefixes: list[list[int]],
    *,
    max_new_tokens: int = 192,
    temperature: float = 1.0,
    top_k: int | None = 50,
    batch_size: int = 64,
    seed: int = 1337,
    device: str = "cuda",
) -> list[str]:
    """Continue each opening with the (weak) model, as raw story text.

    The weak model is a base checkpoint, so it is prompted the way it was
    trained -- <bos> + story opening -- not with the instruction template.
    All prefixes have the same token length, so they batch without padding.
    """
    from tqdm.auto import tqdm

    from slm.generation.generate import generate, strip_after_eos

    model.eval().to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    texts: list[str] = []
    for i in tqdm(range(0, len(prefixes), batch_size), desc="rejected"):
        chunk = prefixes[i : i + batch_size]
        x = torch.tensor([[tok.bos_id] + p for p in chunk], dtype=torch.long, device=device)
        out = generate(model, x, max_new_tokens=max_new_tokens, temperature=temperature,
                       top_k=top_k, eos_id=tok.eos_id, generator=gen)
        for row in out[:, x.size(1):].tolist():
            texts.append(tok.decode(strip_after_eos(row, tok.eos_id)).strip())
    return texts


def build_pairs(tok, prefixes_and_rests, rejected_texts) -> list[PreferencePair]:
    pairs = []
    for (prefix, rest), rejected in zip(prefixes_and_rests, rejected_texts):
        chosen = tok.decode(rest).strip()
        if not rejected or rejected == chosen:
            continue    # nothing to prefer between
        pairs.append(PreferencePair(STORY_INSTRUCTION, tok.decode(prefix).strip(),
                                    chosen, rejected))
    return pairs


# -- save / load --------------------------------------------------------------

def save_pairs(pairs: list[PreferencePair], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def load_pairs(path: str | Path) -> list[PreferencePair]:
    with Path(path).open(encoding="utf-8") as f:
        return [PreferencePair(**json.loads(line)) for line in f if line.strip()]


def split_pairs(pairs: list, n_test: int = 200, seed: int = 1337) -> tuple[list, list]:
    """Deterministic train/test split. Test pairs are never trained on."""
    idx = np.random.default_rng(seed).permutation(len(pairs))
    n_test = min(n_test, len(pairs) // 5)
    test = [pairs[i] for i in idx[:n_test]]
    train = [pairs[i] for i in idx[n_test:]]
    return train, test


# -- tokenize and batch -------------------------------------------------------

def tokenize_pairs(pairs: list[PreferencePair], tok, max_len: int = 512) -> list[TokenizedPair]:
    """Same build_example as SFT: identical template, identical masking."""
    out = []
    for p in pairs:
        c = build_example(tok, p.instruction, p.chosen, p.input, max_len=max_len)
        r = build_example(tok, p.instruction, p.rejected, p.input, max_len=max_len)
        if c is not None and r is not None:
            out.append(TokenizedPair(c, r))
    return out


def collate_pairs(batch: list[TokenizedPair], pad_id: int, device: str):
    """One padded batch of 2B rows: chosen[0..B-1] then rejected[0..B-1].

    Chosen and rejected go through the model in a single forward pass;
    split the result with .chunk(2).
    """
    rows = [p.chosen for p in batch] + [p.rejected for p in batch]
    return collate(rows, pad_id, device)


def pair_length_stats(pairs: list[TokenizedPair]) -> dict:
    """Mean response lengths. A big gap means DPO could learn 'prefer
    longer/shorter' instead of 'prefer better' -- report it."""
    c = np.array([p.chosen.n_supervised for p in pairs])
    r = np.array([p.rejected.n_supervised for p in pairs])
    return {"chosen_mean_tokens": float(c.mean()), "rejected_mean_tokens": float(r.mean()),
            "n_pairs": len(pairs)}


__all__ = [
    "IGNORE_INDEX", "STORY_INSTRUCTION", "PreferencePair", "TokenizedPair",
    "load_story_prefixes", "generate_rejected", "build_pairs", "save_pairs",
    "load_pairs", "split_pairs", "tokenize_pairs", "collate_pairs", "pair_length_stats",
]
