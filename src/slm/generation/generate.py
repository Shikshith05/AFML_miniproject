"""Generate text with sampling and key-value caching.

Owner: track D.

The model only predicts ONE next token: logits [B, S, vocab] in, where the
last position's 4096 scores rate every possible next token. Generation is
the loop that turns that into text: pick a token, append it, predict again.

    tok = Tokenizer.load()
    text = generate_text(model, tok, "Once upon a time", max_new_tokens=100)

Two phases, because of the KV cache (model/kv_cache.py):
  * prefill -- run the whole prompt once; every layer stores its K/V.
  * decode  -- feed only the newest token each step; earlier K/V are read
    from the cache instead of recomputed. Step cost stays flat instead of
    growing with the sequence length.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from slm.model.kv_cache import KVCache


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one next token per sequence from the last position's scores.

    logits: [B, vocab] scores for the next token (one row per sequence).
    Returns: [B, 1] chosen token ids.

    Order of operations: divide by temperature -> keep top-k -> softmax ->
    random draw. temperature=0 means greedy (always take the highest score).
    """
    # Greedy: the highest score wins, no randomness. Same input -> same output.
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    # Temperature: T < 1 widens the gaps between scores (safer, more focused),
    # T > 1 narrows them (more random). T = 1 leaves the model's own odds.
    logits = logits.float() / temperature

    # Top-k: find the k-th highest score in each row and push every score
    # below it to -inf. softmax(-inf) = 0, so those tokens can never be drawn.
    if top_k is not None and top_k < logits.size(-1):
        kth_best = torch.topk(logits, top_k, dim=-1).values[:, -1:]   # [B, 1]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))

    # Softmax turns scores into probabilities that sum to 1; multinomial draws
    # one index per row, weighted by those probabilities.
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 50,
    eos_id: int | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend each prompt by up to max_new_tokens tokens.

    prompt_ids: [B, P] token ids. All rows must be the same length -- the
        model has no padding mask, so for a batch use equal-length prompts
        (e.g. the first N tokens of each story).
    Returns: [B, P + n] -- prompt followed by the new tokens. A row that hits
        eos_id early keeps eos_id in the remaining slots.

    use_cache=False recomputes the whole sequence every step. Same output,
    much slower; it exists so the test can check the cache changes nothing.
    """
    was_training = model.training
    model.eval()                       # dropout off: generation must be deterministic given the seed

    B, P = prompt_ids.shape
    max_len = model.cfg.max_seq_len
    max_new_tokens = min(max_new_tokens, max_len - P)   # RoPE has no positions past max_seq_len
    if max_new_tokens <= 0:
        raise ValueError(f"prompt of {P} tokens leaves no room in max_seq_len {max_len}")

    device = prompt_ids.device
    seq = prompt_ids
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    cache = None
    if use_cache:
        # Cache dtype follows the weights (fp16 under autocast still writes
        # fp16 K/V; KVCache casts on write).
        dtype = next(model.parameters()).dtype
        cache = KVCache(model.cfg, batch_size=B, max_seq_len=P + max_new_tokens,
                        device=device, dtype=dtype)
        logits = model(prompt_ids, kv_cache=cache)            # prefill: whole prompt once

    for _ in range(max_new_tokens):
        if not use_cache:
            logits = model(seq)                               # recompute everything
        next_tok = sample_next_token(logits[:, -1, :], temperature, top_k, generator)  # [B, 1]

        if eos_id is not None:
            # Rows already done keep emitting eos so the tensor stays rectangular.
            next_tok = torch.where(finished.unsqueeze(1), torch.full_like(next_tok, eos_id), next_tok)
            finished |= next_tok.squeeze(1) == eos_id

        seq = torch.cat([seq, next_tok], dim=1)
        if eos_id is not None and finished.all():
            break
        if use_cache:
            logits = model(next_tok, kv_cache=cache)          # decode: one new token

    if was_training:
        model.train()
    return seq


def strip_after_eos(ids: list[int], eos_id: int) -> list[int]:
    """Cut a generated row at its first eos (eos excluded)."""
    return ids[: ids.index(eos_id)] if eos_id in ids else ids


@torch.no_grad()
def generate_text(
    model: nn.Module,
    tok,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 50,
    seed: int | None = None,
    add_bos: bool = True,
) -> str:
    """Convenience wrapper: text in, continuation text out (prompt excluded)."""
    device = next(model.parameters()).device
    ids = tok.encode(prompt, add_bos=add_bos)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    out = generate(model, x, max_new_tokens=max_new_tokens, temperature=temperature,
                   top_k=top_k, eos_id=tok.eos_id, generator=gen)
    new = strip_after_eos(out[0, len(ids):].tolist(), tok.eos_id)
    return tok.decode(new)
