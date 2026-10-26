"""Define key-value cache structures.

Owner: track B.

Generating token t+1 needs every earlier token's K and V in every layer.
Without a cache the model recomputes them all for each new token, so
generating n tokens costs O(n^2) token-forwards. With one, each step runs
the model on the new token only and reads earlier K/V from here.

K and V are stored at n_kv_heads -- before Attention repeats them up to
n_heads. That is where GQA's memory saving is realised: with n_kv_heads=2
instead of 8, this cache is 4x smaller.

    cache = KVCache(model.cfg, batch_size=1)
    logits = model(prompt, kv_cache=cache)      # prefill: whole prompt at once
    logits = model(next_token, kv_cache=cache)  # then one token per step
"""

from __future__ import annotations

import torch

from slm.model.config import ModelConfig


class KVCache:
    """Pre-allocated K/V buffers for every layer.

    Shape of each buffer: [n_layers, batch, n_kv_heads, max_seq_len, head_dim].
    Allocating once up front (rather than concatenating each step) avoids
    re-copying the whole cache on every token, and makes the memory cost
    exactly ModelConfig.kv_cache_bytes().

    For fp16 autocast generation pass dtype=torch.float16; the default fp32
    doubles the memory.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_seq_len = max_seq_len or cfg.max_seq_len
        if self.max_seq_len > cfg.max_seq_len:
            raise ValueError(
                f"cache length {self.max_seq_len} exceeds the model's "
                f"max_seq_len {cfg.max_seq_len} (RoPE has no angles past it)"
            )
        self.batch_size = batch_size

        shape = (cfg.n_layers, batch_size, cfg.n_kv_heads, self.max_seq_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

        # Tokens stored so far. Advanced by Transformer.forward once all
        # layers have written, so every layer in one forward sees the same
        # start position.
        self.seq_len = 0

    def update(
        self, layer_idx: int, start_pos: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new k, v [B, H_kv, S, hd] at positions start_pos .. start_pos+S,
        and return this layer's K/V for all positions so far, [B, H_kv, T, hd]."""
        B, _, S, _ = k.shape
        end = start_pos + S
        if B != self.batch_size:
            raise ValueError(f"batch {B} does not match cache batch {self.batch_size}")
        if end > self.max_seq_len:
            raise ValueError(f"cache full: need {end} positions, have {self.max_seq_len}")

        self.k[layer_idx, :, :, start_pos:end] = k
        self.v[layer_idx, :, :, start_pos:end] = v
        # Slices are views: no copy of the stored history.
        return self.k[layer_idx, :, :, :end], self.v[layer_idx, :, :, :end]

    def advance(self, n_tokens: int) -> None:
        self.seq_len += n_tokens

    def reset(self) -> None:
        """Start a new sequence. Old values are overwritten, not cleared."""
        self.seq_len = 0

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes
