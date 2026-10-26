"""Define the decoder-only language model.

Owner: track B.

    tokens [B, S] -> embedding -> N x TransformerBlock -> RMSNorm -> lm_head
                  -> logits [B, S, vocab_size]

Contract with the training loop (track C, slm/training/pretrain.py):
model(x) returns logits only. The loss is computed outside, by
compute_loss, so SFT and DPO can mask tokens with ignore_index=-100.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from slm.model.attention import Attention
from slm.model.config import ModelConfig
from slm.model.layers import RMSNorm, SwiGLU

if TYPE_CHECKING:
    from slm.model.kv_cache import KVCache


class TransformerBlock(nn.Module):
    """Pre-norm decoder block:

        x = x + attention(norm(x))
        x = x + ffn(norm(x))

    Normalising the *input* of each sub-layer (pre-norm) rather than its
    output (post-norm, the original 2017 design) leaves the residual stream
    x untouched from embedding to output. Gradients flow straight down it,
    which is what lets deep stacks train without careful warmup tricks.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos, kv_cache, layer_idx)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """Decoder-only language model. Built as Transformer(cfg.model)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # One matrix, two jobs (decision S4): row t of tok_emb is token
            # t's input vector, and the same row scores token t at the output.
            # Both are [vocab_size, d_model], so this is a plain assignment.
            # parameters() yields the shared tensor once, so it is counted
            # and optimised once.
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights(cfg)

    def _init_weights(self, cfg: ModelConfig) -> None:
        """GPT-2 style init.

        Every Linear and the embedding start at N(0, 0.02). PyTorch's default
        N(0, 1) embedding would make tied output logits ~sqrt(d_model) = 22
        wide at step 0 instead of near-uniform.

        The two projections that write into the residual stream (o_proj,
        w_down) get std / sqrt(2 * n_layers): with 2 * n_layers additions,
        this keeps the residual stream's variance from growing with depth.
        """
        std = 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        resid_std = std / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=resid_std)

    def forward(self, idx: torch.Tensor, kv_cache: KVCache | None = None) -> torch.Tensor:
        """idx [B, S] int64 token ids -> logits [B, S, vocab_size].

        With a kv_cache, idx holds only the *new* tokens; their positions
        continue from what the cache already holds.
        """
        B, S = idx.shape
        start_pos = kv_cache.seq_len if kv_cache is not None else 0

        x = self.emb_dropout(self.tok_emb(idx))                     # [B, S, D]
        for i, layer in enumerate(self.layers):
            x = layer(x, start_pos, kv_cache, layer_idx=i)
        x = self.norm(x)
        logits = self.lm_head(x)                                    # [B, S, vocab]

        if kv_cache is not None:
            kv_cache.advance(S)  # every layer has now stored these S tokens
        return logits
