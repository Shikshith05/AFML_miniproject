# Person B notes — Phase 2: model, attention, tests

Merged into `main`. Final checks 2026-11-09 on torch 2.14.1 (CPU): **33/33
track B tests passing** in about 20 s (`python -m pytest -q
tests/test_attention.py tests/test_model.py`). Whole repo: 70 passed, 1 skipped
(the CUDA-only tokenizer test).

Final model: both variants have **27,795,968 parameters**. MHA is 8 KV heads,
`ffn_hidden` 1408. GQA is 2 KV heads, `ffn_hidden` 1664 (decision S7).

| Commit | What |
|---|---|
| `5a97758` | config.py copied verbatim from `track/training` |
| `ad800b5` | layers.py — RMSNorm, SwiGLU, RoPE |
| `b32db66` | attention.py + tests/test_attention.py |
| `048c1a2` | transformer.py + first tests in tests/test_model.py |
| `1d57293` | kv_cache.py + cache tests |
| `076dbbc` | utils/params.py + parameter-count tests |
| `d104c8f` | overfit test |
| `5951d91`, `6459608` | attention.py: `enable_gqa` path and probe (track C, 6 Nov) |
| `3557cd5` | attention.py: `repeat_interleave` default again, `enable_gqa` opt-in (12 Nov, decision S14) |

How the pieces fit together:

```
token ids [B, S]
   │  tok_emb (4096 × 512)                        transformer.py
   ▼
x [B, S, 512]  ── the "residual stream" ──────────────────────────────┐
   │  for each of 8 TransformerBlocks:                                │
   │     x = x + Attention( RMSNorm(x) )   attention.py, layers.py    │
   │     x = x + SwiGLU(    RMSNorm(x) )   layers.py                  │
   ▼                                                                  │
RMSNorm → lm_head (same matrix as tok_emb) → logits [B, S, 4096] ◄────┘
```

During generation, each Attention layer also reads and writes a `KVCache`
(kv_cache.py) so earlier tokens are not recomputed.

---

## 1. File by file

### `src/slm/model/config.py` — not ours, shared

**What it does:** holds every hyperparameter for the whole project. Person C
wrote it and it is now one shared file on `main`. Track B only reads it;
suggested changes are in section 4.

**Fields the model uses:** `vocab_size, d_model, n_layers, n_heads,
n_kv_heads, ffn_hidden, max_seq_len, rope_theta, rms_norm_eps,
tie_embeddings, dropout`, plus the derived properties `head_dim`
(= d_model // n_heads = 64) and `n_rep` (= n_heads // n_kv_heads).

### `src/slm/model/layers.py` — RMSNorm, SwiGLU, RoPE

**RMSNorm** — *keeps the numbers a sensible size.* Each token's 512-number
vector is divided by its root-mean-square, so its typical size becomes 1, then
multiplied by a learned per-feature gain that starts at 1. It is LayerNorm
without the mean subtraction or the bias: cheaper, and works as well.

```python
x32 = x.float()                                    # do the maths in fp32
inv_rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
return (x32 * inv_rms * self.weight).type_as(x)    # back to the input dtype
```

**SwiGLU** — *the feed-forward step, with a gate.* After attention mixes
information between tokens, the FFN processes each token on its own. SwiGLU
uses two "up" projections; one goes through SiLU and acts as a soft on/off
dial for each hidden unit, multiplied into the other:

```python
return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
```

Three matrices instead of the classic two, so the hidden size shrinks from
4 × 512 to 8/3 × 512 ≈ 1365, rounded up to 1408. That keeps the parameter count
about the same (3 × 512 × 1408 = 2.16M, against 2 × 512 × 2048 = 2.10M). That
is the default and MHA's value. The GQA preset uses 1664 to match MHA's total
parameters (S7, section 3).

**RoPE (rotary position embeddings)** — *how attention knows token order.*
Attention on its own can't tell "dog bites man" from "man bites dog". RoPE
splits each 64-number query/key head into 32 pairs, treats each pair as a point
on a plane, and **rotates** it by `position × freq_i`. Early pairs spin fast
(fine position) and late pairs spin slowly (coarse position), like a clock's
second, minute and hour hands. The key property: the dot product of two rotated
vectors depends only on the *difference* of their angles, so the score between
positions m and n depends only on **m − n**. It is applied to Q and K, never to V.

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # 32 frequencies
angles = torch.outer(positions, inv_freq)              # [max_seq_len, 32]
angles = torch.cat((angles, angles), dim=-1)           # both halves of a pair share an angle
...
return (x32 * cos + rotate_half(x32) * sin).type_as(x) # the 2-D rotation, all pairs at once
```

`start_pos` in `RotaryEmbedding.forward` tells RoPE where the chunk starts.
During cached generation the new token might be at position 37, not 0.

### `src/slm/model/attention.py` — one class for MHA, GQA and MQA

**What it does:** causal self-attention. Each token builds a query ("what am I
looking for?"), a key ("what do I contain?") and a value ("what do I pass
on?"). Scores are `softmax(Q·Kᵀ / √64)`, masked so a token only sees itself and
earlier tokens, and the output is the score-weighted sum of the values.

**Key concept — grouped-query attention.** In MHA, each of the 8 query heads
has its own K and V head. In GQA, query heads share K/V heads in groups. With
`n_kv_heads = 2`, query heads 0–3 share K/V head 0 and heads 4–7 share K/V
head 1. MQA is the extreme case: `n_kv_heads = 1`. One class handles all three:

```python
self.q_proj = nn.Linear(d_model, n_heads    * head_dim, bias=False)  # 512 → 512
self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)  # 512 → 128 for GQA-2
self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
...
if self.n_rep > 1 and not _USE_ENABLE_GQA:           # default path
    k = k.repeat_interleave(self.n_rep, dim=1)       # [B, 2, T, 64] → [B, 8, T, 64]
    v = v.repeat_interleave(self.n_rep, dim=1)
gqa_kw = {"enable_gqa": True} if (self.n_rep > 1 and _USE_ENABLE_GQA) else {}
```

Each K/V head has to serve `n_rep = n_heads // n_kv_heads` query heads. By
default `repeat_interleave` expands K/V to all 8 heads right before SDPA, and
the rest is plain MHA (the cache still stores 2 heads). With the opt-in
`SLM_SDPA_ENABLE_GQA=1`, read once at import, and only if `_probe_sdpa_gqa`
finds support (a real SDPA call, because SDPA is a C builtin and
`inspect.signature` raises on it), K/V go to SDPA at 2 heads with
`enable_gqa=True` and SDPA does the grouping. That was the default from 6 to
12 Nov; it was reverted after the T4 benchmark (design row 7, decision S14).

Both paths give the same grouping: 0,0,0,0,1,1,1,1, each group contiguous.
Plain `repeat` would give 0,1,0,1,… which is a different (wrong) grouping, and
a test catches it. With `n_kv_heads == n_heads`, `n_rep = 1`,
neither path applies, and the class *is* standard MHA (tested against a
reference).

**The other important lines:**

```python
q, k = self.rope(q, k, start_pos)                    # position goes in here, Q and K only
if kv_cache is not None:
    k, v = kv_cache.update(layer_idx, start_pos, k, v)   # stored at n_kv_heads, never expanded
...
if T == S:                                           # normal forward
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, ..., **gqa_kw)
else:                                                # cached: queries are the LAST S of T
    mask = torch.ones(S, T, dtype=torch.bool).tril(diagonal=T - S)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, ..., **gqa_kw)
```

### `src/slm/model/transformer.py` — the block and the full model

**TransformerBlock (pre-norm):**

```python
x = x + self.attn(self.attn_norm(x), start_pos, kv_cache, layer_idx)
x = x + self.ffn(self.ffn_norm(x))
```

**Key concept — the residual stream.** `x` passes from the embedding to the
output, and each sub-layer *adds* something to it. Pre-norm normalises the
copy going *into* each sub-layer and never `x` itself, so gradients can flow
straight down the stream. That is why deep pre-norm models train stably.

**Transformer:** embedding → 8 blocks → final RMSNorm → `lm_head`. It returns
**logits only**, because Person C's `compute_loss` computes the
cross-entropy (it needs `ignore_index=-100` for SFT/DPO masking).

```python
if cfg.tie_embeddings:
    self.lm_head.weight = self.tok_emb.weight    # one matrix, used twice
```

**Weight tying:** row *t* of the embedding is token *t*'s input vector, and
the same row scores token *t* at the output. This saves 4096 × 512 = 2,097,152
parameters (decision S4).

**Init (`_init_weights`):** every Linear and the embedding start at N(0, 0.02).
The two projections that write into the residual stream (`o_proj`, `w_down`)
start at 0.02 / √(2 × 8). That is 16 additions into the stream, so their scale
is shrunk so the stream doesn't grow with depth.

### `src/slm/model/kv_cache.py` — memory for generation

**What it does:** while generating, token *t+1* needs K and V from every
earlier token in every layer. Without a cache the model recomputes all of
them for each new token. With the cache it runs on just the new token and
reads the rest from memory.

**Key concept — this is where GQA saves memory:**

```python
shape = (n_layers, batch_size, n_kv_heads, max_seq_len, head_dim)   # n_kv_heads, NOT n_heads
self.k = torch.zeros(shape, ...)
self.v = torch.zeros(shape, ...)
```

K/V are stored at `n_kv_heads`, so GQA-2 needs 4× less cache than MHA. At
512 tokens in fp16, that's 2.10 MB against 8.39 MB per sequence.

How generation uses it (the same pattern appears in the cached-generation test):

```python
cache = KVCache(model.cfg, batch_size=1)
logits = model(prompt, kv_cache=cache)        # "prefill": whole prompt in one go
logits = model(next_token, kv_cache=cache)    # then one token per step
```

`Transformer.forward` reads `start_pos = cache.seq_len`, passes it to every
layer, and calls `cache.advance(S)` once all layers have written.

### `src/slm/utils/params.py` — parameter counting

**What it does:** counts parameters from the built model,
independently of Person C's formula in `ModelConfig.param_counts()`, so each
checks the other. Run `python -m slm.utils.params` for the table in section 3.

- `count_params(model)` → total / embedding / non-embedding. `named_parameters()`
  yields the tied matrix once, so it is counted once.
- `param_breakdown(model)` → per component (embedding, attention, ffn, norms).
- `matched_ffn_hidden(gqa_cfg, mha_cfg)` → the `ffn_hidden` that gives GQA
  the same total as MHA. Starting from 1408, **it's 1664, and the match is
  exact** (S7, section 3). The preset already uses 1664.

### Tests — `tests/test_attention.py`, `tests/test_model.py`

All of them use random token IDs (`torch.randint(0, 4096, (2, 16))`), so none
need the tokenizer. Each was also checked to **fail** on a deliberately broken
model. Re-run on the final code (2026-11-09): no causal mask, `repeat` instead
of `repeat_interleave` on the default path, `enable_gqa` missing on the cached
branch, a cached mask without the `T − S` offset, and RoPE ignoring `start_pos`
were each caught.

---

## 2. Choices made that the instructions didn't specify

| # | Picked | Alternatives | Why |
|---|---|---|---|
| 1 | Copy C's config.py verbatim onto this branch (own commit) | Branch off `track/training`; wait for her merge | Let us import `ModelConfig` during Phase 2. Merged since; there is now one config.py on `main`. |
| 2 | RoPE "half-split" pairing (feature *i* with *i*+32) via `rotate_half` | Adjacent pairs (0,1),(2,3) as in the paper; complex numbers as in Meta's Llama | Same maths up to a fixed reordering. It matches HuggingFace Llama, uses only real tensors, and complex numbers don't mix well with fp16 autocast. |
| 3 | One RoPE table per Attention layer | One shared table owned by Transformer | Attention works and can be tested on its own. Cost: about 260 KB per layer, 2 MB total. |
| 4 | RoPE cos/sin as `persistent=False` buffers | Normal buffers; recompute every forward | They move with `.to(device)` but stay out of checkpoints, since they can always be rebuilt. |
| 5 | fp32 maths inside RMSNorm and RoPE, cast back after | Whatever dtype comes in | C trains in fp16 on a T4. Squaring activations in fp16 overflows past 65,504. |
| 6 | `F.scaled_dot_product_attention` (fused) | Hand-written softmax | Faster and lower-memory. The hand-written version lives in the test as the reference. |
| 7 | `repeat_interleave` K/V to all heads before SDPA (default); SDPA's `enable_gqa=True` only as an opt-in (`SLM_SDPA_ENABLE_GQA=1` and `_probe_sdpa_gqa()`) | `enable_gqa` as the default (tried 6–12 Nov, track C); `expand`+`reshape` | `repeat_interleave` copies K/V on every decode step (2.15 GB per token at batch 64, context 2048, fp16), and on that path GQA decode was 48 % / 52 % below MHA at batch 64, context 1024 / 2048 (T4, 1 Nov). `enable_gqa` was meant to remove the copy, but on the T4 (sm_75, torch 2.11) it drops SDPA to the math backend and was worse: training −39.1 % vs MHA (spread 3.5 %, batch 48), training peak memory +58 %, decode slower at all 20 configs (−17 % to −71.3 % at context 1024, batch 64), OOM at context 2048, batch 64 where `repeat_interleave` completed. KV cache 4x smaller on both paths. Reverted 12 Nov; may differ on Ampere+ (FlashAttention), untested. Same output on both paths (tested on CPU), no weights changed. |
| 8 | Explicit offset mask when the cache already holds tokens | Always `is_causal=True` | `is_causal` aligns the mask to the top-left, which is wrong when the queries are the last S of T keys. Needed for multi-token chunks on a non-empty cache. |
| 9 | Dropout on embedding, attention probs, attention output, FFN output | Fewer places | The usual GPT-2 placement. `dropout = 0.0` in the config, so it is inactive anyway. |
| 10 | GPT-2 init: N(0, 0.02), residual projections scaled by 1/√(2·n_layers) | PyTorch defaults | The default embedding is N(0, 1): tied logits would be about 22 wide at step 0. Measured initial loss with our init: 8.33, against ln(4096) = 8.32 (ideal). |
| 11 | KV cache **pre-allocated** to `max_seq_len` | Grow with `torch.cat` each step | No re-copying per token, and memory equals `kv_cache_bytes()` exactly (tested). |
| 12 | KV cache is a plain object, not an `nn.Module` | `nn.Module` with buffers | It is temporary generation state: it should never land in a checkpoint or `state_dict`. |
| 13 | Cache dtype is a constructor argument (default fp32) | Infer from the first write | Explicit, and the size is known up front. Pass `torch.float16` for fp16 generation. |
| 14 | `start_pos` passed explicitly to every layer; cache advanced once by the model | Each layer tracks its own length | One source of truth, and every layer in a forward sees the same position. |
| 15 | Each file's tests committed with that file | All tests at the end | Every commit is tested; the files were still written in the requested order. |
| 16 | Extra tests beyond the list (GQA vs duplicated-weight MHA, RoPE relative property, chunked prefill, cache size, tying, state_dict round trip, per-component counts, matched ffn) | Only the listed tests | Each covers a specific bug class that the listed tests would miss. |
| 17 | Cached-vs-uncached test feeds both runs the *same* tokens and compares with `atol=1e-5` | Let each run pick its own argmax; demand bit-equality | Different matmul shapes round differently in the last bits. Feeding identical tokens means a near-tie can't make the runs diverge. |
| 18 | Overfit model: d_model 64, 2 layers, 4 heads, **2 KV heads**, lr 3e-3, 300 steps, pass if loss < 0.05 and every token recalled | lr 1e-2 (also works) | GQA path exercised; smoother curve; ~2 s. Reaches 0.004, so the threshold has margin. |
| 19 | "Non-embedding" excludes the token embedding (and an untied lm_head), counts the final norm | Other definitions | Matches C's `param_counts()` exactly. |
| 20 | Named it `count_params` | `count_parameters` | C's `training/utils.py` already has a `count_parameters`; avoids confusion. |
| 21 | Didn't write `generation/generate.py` (still only a docstring on `main`) | Write it | Not in the Phase 2 list, and probably another track's. The test contains a minimal greedy loop to copy from. |
| 22 | No new config fields | Ask C for `init_std` etc. | Nothing needed; the 0.02 init scale is a constant in transformer.py. |

---

## 3. Test results

```
tests/test_attention.py  18 passed
  mha_matches_reference                 n_kv_heads=8 output == naive per-head MHA (allclose)
  gqa_equals_mha_with_shared_kv_weights GQA-2 == MHA whose K/V heads are copied in groups
  variants_shapes [MHA, GQA-2, MQA]     projection sizes and output shape
  rope_score_depends_only_on_relative_position
  enable_gqa_matches_repeat_interleave_fallback [8, 4, 2, 1]
                                        both paths forced; forward, prefill, cached steps
  probe_reports_false_on_torch_without_enable_gqa
  default_kv_path_is_repeat_interleave  env unset -> K/V expanded to 8 heads, no enable_gqa keyword
  env_opt_in_switches_path [5 cases]    only SLM_SDPA_ENABLE_GQA=1 on a supporting torch opts in
  kv_path_reports_flag
tests/test_model.py      22 passed
  output_shape [8, 2, 1]                (2, 16, 4096), finite
  causality [t = 1, 7, 15]              logits before t unchanged; at/after t changed
  embeddings_are_tied, state_dict_round_trip
  cached_generation_matches_uncached [MHA, GQA]
  chunked_prefill_matches_full_forward
  kv_cache_size_matches_config [8, 2, 1], gqa_cache_is_4x_smaller_than_mha
  kv_cache_rejects_overflow
  param_count_matches_config_and_hand_calculation [MHA, GQA]
  param_breakdown_matches_config_per_component [MHA, GQA]
  matched_ffn_hidden_equalises_gqa_with_mha
  tiny_model_memorises_one_batch
33 passed in ~20 s
```

### Parameter counts — the hand calculation

Settings: d_model 512, n_heads 8, head_dim 64, vocab 4096, 8 layers. `ffn_hidden`
is 1408 for MHA and 1664 for GQA.
A linear layer from *a* to *b* features (no bias) has *a × b* weights.

| Per layer | MHA (n_kv_heads = 8) | GQA (n_kv_heads = 2) |
|---|---|---|
| q_proj | 512 × (8·64) = 262,144 | 512 × (8·64) = 262,144 |
| k_proj | 512 × (8·64) = 262,144 | 512 × (**2**·64) = 65,536 |
| v_proj | 512 × (8·64) = 262,144 | 512 × (**2**·64) = 65,536 |
| o_proj | (8·64) × 512 = 262,144 | 262,144 |
| **attention** | **1,048,576** | **655,360** |
| w_gate, w_up, w_down | 3 × 512 × 1408 = 2,162,688 | 3 × 512 × **1664** = 2,555,904 |
| 2 RMSNorm gains | 2 × 512 = 1,024 | 1,024 |
| **per layer** | **3,212,288** | **3,212,288** |

| Whole model | MHA | GQA |
|---|---|---|
| 8 layers | 25,698,304 | 25,698,304 |
| + final RMSNorm | 512 | 512 |
| **= non-embedding** | **25,698,816** | **25,698,816** |
| + embedding 4096 × 512 | 2,097,152 | 2,097,152 |
| + lm_head (tied) | 0 | 0 |
| **= total** | **27,795,968** (27.80M) | **27,795,968** (27.80M) |
| embedding share | 7.5 % | 7.5 % |

Why GQA was widened (S7): at `ffn_hidden` 1408 it would have 24,650,240
parameters, 3,145,728 fewer than MHA = 8 layers × 2 matrices (K, V) × 512 ×
(512 − 128). Per layer the K/V projections lose 393,216 weights, and 256 more
hidden units give back 3 × 512 × 256 = 393,216, so the match is exact.
The model's own count (`count_params`), C's formula (`param_counts()`) and
the hand calculation all agree exactly.

`python -m slm.utils.params`:

```
                           MHA  GQA-2 ffn=1408  GQA-2 ffn=1664
embedding            2,097,152       2,097,152       2,097,152
attention            8,388,608       5,242,880       5,242,880
ffn                 17,301,504      17,301,504      20,447,232
norms                    8,704           8,704           8,704
lm_head                      0               0               0
total               27,795,968      24,650,240      27,795,968
non_embedding       25,698,816      22,553,088      25,698,816
```

### KV cache size (fp16, 512 tokens)

`2 (K and V) × 8 layers × n_kv_heads × 64 × 512 × 2 bytes`:

| | batch 1 | batch 64 |
|---|---|---|
| MHA (8 KV heads) | 8.39 MB | 536.9 MB |
| GQA-2 | 2.10 MB | 134.2 MB |
| MQA (1 KV head) | 1.05 MB | 67.1 MB |

### Overfit check — loss curve

Tiny model (d_model 64, 2 layers, 4 query / 2 KV heads, ffn_hidden 256, vocab
4096; 385,344 parameters), one batch of 2 × 50 = 100 random target tokens,
AdamW lr 3e-3, no weight decay, CPU:

| step | 0 | 5 | 10 | 25 | 50 | 75 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loss | 8.331 | 6.680 | 5.371 | 2.088 | 0.187 | 0.043 | 0.024 | 0.012 | 0.008 | 0.006 | **0.004** |

It starts at ln(4096) = 8.318, as an untrained model should, and memorises all
100 tokens. Run time is about 2 s. (lr 1e-2 also works: 0.0013 at step 300.)

### Integration check with Person C's training loop

Her `pretrain.py`, `training/utils.py` and `utils/seed.py` were copied from
`track/training` into a throwaway folder outside the repo, then run unchanged
with our `Transformer` (tiny config, CPU, random data). It trained (loss 8.32
→ 6.97 over 30 steps), ran eval, saved `step_000020.pt`, and on a second run
**resumed at step 21** and continued to 6.84. Her optimizer reported 15 decayed
and 5 undecayed tensors, which is right: the 1-D norm gains are undecayed. The
contract `model(x) → logits` works as she expects.

---

## 4. Kickoff items (25 Oct) and where they ended up

1. **O1 — field naming. Settled as S6.** Research-code names
   (`n_layers, d_model, n_heads, n_kv_heads, …`). The old HF-named
   `configs/model_mha.yaml` / `model_gqa.yaml` are deleted, and
   `configs/mha_baseline.yaml` / `gqa_base.yaml` are generated from the presets.

2. **O2 — parameter fairness. Settled as S7.** GQA uses `ffn_hidden = 1664`
   (1408 + 256), which gives **exactly** MHA's 27,795,968 parameters, and
   1664 = 13 × 128, so it stays aligned like C's `round_to_multiple`. Trade-off
   to state in the paper: the matched GQA moves capacity from attention into
   the FFN.

3. **Notes on config.py for Person C** (suggestions only, not changed):
   - The presets do `cfg.model.n_kv_heads = …` *after* construction, so
     `__post_init__` validation doesn't run on the new value. A bad value
     like 3 wouldn't raise there; it would only fail later, when the model is
     built. Building `ModelConfig(n_kv_heads=…)` directly in the presets would
     validate.
   - A bare `ModelConfig()` is GQA-2 (default `n_kv_heads=2`). Fine, but
     worth everyone knowing; our tests always set it explicitly.
   - `kv_cache_bytes(seq_len=0)` silently means `max_seq_len` (`seq_len or …`). Minor.
   - `head_dim` is derived (`d_model // n_heads`), so it can't be set
     independently. Fine for this project.
   - Her optimizer applies weight decay to the tied embedding (it's 2-D). That's
     standard (nanoGPT does it), just worth knowing.

4. **For whoever owns generation/evaluation:** use `KVCache(model.cfg,
   batch_size, device=…, dtype=torch.float16 under fp16)`. Prefill the prompt,
   then feed one token at a time, and call `cache.reset()` between prompts.
   Prompt plus generated tokens must stay within `max_seq_len = 512`;
   overflow raises a clear error. The greedy loop in
   `test_cached_generation_matches_uncached` is a working template.

5. **For the attention benchmark:** GQA's saving in this code is in *memory*
   (K/V projections and, above all, the KV cache), not attention FLOPs: all 8
   query heads are still scored against every key. `results/benchmark/benchmark.json`
   (T4, fp16, 1 Nov) measures the `repeat_interleave` path, which is the
   default again, so it describes the current code. GQA vs MHA decode tokens/s:

   | batch × context | MHA | GQA | GQA vs MHA |
   |---|---|---|---|
   | 1 × 2048 | 86.5 | 88.9 | +2.8 % |
   | 16 × 2048 | 1,289 | 1,293 | +0.3 % |
   | 64 × 128 | 5,588 | 4,841 | −13.4 % |
   | 64 × 512 | 5,695 | 5,026 | −11.7 % |
   | 64 × 1024 | 5,441 | 2,821 | −48.2 % |
   | 64 × 2048 | 3,386 | 1,632 | −51.8 % |

   GQA also has the wider FFN (1664 against 1408), and the benchmark can't
   separate that from the copy, so the −12 to −13 % at batch 64 and short
   context shouldn't be put down to the copy alone.
   **The `enable_gqa` (opt-in) path was benchmarked on a Kaggle T4** (sm_75,
   torch 2.11) and is worse: SDPA drops to the math backend. GQA training
   −39.1 % vs MHA (spread 3.5 %, `--train-batch 48`), training peak memory
   +58 % vs MHA, decode slower than MHA at all 20 configs (−17 % to −71.3 %,
   the worst at context 1024, batch 64), and OOM at context 2048, batch 64,
   where the `repeat_interleave` run completed. KV cache exactly 4x smaller,
   as on the default path. Hence the revert (decision S14). The file goes in
   `results/benchmark_enable_gqa/benchmark.json` once committed; `make_results.py`
   labels it "enable_gqa (opt-in) path, T4". (On CPU, torch 2.14.1, one SDPA
   call at batch 64, context 2048 took 6.7 ms with `enable_gqa` against 64 ms
   with `repeat_interleave`, which shows CPU timings don't transfer to the T4.)

6. **Merging:** done. `track/model`, `track/training` and `track/tokenizer` are
   all in `main`.

7. `ROADMAP.md` isn't in the repo. Worth adding so everyone works from the same plan.

---

## 5. Likely viva questions

1. **What does `n_kv_heads` control, and how do you know `n_kv_heads = n_heads`
   is real MHA?** It's the number of K/V heads. Query heads are split into
   groups that share one K/V head. When it equals `n_heads`, `n_rep = 1`, no
   grouping happens, and `test_mha_matches_reference` shows the output equals a
   textbook per-head MHA.

2. **Where exactly does GQA save memory, and does it save compute?** The K/V
   projections shrink (8.39M → 5.24M attention parameters in total), and the KV
   cache shrinks 4× (8.39 → 2.10 MB at 512 tokens). Attention compute does not
   shrink, because all 8 query heads are still scored against every key. S7
   gives the saved parameters back to the FFN, so GQA's total is the same as
   MHA's. The remaining saving is cache memory during generation.

3. **Where does 27,795,968 come from, and why does 24.65M appear?** Per layer:
   4 attention matrices (K and V are 512×128 in GQA instead of 512×512), 3 FFN
   matrices (512×1408 for MHA, 512×1664 for GQA), and 2 norm vectors. Multiply
   by 8 layers, add the final norm and the 4096×512 embedding, and add nothing
   for the tied head. GQA at 1408 would be 24.65M, short by
   8 × 2 × 512 × 384 = 3,145,728. Widening by 256 returns exactly that (S7).

4. **Why RoPE rather than learned position embeddings?** No parameters, and the
   attention score depends only on relative distance (m − n), which is what
   language needs. Learned absolute embeddings give every position a separate
   vector that the model must learn to compare.

5. **Why rotate Q and K but not V? And why can cached keys be stored already
   rotated?** Position should affect *who attends to whom* (the Q·K score), not
   *what information is passed* (V). Each token's rotation depends only on its
   own absolute position, which never changes, so a key is rotated once when
   written and stays valid. That's what makes RoPE cache-friendly.

6. **Why pre-norm, and why RMSNorm?** Pre-norm keeps the residual stream
   un-normalised, so gradients flow straight through all 8 blocks and training
   is stable without delicate warmup. RMSNorm drops LayerNorm's mean-centring
   and bias: fewer operations, and the same quality in practice (Llama uses it).

7. **Why is `ffn_hidden` 1408 and not 2048, and why is GQA's 1664?** SwiGLU has
   three matrices, not two. 8/3 × 512 ≈ 1365 keeps the FFN parameter count close
   to a classic 4 × 512 two-matrix FFN; rounding up to 1408 (11 × 128) suits GPU
   matmuls. GQA's 1664 (13 × 128) is that plus 256, to spend the parameters saved
   on K/V (S7).

8. **What is weight tying, and what does it save?** The output head reuses the
   embedding matrix. That saves 2,097,152 parameters (7.5% of MHA) and forces
   input and output to agree on what each token means. Tests check it survives
   a checkpoint round trip.

9. **How do you know the model is causal?** `test_causality` changes token *t*
   and checks that logits at every position before *t* are unchanged, and
   that logits from *t* onward did change (otherwise the test proves nothing).

10. **Why does the loss start at 8.32, and what does the overfit test prove?**
    An untrained model should give roughly equal probability to all 4096
    tokens, so the loss is ln(4096) = 8.32. A much higher start means a bad
    init. Driving the loss on one batch to ~0 proves gradients reach every
    parameter and nothing (masking, tying, init) blocks learning. It doesn't
    prove the model generalises, only that it can learn.

11. **Why does cached attention need a different mask?** With 10 cached tokens
    and 3 new ones, the 3 queries sit at positions 10–12, so query *i* may see
    keys 0..10+i. `is_causal=True` assumes queries start at position 0, which
    would hide keys they are allowed to see. The offset mask
    `tril(diagonal=T−S)` fixes that, and `test_chunked_prefill_matches_full_forward`
    checks it.

12. **Why `repeat_interleave` and not SDPA's `enable_gqa`?** We tried
    `enable_gqa` as the default (6 Nov), because `repeat_interleave` copies K/V
    to all 8 heads on every decode step (2.15 GB per generated token at batch
    64, context 2048, fp16) and GQA decoded 52 % slower than MHA there. Then we
    measured it on the T4 (sm_75, torch 2.11): `enable_gqa=True` makes SDPA
    fall back to the math backend, so it was worse everywhere except the cache.
    GQA training −39.1 % vs MHA, training peak memory +58 %, decode slower at
    all 20 configs (down to −71.3 % at context 1024, batch 64), and OOM at
    context 2048, batch 64, where `repeat_interleave` finished. The KV cache is
    4x smaller on both paths, since it stores K/V at 2 heads either way. So
    `repeat_interleave` is the default again and `enable_gqa` is an opt-in
    (`SLM_SDPA_ENABLE_GQA=1`). On Ampere or newer, with FlashAttention, it might
    win; we haven't tested that. The output is the same on both paths:
    `test_enable_gqa_matches_repeat_interleave_fallback` forces both on the same
    weights and compares forward and cached steps (CPU).
