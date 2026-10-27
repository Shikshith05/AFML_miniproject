# Person B notes — Phase 2: model, attention, tests

Branch `track/model` (from `main`, not pushed). Status on 2026-10-24: **all
files written, 28/28 tests passing** (`.venv\Scripts\python.exe -m pytest -q`,
about 13 s on CPU).

| Commit | What |
|---|---|
| `5a97758` | config.py copied verbatim from `track/training` |
| `ad800b5` | layers.py — RMSNorm, SwiGLU, RoPE |
| `b32db66` | attention.py + tests/test_attention.py |
| `048c1a2` | transformer.py + first tests in tests/test_model.py |
| `1d57293` | kv_cache.py + cache tests |
| `076dbbc` | utils/params.py + parameter-count tests |
| `d104c8f` | overfit test |

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

### `src/slm/model/config.py` — not ours, copied

**What it does:** holds every hyperparameter for the whole project. Person C
wrote it on `track/training`; our branch has a byte-identical copy so the model
can import `ModelConfig`. **Never edit it here** — suggested changes are in
section 4.

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
about the same (3 × 512 × 1408 = 2.16M, against 2 × 512 × 2048 = 2.10M).

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
if self.n_rep > 1:                                   # n_rep = n_heads // n_kv_heads
    k = k.repeat_interleave(self.n_rep, dim=1)       # [B, 2, T, 64] → [B, 8, T, 64]
    v = v.repeat_interleave(self.n_rep, dim=1)
```

After the repeat, the maths is plain MHA. With `n_kv_heads == n_heads`,
`n_rep = 1`, the repeat is skipped, and the class *is* standard MHA (tested
against a reference). `repeat_interleave` gives 0,0,0,0,1,1,1,1, i.e. each
group is contiguous. Plain `repeat` would give 0,1,0,1,… which is a different
(wrong) grouping, and a test catches it.

**The other important lines:**

```python
q, k = self.rope(q, k, start_pos)                    # position goes in here, Q and K only
if kv_cache is not None:
    k, v = kv_cache.update(layer_idx, start_pos, k, v)   # stored BEFORE the repeat
...
if T == S:                                           # normal forward
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, ...)
else:                                                # cached: queries are the LAST S of T
    mask = torch.ones(S, T, dtype=torch.bool).tril(diagonal=T - S)
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

K/V are stored *before* the repeat, so GQA-2 needs 4× less cache than MHA. At
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
  the same total as MHA. **It's 1664, and the match is exact** (see section 4).

### Tests — `tests/test_attention.py`, `tests/test_model.py`

All of them use random token IDs (`torch.randint(0, 4096, (2, 16))`), so none
need the tokenizer. Each was also checked to **fail** on a deliberately broken
model: no causal mask, `repeat` instead of `repeat_interleave`, and RoPE
ignoring `start_pos` were each caught.

---

## 2. Choices made that the instructions didn't specify

| # | Picked | Alternatives | Why |
|---|---|---|---|
| 1 | Copy C's config.py verbatim onto this branch (own commit) | Branch off `track/training`; wait for her merge | Lets us import `ModelConfig` now. An identical file merges cleanly, and if she edits hers first we take her version. |
| 2 | RoPE "half-split" pairing (feature *i* with *i*+32) via `rotate_half` | Adjacent pairs (0,1),(2,3) as in the paper; complex numbers as in Meta's Llama | Same maths up to a fixed reordering. It matches HuggingFace Llama, uses only real tensors, and complex numbers don't mix well with fp16 autocast. |
| 3 | One RoPE table per Attention layer | One shared table owned by Transformer | Attention works and can be tested on its own. Cost: about 260 KB per layer, 2 MB total. |
| 4 | RoPE cos/sin as `persistent=False` buffers | Normal buffers; recompute every forward | They move with `.to(device)` but stay out of checkpoints, since they can always be rebuilt. |
| 5 | fp32 maths inside RMSNorm and RoPE, cast back after | Whatever dtype comes in | C trains in fp16 on a T4. Squaring activations in fp16 overflows past 65,504. |
| 6 | `F.scaled_dot_product_attention` (fused) | Hand-written softmax | Faster and lower-memory. The hand-written version lives in the test as the reference. |
| 7 | `repeat_interleave` to expand K/V (as asked) | `expand`+`reshape`; SDPA's `enable_gqa=True` | Requested, and clearest to read. Alternatives avoid the copy; worth trying only if the benchmark needs speed. |
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
| 21 | Didn't write `generation/generate.py` | Write it | Not in the Phase 2 list, and probably another track's. The test contains a minimal greedy loop to copy from. |
| 22 | No new config fields | Ask C for `init_std` etc. | Nothing needed; the 0.02 init scale is a constant in transformer.py. |

---

## 3. Test results

```
tests/test_attention.py   6 passed
  mha_matches_reference                 n_kv_heads=8 output == naive per-head MHA (allclose)
  gqa_equals_mha_with_shared_kv_weights GQA-2 == MHA whose K/V heads are copied in groups
  variants_shapes [MHA, GQA-2, MQA]     projection sizes and output shape
  rope_score_depends_only_on_relative_position
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
28 passed in ~13 s
```

### Parameter counts — the hand calculation

Settings: d_model 512, n_heads 8, head_dim 64, ffn_hidden 1408, vocab 4096, 8 layers.
A linear layer from *a* to *b* features (no bias) has *a × b* weights.

| Per layer | MHA (n_kv_heads = 8) | GQA (n_kv_heads = 2) |
|---|---|---|
| q_proj | 512 × (8·64) = 262,144 | 512 × (8·64) = 262,144 |
| k_proj | 512 × (8·64) = 262,144 | 512 × (**2**·64) = 65,536 |
| v_proj | 512 × (8·64) = 262,144 | 512 × (**2**·64) = 65,536 |
| o_proj | (8·64) × 512 = 262,144 | 262,144 |
| **attention** | **1,048,576** | **655,360** |
| w_gate, w_up, w_down | 3 × 512 × 1408 = 2,162,688 | 2,162,688 |
| 2 RMSNorm gains | 2 × 512 = 1,024 | 1,024 |
| **per layer** | **3,212,288** | **2,819,072** |

| Whole model | MHA | GQA |
|---|---|---|
| 8 layers | 25,698,304 | 22,552,576 |
| + final RMSNorm | 512 | 512 |
| **= non-embedding** | **25,698,816** | **22,553,088** |
| + embedding 4096 × 512 | 2,097,152 | 2,097,152 |
| + lm_head (tied) | 0 | 0 |
| **= total** | **27,795,968** (27.80M) | **24,650,240** (24.65M) |
| embedding share | 7.5 % | 8.5 % |

The gap is 3,145,728 = 8 layers × 2 matrices (K, V) × 512 × (512 − 128).
The model's own count (`count_params`), C's formula (`param_counts()`) and
the hand calculation all agree exactly.

`python -m slm.utils.params`:

```
                           MHA           GQA-2  GQA-2 ffn=1664
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

## 4. For the kickoff (Sat 25 Oct)

1. **O1 — field naming.** The model reads `n_layers, d_model, n_heads,
   n_kv_heads, ffn_hidden, max_seq_len, rope_theta, rms_norm_eps,
   tie_embeddings, dropout` (plus `head_dim`, `n_rep`). Config is only read in
   constructors, so if we switch to HuggingFace names, renaming is about 30
   lines across attention.py, transformer.py, kv_cache.py and params.py. Either way,
   `configs/model_mha.yaml` / `model_gqa.yaml` on `main` still use HF names and
   must be converted. `num_key_value_groups: 2` is ambiguous: it should become
   `n_kv_heads: 2`.

2. **O2 — parameter fairness. There is a clean option-2 number.** Setting GQA's
   `ffn_hidden = 1664` (1408 + 256) gives **exactly** MHA's 27,795,968
   parameters, and 1664 = 13 × 128, so it stays aligned like C's
   `round_to_multiple`. Trade-off to state in the paper: the matched GQA moves
   capacity from attention into the FFN. Suggestion: run MHA vs GQA-2 at equal
   settings (the planned ablation), and report the parameter gap next to
   perplexity. If there is time, a matched GQA-2 run makes the comparison
   airtight. Option 3 in decisions.md.

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
   (K/V projections and, above all, the KV cache), not attention FLOPs.
   `repeat_interleave` rebuilds full-size K/V before the matmul, so attention
   compute is the same as MHA. Measure cache memory and decode tokens/s at
   large batch × long sequence, where memory bandwidth dominates. Expect little
   speed difference at batch 1 on short sequences.

6. **Merging:** `track/model`'s config.py is byte-identical to
   `track/training`'s. Merging either branch first is clean. If C changes
   config.py before merging, resolve by taking her version.

7. `ROADMAP.md` isn't in the repo. Worth adding so everyone works from the same plan.

---

## 5. Likely viva questions

1. **What does `n_kv_heads` control, and how do you know `n_kv_heads = n_heads`
   is real MHA?** It's the number of K/V heads. Query heads are split into
   groups that share one K/V head. When it equals `n_heads`, `n_rep = 1`, the
   repeat is skipped, and `test_mha_matches_reference` shows the output equals a
   textbook per-head MHA.

2. **Where exactly does GQA save memory, and does it save compute?** The K/V
   projections shrink (8.39M → 5.24M attention parameters in total), and the KV
   cache shrinks 4× (8.39 → 2.10 MB at 512 tokens). Attention compute does not
   shrink here, because K/V are repeated back to 8 heads before the matmul. The
   win comes during generation, where reading the cache from memory is the
   bottleneck.

3. **Where do 27.80M and 24.65M come from?** Per layer: 4 attention matrices
   (K and V are 512×128 in GQA instead of 512×512), 3 FFN matrices of
   512×1408, and 2 norm vectors. Multiply by 8 layers, add the final norm and
   the 4096×512 embedding, and add nothing for the tied head. The difference is
   8 × 2 × 512 × 384 = 3,145,728.

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

7. **Why is `ffn_hidden` 1408 and not 2048?** SwiGLU has three matrices, not
   two. 8/3 × 512 ≈ 1365 keeps the FFN parameter count close to a classic
   4 × 512 two-matrix FFN; rounding up to 1408 (11 × 128) suits GPU matmuls.

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
