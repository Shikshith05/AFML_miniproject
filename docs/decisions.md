# Design Decisions

Every architectural and methodological choice, with the reasoning behind it.
Written as decisions are made, not reconstructed afterwards.

**This file is graded.** The course states that evaluation weights
"understanding of the model itself... the reasoning behind the decisions that
led to that architecture." This log is that reasoning, and the paper's method
section is written from it.

**Format:** one entry per decision. Date, what was chosen, what was rejected,
why. Sign your entries so we know who to ask.

---

## OPEN — to settle in the kickoff session (Sat 25 Oct)

### O1. Config field naming

`configs/model_mha.yaml` uses HuggingFace-style names (`num_layers`,
`hidden_size`, `num_attention_heads`, `max_position_embeddings`). The draft
`config.py` uses research-code style (`n_layers`, `d_model`, `n_heads`,
`max_seq_len`).

Pick one and convert everything. Either is defensible; having both is not.

**Urgent sub-issue:** `configs/model_gqa.yaml` has
`num_key_value_groups: 2`. In HF's Llama implementation that name means the
*repeat factor* (`n_heads // n_kv_heads`), not the number of KV heads. So
`2` is ambiguous — it reads as either 2 KV heads (4 queries each) or 4 KV
heads (2 queries each). Rename to `n_kv_heads` and state the number of KV
heads explicitly.

- Decision:
- Rationale:

### O2. Parameter-count fairness in the ablation

With identical settings apart from `n_kv_heads`, MHA has **27.80M**
parameters and GQA has **24.65M** — GQA is 3.15M smaller. An unadjusted
comparison confounds the attention variant with model capacity, and GQA could
look worse purely from having fewer parameters.

Options:
1. Report perplexity against parameter count, and acknowledge the gap
2. Raise GQA's `ffn_hidden` to equalize total parameters
3. Report both

Option 2 is the cleaner experiment; option 1 is cheaper. **Decide before the
runs start** — it cannot be fixed afterwards.

- Decision:
- Rationale:

### O3. Per-run time budget → `max_steps`

Draft `max_steps=20000` is ~1.31B tokens, likely 4–6 h on a T4. With two
pretraining runs needed and a 15 Nov deadline, that may be too long.

Set a wall-clock budget per run (suggest 3 h), measure throughput in the
smoke run, then derive `max_steps` from it.

- Decision:
- Rationale:

### O4. Batch size and precision

`micro_batch_size=16` is a guess; tune to just under OOM, then set
`grad_accum_steps` to hold the effective batch at 128.

`precision="fp16"` is correct for a Kaggle T4 (Turing — no bf16 support).
Revisit only if we get a P100 or better.

- Decision:
- Rationale:

### O5. Vocabulary size

Draft is 4096. The justification is that a 50k GPT-2 vocab would put more
parameters in the embedding table than in the entire rest of the model. At
4096 the embedding is ~8% of parameters.

Confirm after A measures fertility on TinyStories.

- Decision:
- Rationale:

---

## SETTLED

### S1. Decoder-only transformer, Llama-style components
**Date:** 2026-10-24 · **By:**

RMSNorm (pre-norm), RoPE, SwiGLU, no biases. This is the architecture the
GQA paper studies, so our ablation is comparable to published results.
Deviating would make the comparison harder to interpret, not easier.

### S2. Two pretraining runs, not four
**Date:** 2026-10-24 · **By:**

GQA (architecture/efficiency) and DPO (alignment) are independent axes, so a
2x2 grid is unnecessary. Train MHA and GQA bases, benchmark both, then carry
only the GQA model through SFT and DPO.

```
TinyStories --+--> MHA base --> benchmark only
              +--> GQA base --> benchmark --> SFT --> DPO
```

Motivated partly by the 23-day timeline, but methodologically sound
regardless. Must be stated explicitly in the paper.

### S3. In-domain preference data for DPO
**Date:** 2026-10-24 · **By:**

Standard preference sets (Anthropic HH-RLHF) are multi-turn adult assistant
dialogue — far out of distribution for a model pretrained only on
TinyStories, so DPO on them would optimize noise.

Instead: chosen = ground-truth continuation from held-out data; rejected =
a generation from an earlier, weaker checkpoint of our own model.

Trade-off, stated honestly in the paper: these are synthetic preferences
over story quality, not human preferences over helpfulness. The mechanism is
the same; the semantics are narrower.

### S4. Weight tying
**Date:** 2026-10-24 · **By:**

Input embedding and output projection share one matrix. Saves 2.1M
parameters (~8% of the model) and generally helps at this scale — both
directions are forced to agree on what each token means.

### S5. Matched conditions for the ablation
**Date:** 2026-10-24 · **By:**

MHA and GQA runs use identical seed (1337), identical data order
(`deterministic_data_order=True`), identical token budget, and are launched
from the same commit. Without this the comparison proves nothing.

---

## Template

```
### Sn. <one-line summary>
**Date:** YYYY-MM-DD · **By:** <name>

Chosen: <what>
Rejected: <what else was considered>
Why: <the actual reason, with a number if there is one>
```
