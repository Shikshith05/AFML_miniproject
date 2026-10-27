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

### S6. Config field naming: research-code style, YAMLs generated from code
**Date:** 2026-10-27 · **By:**

Chosen: `n_layers` / `d_model` / `n_heads` / `n_kv_heads` / `max_seq_len`, as
in `src/slm/model/config.py`.

Rejected: the HuggingFace-style names that were in the original
`configs/model_mha.yaml` (`num_layers`, `hidden_size`,
`num_attention_heads`, `max_position_embeddings`).

Why: `config.py` is the only definition any code actually imports -- the old
YAMLs were placeholders nothing read, so keeping two vocabularies bought
nothing and invited drift. `configs/model_gqa.yaml` also had
`num_key_value_groups: 2`, which is ambiguous: in HuggingFace's Llama code
that name means the repeat factor (`n_heads // n_kv_heads`), so it reads as
either 2 KV heads or 4. `n_kv_heads` states the head count directly, and the
repeat factor is the derived property `n_rep`.

Those two files are deleted. `configs/mha_baseline.yaml` and
`configs/gqa_base.yaml` are now generated from the presets via
`Config.to_yaml`, so they cannot drift from the code.

### S7. GQA is parameter-matched to MHA by widening the FFN
**Date:** 2026-10-27 · **By:**

Chosen: GQA uses `ffn_hidden = 1664` (MHA keeps 1408). Both presets are then
**27,795,968** parameters exactly.

Rejected: (a) running GQA at its natural 24.65M and reporting perplexity
against parameter count; (b) shrinking MHA to match.

Why: GQA shrinks the K/V projections, which is where its memory saving comes
from, but that also removes 3,145,728 parameters -- 11% of the model. Run
unadjusted and a worse GQA perplexity has two candidate explanations, the
architecture and the smaller capacity, with no way to separate them. Giving
those parameters back in the feed-forward network equalises capacity while
leaving the KV cache 4x smaller (8.4 MB vs 2.1 MB at seq 512, batch 1), so
`n_kv_heads` is the only free variable.

Honest caveat for the paper: this is no longer "the same model with GQA
dropped in" but the same parameter budget spent differently. That is the
more useful question anyway -- given a fixed budget, is it better spent on
full attention, or on cheaper attention plus a wider FFN? -- but it must be
stated rather than left for a reader to infer.

---

## Template

```
### Sn. <one-line summary>
**Date:** YYYY-MM-DD · **By:** <name>

Chosen: <what>
Rejected: <what else was considered>
Why: <the actual reason, with a number if there is one>
```
