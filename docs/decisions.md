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

O3, O4 and O5 were settled from the pretraining runs: see S11, S12 and S13.

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

### S8. DPO pairs use the SFT instruction template
**Date:** 2026-11-07 · **By:** Suhas

Chosen: each pair is "### Instruction: Continue the story." + the first 48
tokens of a held-out story, with chosen = the real remainder and rejected =
the weak checkpoint's continuation (S3), formatted by the same
`format_prompt` / `build_example` as SFT.

Rejected: raw story continuations with no template.

Why: DPO starts from the SFT model. Raw continuations would train it on a
format SFT just taught it to leave, so DPO would partly undo SFT. With the
template, SFT -> DPO is one format throughout. The weak model itself is a
base checkpoint, so it is prompted raw (`<bos>` + opening) to get its best
attempt; only the resulting text is placed in the template.

Held-out source is the HF `validation` split: pretraining carved its val set
from HF `train`, so neither the model nor the tokenizer has seen these
stories. Only stories ending within 192 tokens are used, and the weak model
gets the same 192-token budget. Mean chosen vs rejected lengths are recorded
in `data/preference/stats.json` as a possible length confound.

### S9. Reference log-probs precomputed; one model in memory
**Date:** 2026-11-07 · **By:** Suhas

Chosen: compute the frozen SFT reference's log-probs for every pair once,
save to `data/ref_logprobs_{train,test}.npz` with a fingerprint of the pairs,
then train the same model object as the policy.

Rejected: keeping a second frozen copy of the model on the GPU.

Why: the reference never changes, so recomputing it every step is wasted
compute, and two 28M models plus activations for chosen and rejected is the
memory pressure we can avoid on a T4. The fingerprint makes a stale cache
recompute instead of silently giving wrong rewards.

### S10. Evaluation: same prompts, same seed, three views
**Date:** 2026-11-07 · **By:** Suhas

Chosen: for base, SFT and DPO -- (1) perplexity on held-out TinyStories as a
forgetting check, (2) samples on 12 fixed prompts in the shared template with
per-prompt seeds, (3) preference accuracy on 200 held-out pairs, both summed
and per-token, plus DPO's implicit-reward accuracy against SFT.

Rejected: an LLM-as-judge win rate (needs an external API and judges
adult-assistant quality our model cannot have); a single preference number
(summed log-prob favours short responses, so it can mislead on its own).

Why: each view answers a different question -- did fluency survive, what
changed in behaviour, does the model prefer real text -- and none is
sufficient alone.

### S11. Pretraining length: 12,000 steps (was O3)
**Date:** 2026-10-29 (MHA run) · **By:** Mythri (runs); recorded by Shikshith

Chosen: `max_steps = 12,000` for both pretraining runs. At 85 x 512 tokens
per step that is 522,240,000 tokens, about 1.08 epochs of the 481.6M-token
train split.

Rejected: the draft 20,000 steps (1.31B tokens, about 2.7 epochs).

Why: a wall-clock budget per run (the notebook's probe is run with
`--budget-hours 3`), with two runs to fit before 15 Nov. Measured from
`results/*/metrics.jsonl`: MHA took 14,188 s (3.94 h) and GQA 11,463 s
(3.18 h) on a Kaggle T4, in different sessions. At those rates the draft
20,000 steps would have been about 6.6 h and 5.3 h.
Validation loss was still falling slowly at the end (MHA 1.3557, GQA 1.3530),
so more steps would have helped both equally; the comparison holds because
the budget is identical (S5).

### S12. Batch size and precision: micro-batch 85, no accumulation, fp16 (was O4)
**Date:** 2026-10-29 (MHA run) · **By:** Mythri (runs); recorded by Shikshith

Chosen: `micro_batch_size = 85`, `grad_accum_steps = 1` (effective batch
85 sequences, 43,520 tokens per step), `precision = "fp16"`, identical for
MHA and GQA (values from the `start` config in both metrics logs).

Rejected: the draft 16 x 8 = 128; bf16.

Why: 85 comes from `scripts/probe_batch_size.py`, which finds the largest
micro-batch that fits the T4 with a 0.9 safety factor. fp16 because the T4
(Turing) has no bf16 tensor cores. The probe would suggest accumulation 2
to approach the 128 target; the runs used 1, so the effective batch is 85,
not 128. The probe output itself was not committed, so the reason for
dropping accumulation is not recorded here -- to confirm with Mythri. It
does not affect the ablation: both runs used the same values.

### S13. Vocabulary size 4096 (was O5)
**Date:** 2026-10-24 · **By:** Saurav (track A); recorded by Shikshith

Chosen: byte-level BPE with 4096 tokens, trained on the TinyStories train
split only.

Rejected: GPT-2's 50,257-token vocabulary.

Why (measured, `docs/track_a_notes.md`): fertility on the validation text
is 1.297 tokens per word against 1.263 for GPT-2, so GPT-2 would save only
about 2.5% of tokens. Its embedding alone would be 50,257 x 512 = 25.7M
parameters, nearly the whole budget; ours is 4096 x 512 = 2,097,152, 7.5%
of the final 27,795,968 (tied with the LM head, S4). `<unk>` rate is 0.0%
(byte-level).

### S14. SDPA's `enable_gqa`: tried, measured, reverted on T4
**Date:** 2026-11-06, revised 2026-11-12 · **By:** Mythri (code, `5951d91`, `6459608`); revert and record by Shikshith

Chosen (current): GQA layers expand K/V with `repeat_interleave` to all
`n_heads` right before SDPA, the original code. SDPA's `enable_gqa=True`
(K/V passed at `n_kv_heads`, SDPA groups the query heads) is kept as an
opt-in only: `SLM_SDPA_ENABLE_GQA=1`, read once at import, and used only
if `attention._probe_sdpa_gqa()` finds support (torch >= 2.5; probed by a
real call because SDPA is a C builtin and `inspect.signature` fails on it).
The KV cache stores K/V at `n_kv_heads` on both paths, so it is 4x smaller
than MHA's either way.

Tried (6 Nov): `enable_gqa=True` as the default whenever torch supports it.
The motivation: `repeat_interleave` copies K/V to all 8 heads on every
forward, including every decode step (2.15 GB written per generated token at
batch 64, context 2048, fp16), and in `results/benchmark/benchmark.json`
(T4, 1 Nov, `repeat_interleave` path) GQA decode tokens/s vs MHA was -48.2%
at batch 64, context 1024 and -51.8% at context 2048; at batch 1 and 16 the
two were within a few percent. (At batch 64 with short contexts GQA was
12-13% slower too, but GQA's wider FFN also costs time there.)

Measured (Kaggle T4, sm_75, torch 2.11, `enable_gqa` path): worse than
`repeat_interleave` on every count except the cache. On this GPU
`enable_gqa=True` drops SDPA to the math backend, which materialises the
full attention matrix:
- training throughput: GQA -39.1% vs MHA (spread 3.5%, `--train-batch 48`);
- training peak memory: +58% vs MHA;
- decode slower than MHA at all 20 configs, from -17% to -71.3% (context
  1024, batch 64);
- OOM at context 2048, batch 64, where the `repeat_interleave` run completed;
- KV cache still exactly 4x smaller (as expected: the cache does not depend
  on the path).

Reverted (12 Nov): `repeat_interleave` is the default again.

Revisit: on Ampere or newer, where SDPA's FlashAttention kernel may accept
grouped K/V without falling back, `enable_gqa` could win. Untested here.

Result file: `results/benchmark_enable_gqa/benchmark.json` (records
`kv_path` / `sdpa_enable_gqa` in the JSON). _To be filled in when it is
committed: link the file and confirm the numbers above against it._

Equivalence: `tests/test_attention.py::test_enable_gqa_matches_repeat_interleave_fallback`
forces both paths on the same weights for `n_kv_heads` in 8, 4, 2, 1 and
compares a full forward, a prefill and cached decode steps (CPU);
`test_default_kv_path_is_repeat_interleave` and `test_env_opt_in_switches_path`
pin the default and the opt-in. No weights, shapes or outputs change, so the
trained checkpoints are unaffected by either path.

---

## OPEN — track D

### O6. SFT dataset
Alpaca is model-generated (text-davinci-003); the TA advised against
synthetic data. `--dataset dolly` (Dolly-15k, human-written) and
`--dataset oasst1` (OpenAssistant, first-turn top-ranked English replies)
are implemented.

- Decision (2026-11-09, Suhas): use the existing Alpaca SFT checkpoint
  (`gqa_sft/final.pt`, run by Mythri, Kaggle notebook60e6ad04ff) and proceed
  to DPO. That run: code at `6459608` (same template and masking as the DPO
  pairs), started from `gqa_base/final.pt`, 45,379 of 52,002 Alpaca examples
  kept, 2 epochs at lr 2e-5, val loss 4.21 -> 2.50 (ppl 67.7 -> 12.2).
- Rationale: the TA was asked which human-written dataset to use but had
  not replied by 9 Nov, and the deadline is 15 Nov. Re-running SFT on Dolly
  or oasst1 remains a one-flag change (`--dataset`) if time allows.
- Limitation to state in the write-up: Alpaca is model-generated
  (text-davinci-003), which the TA advised against.
- 2026-11-12 (Shikshith): an oasst1 re-run (SFT -> DPO -> evaluate, outputs
  `gqa_sft_oasst1`, `gqa_dpo_oasst1`, `results/eval_oasst1`, Alpaca results
  untouched) is prepared on branch `shikshith/final-prep` and pending; steps
  in `docs/RUNBOOK.md`.

---

## Template

```
### Sn. <one-line summary>
**Date:** YYYY-MM-DD · **By:** <name>

Chosen: <what>
Rejected: <what else was considered>
Why: <the actual reason, with a number if there is one>
```
