# Runbook: the remaining Kaggle runs

Three jobs: (a) the oasst1 SFT -> DPO -> evaluation chain, (b) the benchmark
re-run, (c) rebuilding the figures and tables. (a) and (b) need a Kaggle GPU;
(c) runs on a laptop in seconds. All code is on branch `main`;
the notebooks already point at it.

| Job | Where | GPU time | Inputs |
|---|---|---|---|
| (a) oasst1 chain | `notebooks/track_d_kaggle.ipynb` | under 1 h (estimate: the Alpaca chain incl. pairs took ~1 h; oasst1 has fewer examples) | `gqa_base/final.pt` |
| (b) benchmark | `notebooks/afml_kaggle_training.ipynb` (or the same run as (a)) | ~15 min | none |
| (c) figures/tables | laptop | none (seconds, CPU) | the files from (a) and (b) |

Kaggle settings for (a) and (b): *Session options* -> Accelerator **GPU T4**
(x1 or x2; only one GPU is used), Internet **On**. Run with *Save Version ->
Save & Run All*.

---

## (a) oasst1 chain: SFT -> DPO -> evaluate

1. Open `notebooks/track_d_kaggle.ipynb` on Kaggle (*File -> Import notebook*,
   or the existing track D notebook with this version pasted in).
2. *Add Input*: the notebook output or dataset that holds
   **`gqa_base/final.pt`** (the same one used for the Alpaca chain). Nothing
   else is needed: no weak checkpoint, no SFT checkpoint. The preference pairs
   come from the repo.
3. In the settings cell set:
   ```python
   CHAIN_DATASET = "oasst1"
   BRANCH        = "main"   # already the default
   SMOKE         = False                    # True first if you want a 5-min dry run
   ```
   Leave the other switches alone: `CHAIN_DATASET` turns on SFT, DPO and eval
   and turns off pair building. Optional: `RUN_BENCH = True` does job (b) in
   the same run.
4. *Save Version -> Save & Run All*. The notebook runs exactly:
   ```
   python scripts/sft.py --preset gqa --checkpoint checkpoints/gqa_base/final.pt --dataset oasst1 --run-name gqa_sft_oasst1
   python scripts/dpo.py --sft-checkpoint checkpoints/gqa_sft_oasst1/final.pt --run-name gqa_dpo_oasst1
   python scripts/evaluate.py --base checkpoints/gqa_base/final.pt --sft checkpoints/gqa_sft_oasst1/final.pt --dpo checkpoints/gqa_dpo_oasst1/final.pt --out results/eval_oasst1
   ```
5. Check the log:
   - settings cell prints `SFT -> checkpoints/gqa_sft_oasst1 ... eval -> results/eval_oasst1`
   - SFT: `oasst1: N kept of M` (**write N down**; it is much smaller than
     Alpaca's 45,379 because of the length filters), `masking check passed`,
     and val loss falling
   - DPO: `reference log-probs: computing ...` (not `loaded`), `before: test
     loss 0.6931`, and reward accuracy above 0.5 after training
   - evaluation prints a table for base / sft / dpo
6. From the **Output** tab download `track_d_results.zip`. Keep the notebook
   version: it holds the checkpoints (`checkpoints/gqa_sft_oasst1/`,
   `checkpoints/gqa_dpo_oasst1/`), which are too big for GitHub. Note its name.

**Commit** (unzip at the repo root, then `git status`):

```
results/gqa_sft_oasst1/metrics.jsonl
results/gqa_dpo_oasst1/metrics.jsonl
results/eval_oasst1/metrics.json
results/eval_oasst1/samples.json
results/eval_oasst1/samples.md
```

`git status` must show **only new files**. If `results/gqa_sft/`,
`results/gqa_dpo/`, `results/eval/` or `data/preference/` show as modified,
do not commit them (`git checkout -- <path>`).

## (b) Benchmark re-run (enable_gqa opt-in path)

The T4 run of this path has already been done (decision S14: it was slower,
and `repeat_interleave` is the default again). This section is only needed
to reproduce it; the notebooks set `SLM_SDPA_ENABLE_GQA=1` for it.

Skip this if you set `RUN_BENCH = True` in (a); the file is then in the same zip.

1. Open `notebooks/afml_kaggle_training.ipynb`. No inputs needed (random weights).
2. In the settings cell set:
   ```python
   RUN_DATA = False; RUN_PROBE = False; RUN_TRAINING = False
   RUN_BENCH = True
   BRANCH = "main"
   ```
3. *Save Version -> Save & Run All* (~15 min). Section 7 runs exactly:
   ```
   SLM_SDPA_ENABLE_GQA=1 python scripts/benchmark_attention.py --out results/benchmark_enable_gqa
   ```
4. Check that the cell prints `'sdpa_enable_gqa': True` and a torch version
   >= 2.5. If it is `False`, the numbers measure the default repeat_interleave
   path: do not commit them under this folder.
5. Download `AFML_miniproject/results/benchmark_enable_gqa/benchmark.json`
   from the **Output** tab.

**Commit:** `results/benchmark_enable_gqa/benchmark.json`. Never touch
`results/benchmark/benchmark.json`; it is the default-path half of the comparison.

## (c) Figures and tables

On a laptop, after the files from (a) and (b) are in place:

```powershell
git checkout main
python scripts/make_results.py
python -m pytest -q
```

It picks up `results/eval_oasst1`, `results/gqa_*_oasst1` and
`results/benchmark_enable_gqa` automatically. **Commit** everything it changes
under `results/figures/` and `results/tables/`. New files to expect:
`finetune_oasst1.*`, `benchmark_decode_enable_gqa.*`,
`benchmark_memory_enable_gqa.*`, `tables/eval_oasst1.*`,
`tables/benchmark_enable_gqa.*`, `tables/benchmark_side_by_side_enable_gqa.*`
(labelled "enable_gqa (opt-in) path, T4" next to the 1 Nov repeat_interleave
(default) path).
`main_results`, `main_alignment` and `eval_preference` get updated.

## Afterwards (whoever writes it up)

- `README.md`: copy the updated `results/tables/main_results.md` and
  `main_alignment.md` into "Main results", and drop the two "pending" notes.
- `docs/decisions.md`: under O6 record the oasst1 outcome (examples kept,
  SFT val loss, eval numbers, Kaggle notebook name); under S14 fill in
  the "Result file" line and check the numbers there against
  `tables/benchmark_side_by_side_enable_gqa.md`.
