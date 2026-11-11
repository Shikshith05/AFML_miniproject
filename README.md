# AFML SLM GQA DPO

## Datasets

| Dataset | Used for | Link |
|---|---|---|
| TinyStories | Tokenizer training and pretraining (1% held out as val) | https://huggingface.co/datasets/roneneldan/TinyStories |
| Databricks Dolly-15k | SFT (human-written instructions) | https://huggingface.co/datasets/databricks/databricks-dolly-15k |
| Stanford Alpaca | Earlier SFT run, kept for comparison (synthetic) | https://huggingface.co/datasets/tatsu-lab/alpaca |

DPO preference pairs are built in-domain from held-out TinyStories
(`scripts/build_preference_data.py`), not downloaded.

## Data pipeline

Byte-level BPE tokenizer (vocab 4096) + TinyStories token files. Details and
measurements: [docs/track_a_notes.md](docs/track_a_notes.md).

```powershell
.venv\Scripts\Activate.ps1

# Full pipeline (~13 min): 1% val holdout (seed 1337) -> ftfy text cleaning ->
# tokenizer trained on the train side only -> data/train.bin, data/val.bin, data/meta.json
python scripts/prepare_data.py --train-tokenizer

# Sanity check: decode random val windows, fertility, <unk> rate
python scripts/inspect_data.py

# Tests (no dataset needed)
python -m pytest tests/test_tokenizer.py -q
```

The tokenizer is committed at `artifacts/tokenizer/tokenizer.json`, so
teammates can skip retraining it and just run
`python scripts/prepare_data.py` (it checks the tokenizer was trained on the
same split before using it). The tokenizer can also be trained on its own with
`python scripts/train_tokenizer.py [--max-stories N]`.

Smoke run on 20k stories, kept away from the real outputs:

```powershell
python scripts/prepare_data.py --limit 20000 --train-tokenizer --out-dir data/smoke --tokenizer data/smoke/tokenizer.json
python scripts/inspect_data.py --data-dir data/smoke --tokenizer data/smoke/tokenizer.json
```

Using the data from training code:

```python
from slm.data.tinystories import make_get_batch

get_batch = make_get_batch("data", batch_size=16, seq_len=512, device="cuda")
x, y = get_batch("train")   # int64 [16, 512] on cuda; y is x shifted left by one
```

`make_batch_provider(cfg)` does the same from track C's `Config`.
