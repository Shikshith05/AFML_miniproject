"""scripts/make_results.py on a tiny synthetic results/ tree (seconds, no GPU).

Checks that new folders from a re-run (eval_oasst1, benchmark_enable_gqa,
gqa_sft_oasst1) are picked up next to the originals.
"""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("make_results", REPO / "scripts" / "make_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def bench(gqa_speed: float, **meta):
    inf = [{"variant": v, "batch_size": b, "context_len": c, "kv_cache_bytes_measured": 1e6 * (4 if v == "mha" else 1),
            "decode_tokens_per_s": 100.0 * b * (gqa_speed if v == "gqa" else 1.0), "peak_gb": 0.5}
           for v in ("mha", "gqa") for b in (1, 64) for c in (128, 2048)]
    s = {"tokens_per_s_mean": 1000.0, "tokens_per_s_std": 10.0}
    return {"gpu": "Tesla T4", "precision": "fp16", "inference": inf, **meta,
            "training": {"summary": {"mha": s, "gqa": s}, "gqa_vs_mha_pct": 0.0, "spread_pct": 1.0}}


def test_picks_up_rerun_folders(tmp_path):
    r = tmp_path / "results"
    cfg = {"model": {"n_layers": 8, "n_heads": 8, "n_kv_heads": 2, "d_model": 512, "max_seq_len": 512,
                     "ffn_hidden": 1664},
           "train": {"micro_batch_size": 85, "grad_accum_steps": 1, "max_steps": 12000}}
    for v in ("mha", "gqa"):
        write_jsonl(r / f"{v}_base" / "metrics.jsonl", [
            {"event": "start", "step": 0, "config": cfg},
            {"event": "train", "step": 0, "loss": 8.3},
            {"event": "eval", "step": 500, "val_loss": 2.2, "val_ppl": 9.0},
            {"event": "done", "step": 12000, "val_loss": 1.35, "val_ppl": 3.87, "wall_s": 3600}])
    for sfx in ("", "_oasst1"):
        write_jsonl(r / f"gqa_sft{sfx}" / "metrics.jsonl", [
            {"event": "train", "step": 0, "loss": 4.0}, {"event": "eval", "step": 0, "val_loss": 4.2}])
        write_jsonl(r / f"gqa_dpo{sfx}" / "metrics.jsonl", [
            {"event": "train", "step": 0, "reward_acc": 0.5}, {"event": "eval", "step": 0, "reward_acc": 0.0}])
        m = {k: {"perplexity": {"ppl": 4.0}, "preference": {"acc_sum": 0.7, "acc_mean": 0.9, "n_pairs": 200},
                 "generation": {"eos_rate": 0.5, "distinct_2": 0.8, "mean_tokens": 90.0}} for k in ("base", "sft", "dpo")}
        (r / f"eval{sfx}").mkdir(parents=True)
        (r / f"eval{sfx}" / "metrics.json").write_text(json.dumps(m))
    (r / "benchmark").mkdir()
    (r / "benchmark" / "benchmark.json").write_text(json.dumps(bench(0.5)))
    (r / "benchmark_enable_gqa").mkdir()
    (r / "benchmark_enable_gqa" / "benchmark.json").write_text(json.dumps(bench(
        1.2, sdpa_enable_gqa=True, torch_version="2.6.0", git_commit="a" * 40, date_utc="2026-11-13T00:00:00+00:00")))

    assert load().main(["--results", str(r), "--benchmark-note", "OLD PATH NOTE"]) == 0
    figs = {p.name for p in (r / "figures").iterdir()}
    tables = {p.name for p in (r / "tables").iterdir()}
    for name in ("pretrain_loss", "benchmark_decode", "benchmark_decode_enable_gqa",
                 "finetune", "finetune_oasst1", "eval_preference"):
        assert f"{name}.png" in figs and f"{name}.pdf" in figs
    for name in ("main_results", "main_alignment", "pretrain", "benchmark", "benchmark_enable_gqa",
                 "benchmark_before_after_enable_gqa", "eval", "eval_oasst1"):
        assert f"{name}.md" in tables and f"{name}.tex" in tables
    assert "OLD PATH NOTE" in (r / "tables" / "benchmark.md").read_text(encoding="utf-8")
    after = (r / "tables" / "benchmark_enable_gqa.md").read_text(encoding="utf-8")
    assert "SDPA enable_gqa" in after and "OLD PATH NOTE" not in after
    assert "oasst1" in (r / "tables" / "main_alignment.md").read_text(encoding="utf-8")
