"""Benchmark provenance and output-folder guard (no GPU needed).

Owner: tracks B + C.
"""

import datetime
import importlib.util
import json
from pathlib import Path

from slm.model import attention

REPO = Path(__file__).resolve().parents[1]


def load_bench():
    spec = importlib.util.spec_from_file_location(
        "benchmark_attention", REPO / "scripts" / "benchmark_attention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metadata_records_torch_gqa_path_commit_and_date():
    import torch

    meta = load_bench().run_metadata()
    assert meta["torch_version"] == torch.__version__
    assert meta["sdpa_enable_gqa"] is bool(attention._USE_ENABLE_GQA)
    assert meta["kv_path"] == attention.kv_path()
    assert meta["sdpa_has_enable_gqa"] is bool(attention._SDPA_HAS_GQA)
    assert "enable_gqa_env" in meta
    assert datetime.datetime.fromisoformat(meta["date_utc"]).tzinfo is not None
    if (REPO / ".git").exists():
        assert meta["git_commit"] and len(meta["git_commit"]) == 40
        assert isinstance(meta["git_dirty"], bool)
    json.dumps(meta)    # must be JSON-serialisable as it goes into benchmark.json


def test_metadata_records_the_kv_path_actually_used(monkeypatch):
    bench = load_bench()
    monkeypatch.setattr(attention, "_USE_ENABLE_GQA", False)
    meta = bench.run_metadata()
    assert (meta["kv_path"], meta["sdpa_enable_gqa"]) == ("repeat_interleave", False)
    monkeypatch.setattr(attention, "_USE_ENABLE_GQA", True)
    meta = bench.run_metadata()
    assert (meta["kv_path"], meta["sdpa_enable_gqa"]) == ("sdpa_enable_gqa", True)


def test_default_out_follows_kv_path_and_never_the_1_nov_folder():
    out = load_bench().DEFAULT_OUT
    assert out == {"repeat_interleave": "results/benchmark_repeat_interleave",
                   "sdpa_enable_gqa": "results/benchmark_enable_gqa"}
    assert "results/benchmark" not in out.values()


def test_refuses_to_overwrite_existing_benchmark(tmp_path):
    old = tmp_path / "benchmark.json"
    old.write_text('{"old": true}')
    assert load_bench().main(["--out", str(tmp_path)]) == 2
    assert old.read_text() == '{"old": true}'
