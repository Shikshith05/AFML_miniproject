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
    assert meta["sdpa_enable_gqa"] is bool(attention._SDPA_HAS_GQA)
    assert datetime.datetime.fromisoformat(meta["date_utc"]).tzinfo is not None
    if (REPO / ".git").exists():
        assert meta["git_commit"] and len(meta["git_commit"]) == 40
        assert isinstance(meta["git_dirty"], bool)
    json.dumps(meta)    # must be JSON-serialisable as it goes into benchmark.json


def test_default_out_is_a_new_folder():
    import inspect

    src = inspect.getsource(load_bench().main)
    assert 'default="results/benchmark_enable_gqa"' in src


def test_refuses_to_overwrite_existing_benchmark(tmp_path):
    old = tmp_path / "benchmark.json"
    old.write_text('{"old": true}')
    assert load_bench().main(["--out", str(tmp_path)]) == 2
    assert old.read_text() == '{"old": true}'
