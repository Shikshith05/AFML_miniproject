"""scripts/generate.py with a tiny random checkpoint (CPU, seconds).

Owner: track D.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from slm.data.alpaca import format_prompt
from slm.model.config import Config, ModelConfig
from slm.model.transformer import Transformer
from slm.training.utils import save_checkpoint

REPO = Path(__file__).resolve().parents[1]


def load_cli():
    spec = importlib.util.spec_from_file_location("generate_cli", REPO / "scripts" / "generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_checkpoint(path: Path, extra: dict | None = None, seed: int = 0) -> Path:
    """A real checkpoint file (same payload as training), tiny model, vocab 4096
    so the committed tokenizer fits."""
    torch.manual_seed(seed)
    cfg = Config(model=ModelConfig(vocab_size=4096, d_model=32, n_layers=1, n_heads=4,
                                   n_kv_heads=2, ffn_hidden=64, max_seq_len=96))
    model = Transformer(cfg.model)
    save_checkpoint(path, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                    scaler=None, step=7, config=cfg.to_dict(), extra=extra)
    return path


@pytest.fixture
def ckpts(tmp_path):
    base = tiny_checkpoint(tmp_path / "gqa_base" / "final.pt", {"final_val_loss": 1.4})
    sft = tiny_checkpoint(tmp_path / "gqa_sft_oasst1" / "final.pt",
                          {"final_val_loss": 2.5, "baseline_val_loss": 4.2})
    return base, sft


def test_cli_writes_side_by_side_markdown(tmp_path, ckpts):
    cli = load_cli()
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("# a comment\n\nOnce upon a time\n  What color is the sky?  \n",
                       encoding="utf-8")
    out = tmp_path / "gen.md"
    assert cli.main(["--ckpt", str(ckpts[0]), "--ckpt", str(ckpts[1]),
                     "--prompts-file", str(prompts), "--prompt", "Say hi.",
                     "--max-new-tokens", "6", "--device", "cpu", "--out", str(out)]) == 0
    md = out.read_text(encoding="utf-8")
    assert "## 1. Say hi." in md and "## 2. Once upon a time" in md
    assert "## 3. What color is the sky?" in md and "a comment" not in md
    assert "| gqa_base |" in md and "raw text" in md
    assert "| gqa_sft_oasst1 |" in md and "SFT template" in md
    assert md.count("**gqa_base:**") == 3 and md.count("**gqa_sft_oasst1:**") == 3


def test_same_seed_per_prompt_for_every_model(tmp_path):
    """Identical weights + identical prompt format -> identical text: the only
    thing that differs between models in a comparison is the model."""
    cli = load_cli()
    a = tiny_checkpoint(tmp_path / "a" / "final.pt")
    b = tiny_checkpoint(tmp_path / "b" / "final.pt")
    out = tmp_path / "gen.md"
    cli.main(["--ckpt", str(a), "--ckpt", str(b), "--prompt", "The dog", "--prompt", "A cat",
              "--max-new-tokens", "8", "--temperature", "1.0", "--device", "cpu",
              "--template", "never", "--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    a_out = [l.split(":**", 1)[1] for l in lines if l.startswith("**a:**")]
    b_out = [l.split(":**", 1)[1] for l in lines if l.startswith("**b:**")]
    assert a_out == b_out and len(a_out) == 2


def test_stage_detection_and_template_choice(tmp_path, ckpts):
    cli = load_cli()
    _, base_info = cli.load_model(ckpts[0], "cpu")
    _, sft_info = cli.load_model(ckpts[1], "cpu")
    assert base_info["stage"] == "base" and sft_info["stage"] == "sft"
    assert cli.checkpoint_stage({"extra": {"before": {}, "test": {}}}, "x/final.pt") == "dpo"
    assert cli.checkpoint_stage({"extra": {"epoch": 0, "val_loss": 2.0}}, "x/e.pt") == "sft"
    assert cli.checkpoint_stage({"extra": {}}, "checkpoints/gqa_dpo_oasst1/final.pt") == "dpo"
    assert cli.checkpoint_stage({"extra": {}}, "checkpoints/gqa_base/step_002000.pt") == "base"
    assert cli.use_template("sft", "auto") and cli.use_template("dpo", "auto")
    assert not cli.use_template("base", "auto")
    assert cli.use_template("base", "always") and not cli.use_template("sft", "never")
    assert cli.build_prompt("Hi", True) == format_prompt("Hi")
    assert cli.build_prompt("Hi", False) == "Hi"


def test_demo_prompts_file_parses():
    cli = load_cli()
    prompts = cli.read_prompts([], REPO / "data" / "demo_prompts.txt")
    assert len(prompts) >= 4 and not any(p.startswith("#") for p in prompts)
