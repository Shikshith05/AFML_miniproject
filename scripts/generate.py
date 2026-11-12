"""Generate from one or more checkpoints, side by side.

    python scripts/generate.py \
        --ckpt checkpoints/gqa_base/final.pt \
        --ckpt checkpoints/gqa_sft/final.pt \
        --ckpt checkpoints/gqa_dpo/final.pt \
        --prompts-file data/demo_prompts.txt --out results/generations.md

    python scripts/generate.py --ckpt checkpoints/gqa_sft/final.pt \
        --prompt "Write a story about a red ball."

Each model is built from the config saved inside its checkpoint, so MHA, GQA
and tiny test models all load without flags. CPU works (a few seconds per
sample for the 28M model); CUDA is used when available (--device to choose).

Prompt format (--template):
  auto    SFT/DPO checkpoints get the SFT template (format_prompt, the same
          one used in SFT, DPO and evaluation); base checkpoints get the raw
          text, which they simply continue.
  always  template for every model
  never   raw text for every model

Fair comparison: prompt k is sampled with seed (--seed + k) for every model,
the same per-prompt scheme as scripts/evaluate.py. Differences between
models therefore come from the weights, not from the dice.

Owner: track D.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from slm.data.alpaca import format_prompt
from slm.generation.generate import generate_text
from slm.model.config import EvalConfig, ModelConfig
from slm.model.transformer import Transformer
from slm.tokenizer.tokenizer import Tokenizer

FINE_TUNED = ("sft", "dpo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    e = EvalConfig()
    p = argparse.ArgumentParser(description="Side-by-side generation from checkpoints")
    p.add_argument("--ckpt", action="append", required=True,
                   help="checkpoint path; repeat for several models")
    p.add_argument("--prompt", action="append", default=[],
                   help="a prompt; repeatable")
    p.add_argument("--prompts-file",
                   help="one prompt per line; blank lines and lines starting with # are skipped")
    p.add_argument("--template", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--max-new-tokens", type=int, default=e.max_new_tokens)
    p.add_argument("--temperature", type=float, default=e.temperature,
                   help="0 = greedy")
    p.add_argument("--top-k", type=int, default=e.top_k)
    p.add_argument("--seed", type=int, default=e.gen_seed,
                   help="prompt k uses seed + k, identical for every model")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--tokenizer", default=str(REPO / "artifacts" / "tokenizer"))
    p.add_argument("--out", help="write the comparison here as markdown")
    return p.parse_args(argv)


# -- inputs ---------------------------------------------------------------------

def read_prompts(prompts: list[str], prompts_file: str | None) -> list[str]:
    out = [p.strip() for p in prompts if p.strip()]
    if prompts_file:
        for line in Path(prompts_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def checkpoint_stage(payload: dict, path: str | Path) -> str:
    """'base', 'sft' or 'dpo'.

    First from what the training loops store in `extra` (DPO: before/test
    results; SFT: baseline_val_loss in final.pt, epoch in epoch_*.pt), then
    from the folder/file name, else base. Pretraining also stores
    final_val_loss, so that key alone says nothing.
    """
    extra = payload.get("extra") or {}
    if "before" in extra and "test" in extra:
        return "dpo"
    if "baseline_val_loss" in extra or ("epoch" in extra and "val_loss" in extra):
        return "sft"
    name = "/".join(Path(path).parts[-2:]).lower()
    if "dpo" in name:
        return "dpo"
    if "sft" in name:
        return "sft"
    return "base"


def use_template(stage: str, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return stage in FINE_TUNED


def build_prompt(text: str, templated: bool) -> str:
    return format_prompt(text) if templated else text


def label_for(path: str | Path) -> str:
    """checkpoints/gqa_sft_oasst1/final.pt -> gqa_sft_oasst1;
    checkpoints/gqa_base/step_002000.pt -> gqa_base/step_002000."""
    path = Path(path)
    return path.parent.name if path.stem == "final" else f"{path.parent.name}/{path.stem}"


def load_model(path: str | Path, device: str) -> tuple[Transformer, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_cfg = (payload.get("config") or {}).get("model")
    if not model_cfg:
        raise SystemExit(f"{path}: no model config inside the checkpoint")
    known = {f.name for f in fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in model_cfg.items() if k in known})
    model = Transformer(cfg)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    stage = checkpoint_stage(payload, path)
    info = {
        "path": str(path),
        "label": label_for(path),
        "stage": stage,
        "step": payload.get("step"),
        "variant": cfg.attention_variant,
        "sft_dataset": ((payload.get("config") or {}).get("sft") or {}).get("dataset_name")
                       if stage in FINE_TUNED else None,
    }
    return model, info


# -- output ---------------------------------------------------------------------

def to_markdown(prompts: list[str], infos: list[dict], outputs: dict, args) -> str:
    lines = ["# Generations: same prompts, same seed per prompt, every model", "",
             f"temperature {args.temperature}, top-k {args.top_k}, "
             f"max {args.max_new_tokens} new tokens, prompt k uses seed {args.seed} + k, "
             f"template `{args.template}`", "",
             "| model | checkpoint | stage | step | attention | SFT data | prompt format |",
             "|---|---|---|---|---|---|---|"]
    for i in infos:
        fmt = "SFT template" if i["templated"] else "raw text"
        lines.append(f"| {i['label']} | `{i['path']}` | {i['stage']} | {i['step']} | "
                     f"{i['variant']} | {i['sft_dataset'] or '-'} | {fmt} |")
    lines.append("")
    for k, prompt in enumerate(prompts):
        lines += [f"## {k + 1}. {prompt}", ""]
        for i in infos:
            text = outputs[i["label"]][k].strip() or "(empty)"
            lines += [f"**{i['label']}:** {text}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Model output can contain any character (or a stray byte decoded as U+FFFD);
    # a Windows console's cp1252 cannot print all of them. The markdown file is UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    prompts = read_prompts(args.prompt, args.prompts_file)
    if not prompts:
        raise SystemExit("no prompts: give --prompt and/or --prompts-file")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but no GPU is available")

    from slm.data.tinystories import clean_text   # same text cleaning as pretraining

    tok = Tokenizer.load(args.tokenizer)
    infos, outputs, seen = [], {}, set()
    for path in args.ckpt:
        model, info = load_model(path, device)
        if model.cfg.vocab_size != tok.vocab_size:
            raise SystemExit(f"{path}: vocab {model.cfg.vocab_size} != tokenizer {tok.vocab_size}")
        base_label, n = info["label"], 2
        while info["label"] in seen:                 # same folder twice -> keep both
            info["label"], n = f"{base_label} ({n})", n + 1
        seen.add(info["label"])
        info["templated"] = use_template(info["stage"], args.template)
        print(f"== {info['label']}  ({info['stage']}, step {info['step']}, {info['variant']}, "
              f"{'SFT template' if info['templated'] else 'raw text'}, {device})")

        outs = []
        for k, prompt in enumerate(prompts):
            text = build_prompt(clean_text(prompt), info["templated"])
            try:
                out = generate_text(model, tok, text, max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature, top_k=args.top_k,
                                    seed=args.seed + k)
            except ValueError as err:                 # prompt fills the whole context
                out = f"(skipped: {err})"
            outs.append(out)
            print(f"  [{k + 1}] {prompt}\n      -> {out.strip()!r}")
        outputs[info["label"]] = outs
        infos.append(info)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_markdown(prompts, infos, outputs, args), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
