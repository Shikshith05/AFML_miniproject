"""Generate from a fixed prompt set and score the outputs.

Owner: track D.

Every model (base, SFT, DPO) gets the SAME prompts, in the SAME template
(config.PROMPT_TEMPLATE via format_prompt), with the SAME seed, so the
samples are directly comparable side by side. The base model was never
taught the template, so its samples show what SFT added.

Automatic scores are deliberately simple and honest about what they are:
  * mean_tokens      -- response length
  * eos_rate         -- fraction that ended on their own (vs hitting the cap)
  * distinct_2       -- unique word bigrams / all bigrams; low = repetitive
They measure form, not quality. Quality is judged by reading the samples.
"""

from __future__ import annotations

import torch

from slm.data.alpaca import format_prompt
from slm.generation.generate import generate, strip_after_eos

# Kept in code (not data/, which is gitignored) so everyone evaluates the same set.
EVAL_PROMPTS: list[tuple[str, str]] = [
    ("Write a short story about a cat who finds a red hat.", ""),
    ("Write a story about a boy who is afraid of the dark.", ""),
    ("Tell me a story about two friends who share a toy.", ""),
    ("Write a happy story about a dog at the beach.", ""),
    ("Continue the story.", "Once upon a time, there was a little bird who could not fly. Every day"),
    ("Continue the story.", "Tom and Sue went to the park. They saw a big tree with"),
    ("Continue the story.", "Lily had a small box. She did not know what was inside. One day"),
    ("What color is the sky?", ""),
    ("Name three animals that live on a farm.", ""),
    ("Why should we be kind to our friends?", ""),
    ("Give me a word that rhymes with cat.", ""),
    ("What do you do when you feel sad?", ""),
]


def distinct_n(texts: list[str], n: int = 2) -> float:
    grams, total = set(), 0
    for t in texts:
        w = t.split()
        g = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
        grams.update(g)
        total += len(g)
    return len(grams) / total if total else 0.0


@torch.no_grad()
def run_prompts(model, tok, prompts=EVAL_PROMPTS, *, max_new_tokens: int = 128,
                temperature: float = 0.8, top_k: int = 50, seed: int = 42,
                device: str = "cuda") -> dict:
    model.eval().to(device)
    samples, n_eos, n_tok = [], 0, 0
    for k, (instruction, input_text) in enumerate(prompts):
        ids = tok.encode(format_prompt(instruction, input_text), add_bos=True)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        g = torch.Generator(device=device).manual_seed(seed + k)   # per-prompt seed: stable if prompts are added
        out = generate(model, x, max_new_tokens=max_new_tokens, temperature=temperature,
                       top_k=top_k, eos_id=tok.eos_id, generator=g)[0, len(ids):].tolist()
        new = strip_after_eos(out, tok.eos_id)
        n_eos += int(tok.eos_id in out)
        n_tok += len(new)
        samples.append({"instruction": instruction, "input": input_text,
                        "response": tok.decode(new).strip()})
    texts = [s["response"] for s in samples]
    return {
        "samples": samples,
        "scores": {"mean_tokens": n_tok / len(prompts), "eos_rate": n_eos / len(prompts),
                   "distinct_2": distinct_n(texts, 2)},
    }
