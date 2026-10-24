"""Configuration for the AFML SLM project.

Single source of truth for every hyperparameter. All four tracks import from
here, so field names are a shared contract -- change them by PR, not in place.

    cfg = Config.from_yaml("configs/gqa_base.yaml")
    print(cfg.summary())

DRAFT -- field naming is decision O1 in docs/decisions.md and is NOT settled.
This file uses research-code naming (n_layers, d_model, n_heads); the earlier
configs/*.yaml used HuggingFace naming (num_layers, hidden_size,
num_attention_heads). The team picks one in the kickoff session and converts
everything to it.

Note on `n_kv_heads`: this is the NUMBER OF KEY/VALUE HEADS, not the repeat
factor. HuggingFace's Llama code uses `num_key_value_groups` for
n_heads // n_kv_heads, which is a different quantity and reads the same.
The repeat factor here is the derived property `n_rep`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


def round_to_multiple(x: float, multiple: int = 128) -> int:
    """Round up to a multiple -- GPU matmuls prefer aligned dimensions."""
    return int(((x + multiple - 1) // multiple) * multiple)


# --------------------------------------------------------------------------
# Model  (owner: B)
# --------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2          # == n_heads -> MHA, 1 -> MQA, else GQA
    ffn_hidden: int | None = None  # None -> 8/3 * d_model, rounded to 128
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    dropout: float = 0.0          # 0 is right for a single-epoch pretrain

    def __post_init__(self) -> None:
        if self.ffn_hidden is None:
            self.ffn_hidden = round_to_multiple(self.d_model * 8 / 3)

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads}) -- query heads form equal groups"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even -- RoPE rotates pairs"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """How many query heads share each KV head."""
        return self.n_heads // self.n_kv_heads

    @property
    def attention_variant(self) -> str:
        if self.n_kv_heads == self.n_heads:
            return "MHA"
        if self.n_kv_heads == 1:
            return "MQA"
        return f"GQA-{self.n_kv_heads}"

    # -- parameter accounting ------------------------------------------------

    def param_counts(self) -> dict[str, int]:
        d, h, kv, hd = self.d_model, self.n_heads, self.n_kv_heads, self.head_dim
        ff = self.ffn_hidden

        per_layer = {
            "q_proj": d * h * hd,
            "k_proj": d * kv * hd,
            "v_proj": d * kv * hd,
            "o_proj": h * hd * d,
            "w_gate": d * ff,
            "w_up": d * ff,
            "w_down": ff * d,
            "norms": 2 * d,
        }
        layer_total = sum(per_layer.values())

        counts = {f"layer.{k}": v for k, v in per_layer.items()}
        counts["per_layer_total"] = layer_total
        counts["all_layers"] = layer_total * self.n_layers
        counts["embedding"] = self.vocab_size * d
        counts["final_norm"] = d
        counts["lm_head"] = 0 if self.tie_embeddings else d * self.vocab_size
        counts["total"] = (
            counts["all_layers"]
            + counts["embedding"]
            + counts["final_norm"]
            + counts["lm_head"]
        )
        counts["non_embedding"] = counts["all_layers"] + counts["final_norm"]
        return counts

    def kv_cache_bytes(
        self, seq_len: int | None = None, batch_size: int = 1, dtype_bytes: int = 2
    ) -> int:
        """KV cache size during generation. The quantity GQA reduces."""
        seq_len = seq_len or self.max_seq_len
        return (
            2  # K and V
            * self.n_layers
            * self.n_kv_heads
            * self.head_dim
            * seq_len
            * batch_size
            * dtype_bytes
        )

    def param_count_str(self) -> str:
        c = self.param_counts()
        return (
            f"{self.attention_variant}: {c['total'] / 1e6:.2f}M total "
            f"({c['non_embedding'] / 1e6:.2f}M non-embedding, "
            f"{c['embedding'] / 1e6:.2f}M embedding = "
            f"{100 * c['embedding'] / c['total']:.1f}%)"
        )


# --------------------------------------------------------------------------
# Data  (owner: A)
# --------------------------------------------------------------------------

@dataclass
class DataConfig:
    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_path: str = "artifacts/tokenizer"
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    val_fraction: float = 0.01
    max_train_tokens: int | None = None  # None -> whole corpus; set to cap run time


# --------------------------------------------------------------------------
# Training  (owner: C)
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # batch
    micro_batch_size: int = 16
    grad_accum_steps: int = 8      # effective batch = 16 * 8 = 128

    # schedule
    max_steps: int = 20_000
    warmup_ratio: float = 0.02
    lr: float = 3e-4
    min_lr_ratio: float = 0.1      # cosine floor = lr * this

    # optimizer
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95            # 0.95 not 0.999 -- see CONCEPTS.md 4.3
    grad_clip: float = 1.0

    # precision -- "fp16" on T4 (Turing), "bf16" needs Ampere+
    precision: str = "fp16"

    # cadence
    eval_interval: int = 500
    checkpoint_interval: int = 1_000
    log_interval: int = 20

    # reproducibility -- MUST match across MHA and GQA runs
    seed: int = 1337
    deterministic_data_order: bool = True

    device: str = "cuda"
    compile_model: bool = False    # torch.compile; leave off until it works

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"unknown precision: {self.precision}")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps

    @property
    def warmup_steps(self) -> int:
        return max(1, int(self.max_steps * self.warmup_ratio))

    def tokens_per_step(self, seq_len: int) -> int:
        return self.effective_batch_size * seq_len


# --------------------------------------------------------------------------
# Evaluation  (owner: D)
# --------------------------------------------------------------------------

@dataclass
class EvalConfig:
    eval_batches: int = 50
    prompts_path: str = "data/eval_prompts.txt"
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    gen_seed: int = 42


# --------------------------------------------------------------------------
# Fine-tuning stages  (owners: D for SFT, C+D for DPO)
# --------------------------------------------------------------------------

# The prompt template. Byte-identical at SFT, DPO and eval time.
# Do not reformat. Do not inline a copy anywhere else.
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


@dataclass
class SFTConfig:
    dataset_name: str = "tatsu-lab/alpaca"
    max_instruction_tokens: int = 64     # filter: keep it in-distribution
    max_response_tokens: int = 192
    lr: float = 2e-5
    epochs: int = 2
    micro_batch_size: int = 8
    grad_accum_steps: int = 4
    warmup_ratio: float = 0.03


@dataclass
class DPOConfig:
    beta: float = 0.1                    # KL penalty strength
    lr: float = 5e-6                     # lower than SFT
    epochs: int = 1
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    n_pairs: int = 3_000
    rejected_from_checkpoint: str = ""    # weak checkpoint used for rejections
    ref_logprobs_cache: str = "data/ref_logprobs.npz"


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

@dataclass
class Config:
    run_name: str = "debug"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):   # from_yaml path
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.train, dict):
            self.train = TrainConfig(**self.train)
        if isinstance(self.eval, dict):
            self.eval = EvalConfig(**self.eval)
        if isinstance(self.sft, dict):
            self.sft = SFTConfig(**self.sft)
        if isinstance(self.dpo, dict):
            self.dpo = DPOConfig(**self.dpo)

    # -- paths ---------------------------------------------------------------

    @property
    def checkpoint_dir(self) -> Path:
        return Path("checkpoints") / self.run_name

    @property
    def results_dir(self) -> Path:
        return Path("results") / self.run_name

    @property
    def metrics_path(self) -> Path:
        return self.results_dir / "metrics.jsonl"

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls(**yaml.safe_load(Path(path).read_text()))

    # -- reporting -----------------------------------------------------------

    def summary(self) -> str:
        m, t = self.model, self.train
        lines = [
            f"run: {self.run_name}",
            f"  {m.param_count_str()}",
            f"  {m.n_layers}L x {m.d_model}d, {m.n_heads}Q/{m.n_kv_heads}KV heads "
            f"(head_dim {m.head_dim}, n_rep {m.n_rep})",
            f"  ffn_hidden {m.ffn_hidden}, seq {m.max_seq_len}, vocab {m.vocab_size}",
            f"  batch {t.micro_batch_size}x{t.grad_accum_steps}"
            f"={t.effective_batch_size}, "
            f"{t.tokens_per_step(m.max_seq_len):,} tok/step",
            f"  {t.max_steps:,} steps = "
            f"{t.max_steps * t.tokens_per_step(m.max_seq_len) / 1e9:.2f}B tokens",
            f"  lr {t.lr} (warmup {t.warmup_steps}), seed {t.seed}, {t.precision}",
            f"  KV cache @ seq {m.max_seq_len} batch 1: "
            f"{m.kv_cache_bytes() / 1e6:.1f} MB",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Presets -- the two runs the ablation needs.
# Identical in every respect except n_kv_heads.
# --------------------------------------------------------------------------

def mha_baseline() -> Config:
    cfg = Config(run_name="mha_baseline")
    cfg.model.n_kv_heads = cfg.model.n_heads   # 8 -> standard MHA
    return cfg


def gqa_base() -> Config:
    cfg = Config(run_name="gqa_base")
    cfg.model.n_kv_heads = 2
    return cfg


if __name__ == "__main__":
    for make in (mha_baseline, gqa_base):
        cfg = make()
        print(cfg.summary())
        print()

    mha, gqa = mha_baseline().model, gqa_base().model
    print("KV cache comparison (fp16):")
    print(f"{'seq':>6} {'batch':>6} {'MHA':>10} {'GQA':>10} {'saved':>10}")
    for seq in (512, 1024, 2048):
        for bs in (1, 16, 64):
            a = mha.kv_cache_bytes(seq, bs)
            b = gqa.kv_cache_bytes(seq, bs)
            print(
                f"{seq:>6} {bs:>6} {a / 1e6:>9.1f}M {b / 1e6:>9.1f}M "
                f"{(a - b) / 1e6:>9.1f}M"
            )
