"""Load and wrap tokenizer encoding and decoding.

    tok = Tokenizer.load()                      # artifacts/tokenizer/tokenizer.json
    ids = tok.encode("Once upon a time", add_bos=True, add_eos=True)
    text = tok.decode(ids)                      # special tokens dropped by default

Owner: track A.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer as HFTokenizer

DEFAULT_TOKENIZER_PATH = "artifacts/tokenizer/tokenizer.json"

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


class Tokenizer:
    """Thin wrapper over a HF ``tokenizers.Tokenizer`` with fixed special ids."""

    def __init__(self, hf: HFTokenizer):
        self.hf = hf
        # Literal "<eos>" etc. in input text is encoded as ordinary bytes;
        # special ids only ever come from add_bos/add_eos.
        self.hf.encode_special_tokens = True
        ids = {s: hf.token_to_id(s) for s in (PAD, BOS, EOS, UNK)}
        missing = [s for s, i in ids.items() if i is None]
        if missing:
            raise ValueError(f"tokenizer is missing special tokens {missing}")
        self.pad_id: int = ids[PAD]
        self.bos_id: int = ids[BOS]
        self.eos_id: int = ids[EOS]
        self.unk_id: int = ids[UNK]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TOKENIZER_PATH) -> "Tokenizer":
        """Load from a ``tokenizer.json`` file or a directory containing one."""
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"no tokenizer at {path} -- run scripts/train_tokenizer.py")
        return cls(HFTokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self.hf.get_vocab_size(with_added_tokens=True)

    def _wrap(self, ids: list[int], add_bos: bool, add_eos: bool) -> list[int]:
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        return self._wrap(self.hf.encode(text, add_special_tokens=False).ids, add_bos, add_eos)

    def encode_batch(
        self, texts: Sequence[str], add_bos: bool = False, add_eos: bool = False
    ) -> list[list[int]]:
        """Parallel (Rust-side) batch encode."""
        texts = list(texts)
        # encode_batch_fast skips offset tracking; fall back on older versions
        fn = getattr(self.hf, "encode_batch_fast", self.hf.encode_batch)
        return [self._wrap(e.ids, add_bos, add_eos) for e in fn(texts, add_special_tokens=False)]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.hf.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch: Sequence[Sequence[int]], skip_special_tokens: bool = True) -> list[str]:
        return self.hf.decode_batch([list(b) for b in batch], skip_special_tokens=skip_special_tokens)

    def __len__(self) -> int:
        return self.vocab_size
