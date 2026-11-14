"""Test SFT dataset field mapping (Alpaca / Dolly / oasst1) without downloading.

Owner: track D.
"""

from pathlib import Path

import datasets
import pytest

from slm.data.alpaca import iter_instruction_rows

REPO = Path(__file__).resolve().parents[1]


def fake(rows):
    return lambda name, split: rows


def test_alpaca_fields(monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset",
                        fake([{"instruction": "Say hi", "input": "", "output": "Hi!"}]))
    assert list(iter_instruction_rows("alpaca")) == [("Say hi", "", "Hi!")]


def test_dolly_fields(monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset", fake([
        {"instruction": "Summarise", "context": "Cats sleep.", "response": "Cats nap.",
         "category": "summarization"}]))
    assert list(iter_instruction_rows("dolly")) == [("Summarise", "Cats sleep.", "Cats nap.")]


def test_oasst1_takes_first_turn_top_ranked_english(monkeypatch):
    rows = [
        {"message_id": "p1", "parent_id": None, "role": "prompter", "lang": "en", "rank": None, "text": "Tell a story"},
        {"message_id": "a1", "parent_id": "p1", "role": "assistant", "lang": "en", "rank": 0, "text": "Once..."},
        {"message_id": "a2", "parent_id": "p1", "role": "assistant", "lang": "en", "rank": 1, "text": "Worse"},
        {"message_id": "p2", "parent_id": "a1", "role": "prompter", "lang": "en", "rank": None, "text": "More"},
        {"message_id": "a3", "parent_id": "p2", "role": "assistant", "lang": "en", "rank": 0, "text": "2nd turn"},
        {"message_id": "p3", "parent_id": None, "role": "prompter", "lang": "de", "rank": None, "text": "Hallo"},
        {"message_id": "a4", "parent_id": "p3", "role": "assistant", "lang": "de", "rank": 0, "text": "Hi"},
    ]
    monkeypatch.setattr(datasets, "load_dataset", fake(rows))
    assert list(iter_instruction_rows("oasst1")) == [("Tell a story", "", "Once...")]


def test_limit(monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset", fake(
        [{"instruction": str(i), "input": "", "output": "x"} for i in range(10)]))
    assert len(list(iter_instruction_rows("alpaca", limit=3))) == 3


# -- full column layouts ---------------------------------------------------------
# The rows below carry every column the real HF datasets have (checked against
# the dataset cards), and go through datasets.Dataset.from_list so iteration,
# null handling and dtypes behave like the real download. Nothing is fetched.

def oasst1_row(message_id, parent_id, role, text, lang="en", rank=None, deleted=False,
               tree="t1"):
    return {
        "message_id": message_id, "parent_id": parent_id, "user_id": "u-" + message_id,
        "created_date": "2023-02-05T00:00:00+00:00", "text": text, "role": role,
        "lang": lang, "review_count": 3, "review_result": True, "deleted": deleted,
        "rank": rank, "synthetic": False, "model_name": None,
        "detoxify": {"toxicity": 0.001, "severe_toxicity": 0.0, "obscene": 0.0,
                     "identity_attack": 0.0, "insult": 0.0, "threat": 0.0,
                     "sexual_explicit": 0.0},
        "message_tree_id": tree, "tree_state": "ready_for_export",
        "emojis": {"name": ["+1"], "count": [1]},
        "labels": {"name": ["spam"], "value": [0.0], "count": [3]},
    }


OASST1_ROWS = [
    # tree 1: English root, three ranked replies, a second turn below the best one
    oasst1_row("p1", None, "prompter", "Tell me a story about a cat."),
    oasst1_row("a1b", "p1", "assistant", "Second best reply.", rank=1),
    oasst1_row("a1", "p1", "assistant", "Once there was a cat.", rank=0),
    oasst1_row("a1c", "p1", "assistant", "Third reply.", rank=2),
    oasst1_row("p1b", "a1", "prompter", "What happened next?"),
    oasst1_row("a1d", "p1b", "assistant", "Second-turn reply, must be skipped.", rank=0),
    # tree 2: English root whose only reply is unranked -> no pair
    oasst1_row("p2", None, "prompter", "Unranked question", tree="t2"),
    oasst1_row("a2", "p2", "assistant", "Unranked reply", rank=None, tree="t2"),
    # tree 3: Spanish root -> skipped even though its reply has rank 0
    oasst1_row("p3", None, "prompter", "Cuentame un cuento", lang="es", tree="t3"),
    oasst1_row("a3", "p3", "assistant", "Habia una vez", lang="es", rank=0, tree="t3"),
    # tree 4: rank-0 reply that was deleted -> skipped
    oasst1_row("p4", None, "prompter", "Deleted answer question", tree="t4"),
    oasst1_row("a4", "p4", "assistant", "Deleted reply", rank=0, deleted=True, tree="t4"),
    # tree 5: second English root, reply listed BEFORE its root in the file
    oasst1_row("a5", "p5", "assistant", "A dog ran.", rank=0, tree="t5"),
    oasst1_row("p5", None, "prompter", "Write one sentence about a dog.", tree="t5"),
]

DOLLY_ROWS = [
    {"instruction": "Who is the main character?", "context": "Tom the cat lived in a barn.",
     "response": "Tom the cat.", "category": "closed_qa"},
    {"instruction": "Name a farm animal.", "context": "", "response": "A cow.",
     "category": "open_qa"},
]


def fake_hf(rows, calls):
    def load_dataset(name, split):
        calls.append((name, split))
        return datasets.Dataset.from_list(rows)
    return load_dataset


def test_oasst1_real_layout_first_turn_rank0_english_only(monkeypatch):
    calls = []
    monkeypatch.setattr(datasets, "load_dataset", fake_hf(OASST1_ROWS, calls))
    rows = list(iter_instruction_rows("oasst1"))
    assert calls == [("OpenAssistant/oasst1", "train")]
    assert rows == [
        ("Tell me a story about a cat.", "", "Once there was a cat."),
        ("Write one sentence about a dog.", "", "A dog ran."),
    ]


def test_dolly_real_layout_maps_context_to_input(monkeypatch):
    calls = []
    monkeypatch.setattr(datasets, "load_dataset", fake_hf(DOLLY_ROWS, calls))
    rows = list(iter_instruction_rows("dolly"))
    assert calls == [("databricks/databricks-dolly-15k", "train")]
    assert rows == [
        ("Who is the main character?", "Tom the cat lived in a barn.", "Tom the cat."),
        ("Name a farm animal.", "", "A cow."),
    ]


@pytest.mark.parametrize("dataset, rows, expected_prompts", [
    ("oasst1", OASST1_ROWS, [("Tell me a story about a cat.", ""),
                             ("Write one sentence about a dog.", "")]),
    ("dolly", DOLLY_ROWS, [("Who is the main character?", "Tom the cat lived in a barn."),
                           ("Name a farm animal.", "")]),
])
def test_load_alpaca_builds_masked_examples_in_the_sft_template(monkeypatch, dataset, rows,
                                                                 expected_prompts):
    """End to end through the SFT data path: rows -> template -> masked Example."""
    from slm.data.alpaca import IGNORE_INDEX, format_prompt, load_alpaca
    from slm.model.config import gqa_base
    from slm.tokenizer.tokenizer import Tokenizer

    monkeypatch.setattr(datasets, "load_dataset", fake_hf(rows, []))
    cfg = gqa_base()
    cfg.sft.dataset_name = dataset
    tok = Tokenizer.load(REPO / cfg.data.tokenizer_path)

    examples = load_alpaca(cfg, tok)
    assert len(examples) == len(expected_prompts)
    for ex, (instruction, input_text) in zip(examples, expected_prompts):
        n_prompt = ex.labels.index(next(l for l in ex.labels if l != IGNORE_INDEX))
        assert all(l == IGNORE_INDEX for l in ex.labels[:n_prompt])
        assert tok.decode(ex.input_ids[:n_prompt]) == format_prompt(instruction, input_text)
        assert ex.input_ids[-1] == tok.eos_id
