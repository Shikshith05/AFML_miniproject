"""Test SFT dataset field mapping (Alpaca / Dolly / oasst1) without downloading.

Owner: track D.
"""

import datasets

from slm.data.alpaca import iter_instruction_rows


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
