import json
from pathlib import Path

DATA_PATH = Path("data/processed/train_data.jsonl")


def test_record_count():
    records = [json.loads(line) for line in open(DATA_PATH, encoding="utf-8")]
    assert len(records) == 632


def test_record_schema():
    records = [json.loads(line) for line in open(DATA_PATH, encoding="utf-8")]
    for i, r in enumerate(records):
        assert "raw_text" in r, f"record {i} missing raw_text"
        assert "label" in r, f"record {i} missing label"
        assert isinstance(r["raw_text"], str), f"record {i} raw_text is not a string"
        assert r["label"] in (0, 1), \
            f"record {i} has invalid label: {r['label']}"
    empty = sum(1 for r in records if not r["raw_text"].strip())
    if empty:
        print(f"  WARNING: {empty}/{len(records)} records have empty raw_text")


def test_label_distribution():
    records = [json.loads(line) for line in open(DATA_PATH, encoding="utf-8")]
    labels = [r["label"] for r in records]
    assert 0.1 < labels.count(1) / len(labels) < 0.5, \
        "Label distribution is severely imbalanced (>50% mismatch)"
