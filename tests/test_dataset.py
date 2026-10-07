import csv
from pathlib import Path

from app.config import load_config

DATASET = Path(__file__).resolve().parent.parent / "eval" / "dataset.csv"


def test_dataset_labels_match_labels_file():
    config = load_config()
    with DATASET.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 30
    assert {r["category"] for r in rows} <= set(config.categories)
    assert {r["urgency"] for r in rows} <= set(config.urgency)
    assert all(r["text"].strip() for r in rows)
