"""Evaluate the classifier against a labelled CSV.

Usage: python -m eval.evaluate [path/to/dataset.csv]
Requires the model dependencies: pip install -r requirements-model.txt
"""

import csv
import os
import sys
from pathlib import Path

from app.classifier import DEFAULT_MODEL, TicketClassifier, load_hf_pipeline
from app.config import load_config
from app.metrics import accuracy, confusion_matrix, format_matrix, per_class

DEFAULT_DATASET = Path(__file__).resolve().parent / "dataset.csv"


def load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def report(name: str, expected: list[str], predicted: list[str]) -> None:
    print(f"\n## {name}: accuracy {accuracy(expected, predicted):.2%}\n")
    for label, stats in per_class(expected, predicted).items():
        print(
            f"  {label:<16} precision {stats['precision']:.2f}  recall {stats['recall']:.2f}  n={stats['support']}"
        )
    print()
    for line in format_matrix(confusion_matrix(expected, predicted)):
        print("  " + line)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    rows = load_dataset(path)
    model = os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL)
    classifier = TicketClassifier(load_config(), lambda: load_hf_pipeline(model))

    predictions = [classifier.classify(row["text"]) for row in rows]
    review = sum(p.needs_review for p in predictions)

    print(f"# Evaluation of {model} on {len(rows)} tickets ({path.name})")
    print(f"Flagged for human review: {review} ({review / len(rows):.0%})")
    report("Category", [r["category"] for r in rows], [p.category for p in predictions])
    report("Urgency", [r["urgency"] for r in rows], [p.urgency for p in predictions])


if __name__ == "__main__":
    main()
