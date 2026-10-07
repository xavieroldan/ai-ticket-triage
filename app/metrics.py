"""Evaluation metrics for labelled predictions, without external dependencies."""

from collections import Counter
from collections.abc import Sequence


def accuracy(expected: Sequence[str], predicted: Sequence[str]) -> float:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    if not expected:
        return 0.0
    return round(sum(e == p for e, p in zip(expected, predicted)) / len(expected), 4)


def per_class(expected: Sequence[str], predicted: Sequence[str]) -> dict[str, dict]:
    """Precision, recall and support for every label present in either list."""
    labels = sorted(set(expected) | set(predicted))
    pairs = list(zip(expected, predicted))
    report = {}
    for label in labels:
        tp = sum(e == label and p == label for e, p in pairs)
        fp = sum(e != label and p == label for e, p in pairs)
        fn = sum(e == label and p != label for e, p in pairs)
        report[label] = {
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "support": tp + fn,
        }
    return report


def confusion_matrix(expected: Sequence[str], predicted: Sequence[str]) -> dict[str, dict[str, int]]:
    """Rows are expected labels, columns are predicted labels."""
    labels = sorted(set(expected) | set(predicted))
    counts = Counter(zip(expected, predicted))
    return {e: {p: counts.get((e, p), 0) for p in labels} for e in labels}


def format_matrix(matrix: dict[str, dict[str, int]]) -> list[str]:
    """Render a confusion matrix as aligned text lines."""
    labels = list(matrix)
    width = max([len(label) for label in labels] + [8])
    header = " " * width + " | " + " ".join(label[:width].rjust(width) for label in labels)
    lines = [header, "-" * len(header)]
    for e in labels:
        lines.append(e.ljust(width) + " | " + " ".join(str(matrix[e][p]).rjust(width) for p in labels))
    return lines
