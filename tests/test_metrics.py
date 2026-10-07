import pytest

from app.metrics import accuracy, confusion_matrix, format_matrix, per_class

EXPECTED = ["billing", "billing", "bug", "access"]
PREDICTED = ["billing", "bug", "bug", "access"]


def test_accuracy():
    assert accuracy(EXPECTED, PREDICTED) == 0.75
    assert accuracy([], []) == 0.0
    with pytest.raises(ValueError):
        accuracy(["a"], [])


def test_per_class():
    report = per_class(EXPECTED, PREDICTED)
    assert report["billing"] == {"precision": 1.0, "recall": 0.5, "support": 2}
    assert report["bug"] == {"precision": 0.5, "recall": 1.0, "support": 1}


def test_confusion_matrix_and_format():
    matrix = confusion_matrix(EXPECTED, PREDICTED)
    assert matrix["billing"]["bug"] == 1
    assert matrix["access"]["access"] == 1
    lines = format_matrix(matrix)
    assert len(lines) == 2 + len(matrix)
