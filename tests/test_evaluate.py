import pytest

from src.evaluate import analyze_errors, compute_confusion, evaluate


def test_evaluate_returns_report():
    pytest.importorskip("sklearn")
    report = evaluate(["pos", "neg"], ["pos", "neg"])
    assert "precision" in report


def test_compute_confusion_matrix_shape():
    pytest.importorskip("sklearn")
    matrix = compute_confusion(["pos", "neg"], ["pos", "neg"])
    assert matrix == [[1, 0], [0, 1]]


def test_analyze_errors_finds_mismatches():
    pytest.importorskip("pandas")
    errors = analyze_errors(["a", "b"], ["pos", "neg"], ["pos", "pos"])
    assert len(errors) == 1
    assert errors.iloc[0]["text"] == "b"
