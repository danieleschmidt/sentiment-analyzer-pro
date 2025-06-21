import pytest

from src import preprocessing


def test_clean_text():
    text = "Hello!!!"
    assert preprocessing.clean_text(text) == "hello"


def test_remove_stopwords():
    tokens = ["this", "is", "a", "test"]
    assert preprocessing.remove_stopwords(tokens) == ["test"]
