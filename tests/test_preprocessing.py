
from src import preprocessing


def test_clean_text():
    text = "Hello!!!"
    assert preprocessing.clean_text(text) == "hello"


def test_clean_text_collapses_spaces():
    text = "Hello   world"
    assert preprocessing.clean_text(text) == "hello world"


def test_clean_text_lemmatizes_running():
    text = "running"
    clean = preprocessing.clean_text(text)
    assert preprocessing.lemmatize_tokens(clean.split()) == ["run"]


def test_remove_stopwords():
    tokens = ["this", "is", "a", "test"]
    assert preprocessing.remove_stopwords(tokens) == ["test"]
