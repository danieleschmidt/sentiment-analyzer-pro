"""Aspect-based sentiment utilities."""

from typing import Dict, List

try:
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import stopwords
except Exception:  # pragma: no cover - optional dependency
    nltk = None
    pos_tag = word_tokenize = None
    stopwords = None

STOP_WORDS: set[str] = set()


def _ensure_nltk() -> None:
    """Ensure required NLTK data is downloaded."""
    if nltk is None:
        return
    global STOP_WORDS
    if not STOP_WORDS:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:  # pragma: no cover - download if missing
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:  # pragma: no cover - download if missing
            nltk.download("averaged_perceptron_tagger", quiet=True)
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:  # pragma: no cover - download if missing
            nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("english"))


def extract_aspects(text: str) -> List[str]:
    """Return a list of noun tokens representing aspects."""
    if not STOP_WORDS:
        _ensure_nltk()
    if word_tokenize is not None:
        try:
            tokens = word_tokenize(text)
        except LookupError:
            _ensure_nltk()
            try:
                tokens = word_tokenize(text)
            except LookupError:
                tokens = text.split()
    else:
        tokens = text.split()
    if pos_tag is not None:
        try:
            tagged = pos_tag(tokens)
        except LookupError:
            _ensure_nltk()
            try:
                tagged = pos_tag(tokens)
            except LookupError:
                tagged = [(t, "NN") for t in tokens]
    else:
        tagged = [(t, "NN") for t in tokens]
    aspects = [
        w.lower()
        for w, pos in tagged
        if pos.startswith("NN") and w.lower() not in STOP_WORDS
    ]
    return aspects


def predict_aspect_sentiment(text: str, model) -> Dict[str, str]:
    """Predict sentiment for each extracted aspect using the overall text sentiment."""
    aspects = extract_aspects(text)
    sentiment = model.predict([text])[0]
    return {aspect: sentiment for aspect in aspects}
