"""Aspect-based sentiment utilities."""

from typing import Dict, List

try:
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import stopwords

    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("stopwords", quiet=True)

    STOP_WORDS = set(stopwords.words("english"))
except Exception:  # pragma: no cover - optional dependency
    nltk = None
    pos_tag = word_tokenize = None
    STOP_WORDS = set()


def extract_aspects(text: str) -> List[str]:
    """Return a list of noun tokens representing aspects."""
    if word_tokenize is not None:
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
