"""Basic text preprocessing utilities."""

import re
from typing import List

import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:  # pragma: no cover - optional dependency
    nltk = None
    stopwords = None
    WordNetLemmatizer = None

STOP_WORDS = {"the", "is", "a", "an", "this", "of", "and", "in", "on", "to"}
_lemmatizer = None

_CLEAN_RE = re.compile(r"[^a-z]+")
_WS_RE = re.compile(r"\s+")


def _ensure_nltk() -> None:
    """Ensure required NLTK data is available."""
    if nltk is None:
        return
    global STOP_WORDS, _lemmatizer
    if not STOP_WORDS or STOP_WORDS == {"the", "is", "a", "an", "this", "of", "and", "in", "on", "to"}:
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:  # pragma: no cover - download if missing
            nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("english"))
    if _lemmatizer is None and WordNetLemmatizer is not None:
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:  # pragma: no cover - download if missing
            nltk.download("wordnet", quiet=True)
        _lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, remove non-alphabetic characters, and trim."""
    text = text.lower()
    text = _CLEAN_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def clean_series(series: pd.Series) -> pd.Series:
    """Vectorized variant of ``clean_text`` for Pandas series."""
    series = series.str.lower()
    series = series.str.replace(_CLEAN_RE, " ", regex=True)
    series = series.str.replace(_WS_RE, " ", regex=True)
    return series.str.strip()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common English stop words from token list."""
    if not STOP_WORDS or STOP_WORDS == {"the", "is", "a", "an", "this", "of", "and", "in", "on", "to"}:
        _ensure_nltk()
    return [t for t in tokens if t not in STOP_WORDS]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Return lemmatized tokens."""
    if _lemmatizer is None:
        _ensure_nltk()

    if _lemmatizer is None:
        # simple fallback without nltk
        def _fallback(word: str) -> str:
            for suffix in ("ing", "ed", "s"):
                if word.endswith(suffix) and len(word) > len(suffix):
                    stem = word[: -len(suffix)]
                    if len(stem) >= 2 and stem[-1] == stem[-2]:
                        stem = stem[:-1]
                    return stem
            return word

        return [_fallback(t) for t in tokens]

    try:
        return [_lemmatizer.lemmatize(t, "v") for t in tokens]
    except LookupError:  # pragma: no cover - missing wordnet
        return tokens
