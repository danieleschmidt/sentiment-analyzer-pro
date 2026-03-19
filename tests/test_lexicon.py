"""Tests for LexiconSentiment scorer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_analyzer_pro.lexicon import LexiconSentiment


def test_positive_text():
    scorer = LexiconSentiment()
    result = scorer.score("This product is absolutely amazing and fantastic!")
    assert result.label == "positive", f"Expected positive, got {result.label}"
    assert result.score > 0.1, f"Expected score > 0.1, got {result.score}"


def test_negative_text():
    scorer = LexiconSentiment()
    result = scorer.score("This is a terrible, horrible, awful product.")
    assert result.label == "negative", f"Expected negative, got {result.label}"
    assert result.score < -0.1, f"Expected score < -0.1, got {result.score}"


def test_neutral_text():
    scorer = LexiconSentiment()
    result = scorer.score("The product shipped on Tuesday in a brown box.")
    assert result.label == "neutral", f"Expected neutral, got {result.label}"


def test_negation():
    scorer = LexiconSentiment()
    pos = scorer.score("This is great")
    neg = scorer.score("This is not great")
    assert pos.score > 0, f"Expected positive, got {pos.score}"
    # Negated version should be less positive (or negative)
    assert neg.score < pos.score, f"Negated ({neg.score}) should be less than positive ({pos.score})"


def test_intensifier_boost():
    scorer = LexiconSentiment()
    base = scorer.score("This is good")
    boosted = scorer.score("This is extremely good")
    assert boosted.score > base.score, f"Intensified ({boosted.score}) should be > base ({base.score})"


def test_score_range():
    scorer = LexiconSentiment()
    for text in [
        "absolutely incredible amazing wonderful fantastic",
        "terrible horrible awful disgusting appalling",
        "the cat sat on the mat",
    ]:
        result = scorer.score(text)
        assert -1.0 <= result.score <= 1.0, f"Score out of range: {result.score}"
        total = result.positive + result.negative + result.neutral
        assert abs(total - 1.0) < 0.01, f"Proportions don't sum to 1: {total}"


def test_empty_text():
    scorer = LexiconSentiment()
    result = scorer.score("")
    assert result.label == "neutral"
    assert result.score == 0.0


def test_batch():
    scorer = LexiconSentiment()
    texts = ["great", "terrible", "neutral"]
    results = scorer.score_batch(texts)
    assert len(results) == 3
    assert results[0].label == "positive"
    assert results[1].label == "negative"


def test_token_scores_populated():
    scorer = LexiconSentiment()
    result = scorer.score("I love this fantastic product")
    assert len(result.token_scores) > 0, "Expected token scores to be populated"
    tokens = [t for t, _ in result.token_scores]
    assert any(t in ("love", "fantastic") for t in tokens)


def test_caps_boost():
    scorer = LexiconSentiment()
    normal = scorer.score("this is great")
    caps = scorer.score("this is GREAT")
    # CAPS version should score higher
    assert caps.score >= normal.score, f"CAPS ({caps.score}) should >= normal ({normal.score})"


if __name__ == "__main__":
    tests = [
        test_positive_text, test_negative_text, test_neutral_text,
        test_negation, test_intensifier_boost, test_score_range,
        test_empty_text, test_batch, test_token_scores_populated,
        test_caps_boost,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
    sys.exit(0 if failed == 0 else 1)
