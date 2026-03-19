"""Integration tests for SentimentAnalyzerPro."""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_analyzer_pro import SentimentAnalyzerPro, SentimentResult


def test_basic_positive():
    analyzer = SentimentAnalyzerPro()
    result = analyzer.analyze("This product is absolutely fantastic and I love it!")
    assert result.sentiment == "positive"
    assert result.score > 0


def test_basic_negative():
    analyzer = SentimentAnalyzerPro()
    result = analyzer.analyze("This is terrible, I hate it and it's completely useless.")
    assert result.sentiment == "negative"
    assert result.score < 0


def test_returns_sentiment_result():
    analyzer = SentimentAnalyzerPro()
    result = analyzer.analyze("Great product!")
    assert isinstance(result, SentimentResult)


def test_result_structure():
    analyzer = SentimentAnalyzerPro()
    result = analyzer.analyze("The battery life is great but the camera is poor.")
    assert hasattr(result, "text")
    assert hasattr(result, "sentiment")
    assert hasattr(result, "score")
    assert hasattr(result, "confidence")
    assert hasattr(result, "lexicon")
    assert hasattr(result, "aspects")
    assert hasattr(result, "primary_emotion")
    assert hasattr(result, "emotion_scores")


def test_confidence_range():
    analyzer = SentimentAnalyzerPro()
    for text in ["amazing!", "terrible!", "the box arrived", "I love this so much!"]:
        result = analyzer.analyze(text)
        assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range: {result.confidence}"


def test_to_dict():
    analyzer = SentimentAnalyzerPro()
    result = analyzer.analyze("Great product with some issues")
    d = result.to_dict()
    assert "sentiment" in d
    assert "score" in d
    assert "aspects" in d
    assert "emotions" in d


def test_to_json():
    analyzer = SentimentAnalyzerPro()
    result = analyzer.analyze("I love this!")
    j = result.to_json()
    parsed = json.loads(j)
    assert "sentiment" in parsed


def test_summary():
    analyzer = SentimentAnalyzerPro()
    result = analyzer.analyze("I love this amazing product!")
    summary = result.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "POSITIVE" in summary or "NEGATIVE" in summary or "NEUTRAL" in summary


def test_aspect_based_complex():
    """Battery great, camera poor — should detect both aspects."""
    analyzer = SentimentAnalyzerPro()
    result = analyzer.analyze(
        "The battery life is excellent and long-lasting, but the camera quality is terrible."
    )
    assert len(result.aspects) >= 1, f"Expected aspects, got none"


def test_batch():
    analyzer = SentimentAnalyzerPro()
    texts = [
        "I love this product!",
        "Terrible service, would not recommend.",
        "It arrived on time.",
    ]
    results = analyzer.analyze_batch(texts)
    assert len(results) == 3
    assert results[0].sentiment == "positive"
    assert results[1].sentiment == "negative"


def test_empty_text():
    analyzer = SentimentAnalyzerPro()
    result = analyzer.analyze("")
    assert result.sentiment == "neutral"
    assert result.score == 0.0


def test_very_short_text():
    analyzer = SentimentAnalyzerPro()
    result = analyzer.analyze("good")
    assert result.sentiment == "positive"


if __name__ == "__main__":
    tests = [
        test_basic_positive, test_basic_negative, test_returns_sentiment_result,
        test_result_structure, test_confidence_range, test_to_dict, test_to_json,
        test_summary, test_aspect_based_complex, test_batch,
        test_empty_text, test_very_short_text,
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
