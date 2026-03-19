"""Tests for EmotionClassifier."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_analyzer_pro.emotions import EmotionClassifier


def test_joy_detection():
    clf = EmotionClassifier()
    result = clf.classify("I am so happy and joyful today! This is wonderful and delightful!")
    assert result.primary in ("joy",), f"Expected joy, got {result.primary}"
    assert result.emotions["joy"].score > 0


def test_anger_detection():
    clf = EmotionClassifier()
    result = clf.classify("I am furious and enraged! This is outrageous and I hate it!")
    assert result.primary in ("anger",), f"Expected anger, got {result.primary}"
    assert result.emotions["anger"].score > 0


def test_fear_detection():
    clf = EmotionClassifier()
    result = clf.classify("I am terrified and anxious. The situation is dangerous and I am afraid.")
    assert result.primary in ("fear",), f"Expected fear, got {result.primary}"
    assert result.emotions["fear"].score > 0


def test_sadness_detection():
    clf = EmotionClassifier()
    result = clf.classify("I am so sad and depressed. This grief and sorrow is overwhelming.")
    assert result.primary in ("sadness",), f"Expected sadness, got {result.primary}"
    assert result.emotions["sadness"].score > 0


def test_trust_detection():
    clf = EmotionClassifier()
    result = clf.classify("I trust this company. They are reliable, honest, and transparent.")
    assert result.primary in ("trust",), f"Expected trust, got {result.primary}"
    assert result.emotions["trust"].score > 0


def test_anticipation_detection():
    clf = EmotionClassifier()
    result = clf.classify("I am so excited and eager! I can't wait and I'm looking forward to it!")
    assert result.primary in ("anticipation", "joy"), f"Expected anticipation or joy, got {result.primary}"


def test_all_eight_emotions_present():
    clf = EmotionClassifier()
    result = clf.classify("test text")
    emotions = set(result.emotions.keys())
    expected = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"}
    assert emotions == expected, f"Missing emotions: {expected - emotions}"


def test_emotion_scores_range():
    clf = EmotionClassifier()
    result = clf.classify("I am extremely happy and excited about this wonderful opportunity!")
    for name, score in result.all_scores.items():
        assert 0.0 <= score <= 1.0, f"Score out of range for {name}: {score}"


def test_neutral_text_no_dominant_emotion():
    clf = EmotionClassifier()
    result = clf.classify("The package arrived on Wednesday.")
    # Either no primary or very low scores
    if result.primary:
        top_score = result.emotions[result.primary].score
        # Should be low since there are no strong emotional triggers
        assert top_score < 0.8, f"Too confident on neutral text: {top_score}"


def test_valence_positive():
    clf = EmotionClassifier()
    result = clf.classify("I love this! It's fantastic and amazing and I'm so happy!")
    assert result.valence in ("positive", "mixed"), f"Expected positive valence, got {result.valence}"


def test_valence_negative():
    clf = EmotionClassifier()
    result = clf.classify("I hate this. Terrible, horrible, disgusting. I'm angry and sad.")
    assert result.valence in ("negative", "mixed"), f"Expected negative valence, got {result.valence}"


def test_keywords_populated():
    clf = EmotionClassifier()
    result = clf.classify("I am happy and joyful")
    if result.primary:
        assert len(result.emotions[result.primary].keywords) > 0


def test_batch():
    clf = EmotionClassifier()
    texts = ["I am so happy!", "I am very angry.", "Whatever."]
    results = clf.classify_batch(texts)
    assert len(results) == 3


if __name__ == "__main__":
    tests = [
        test_joy_detection, test_anger_detection, test_fear_detection,
        test_sadness_detection, test_trust_detection, test_anticipation_detection,
        test_all_eight_emotions_present, test_emotion_scores_range,
        test_neutral_text_no_dominant_emotion, test_valence_positive,
        test_valence_negative, test_keywords_populated, test_batch,
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
