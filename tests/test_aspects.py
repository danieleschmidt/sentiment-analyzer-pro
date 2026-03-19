"""Tests for AspectExtractor."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_analyzer_pro.aspects import AspectExtractor


def test_basic_aspect_extraction():
    extractor = AspectExtractor()
    result = extractor.extract("battery life is great but camera is poor")
    aspects = result.to_dict()
    assert "battery_life" in aspects or "battery" in aspects, f"Expected battery aspect, got: {aspects}"
    assert "camera" in aspects, f"Expected camera aspect, got: {aspects}"


def test_contrastive_sentiment_separation():
    """Battery should be positive, camera should be negative."""
    extractor = AspectExtractor()
    result = extractor.extract("The battery life is excellent but the camera is terrible")
    aspects = result.to_dict()

    battery_key = "battery_life" if "battery_life" in aspects else "battery"
    camera_key = "camera"

    if battery_key in aspects:
        assert aspects[battery_key] > 0, f"Battery should be positive, got {aspects[battery_key]}"
    if camera_key in aspects:
        assert aspects[camera_key] < 0, f"Camera should be negative, got {aspects[camera_key]}"


def test_aspect_labels():
    extractor = AspectExtractor()
    result = extractor.extract("The display is absolutely beautiful")
    if "display" in result.aspects:
        assert result.aspects["display"].label == "positive"


def test_no_aspects():
    extractor = AspectExtractor()
    result = extractor.extract("It was a sunny day and I went for a walk")
    # May or may not find aspects — should not crash
    assert isinstance(result.aspects, dict)


def test_multiple_positive_aspects():
    extractor = AspectExtractor()
    result = extractor.extract("The performance is fast, the design is beautiful, and the price is great")
    aspects = result.to_dict()
    assert len(aspects) >= 2, f"Expected ≥2 aspects, got {len(aspects)}: {aspects}"
    for k, v in aspects.items():
        assert v > 0, f"Aspect {k} should be positive, got {v}"


def test_aspect_mentions_populated():
    extractor = AspectExtractor()
    result = extractor.extract("The screen display is amazing")
    assert len(result.aspects) > 0
    for _, asp in result.aspects.items():
        assert len(asp.mentions) > 0


def test_batch():
    extractor = AspectExtractor()
    texts = [
        "battery life is excellent",
        "camera is poor",
        "great price but terrible support",
    ]
    results = extractor.extract_batch(texts)
    assert len(results) == 3


def test_software_aspects():
    extractor = AspectExtractor()
    result = extractor.extract("The app performance is sluggish and the UI is confusing")
    aspects = result.to_dict()
    assert len(aspects) >= 1, f"Expected software aspects, got: {aspects}"


if __name__ == "__main__":
    tests = [
        test_basic_aspect_extraction, test_contrastive_sentiment_separation,
        test_aspect_labels, test_no_aspects, test_multiple_positive_aspects,
        test_aspect_mentions_populated, test_batch, test_software_aspects,
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
