# sentiment-analyzer-pro

Multi-level sentiment analysis that goes beyond positive/negative — detects aspects and emotions too.

**No ML frameworks required.** Runs on stdlib + optional numpy. Fast, interpretable, zero-dependency by default.

## Features

- **Lexicon scoring** — VADER-style compound sentiment score with negation, intensifiers, and capitalization handling
- **Aspect extraction** — identifies product/service aspects (battery, camera, price, etc.) and scores each independently
- **Emotion detection** — maps text to 8 basic Plutchik emotions: joy, sadness, anger, fear, surprise, disgust, trust, anticipation
- **CLI** — analyze single texts or batch-process files
- **Structured output** — clean dataclass results with JSON serialization

## Quick Start

```bash
# Install
pip install -e .

# Analyze a single text
sentiment-pro analyze "The battery life is excellent but the camera is disappointing."

# With aspects and emotions
sentiment-pro analyze "I'm so angry about this terrible service!" --aspects --emotions

# JSON output
sentiment-pro analyze "Great product!" --json

# Batch mode (one text per line)
sentiment-pro batch reviews.txt
sentiment-pro batch reviews.txt --json --out results.jsonl
```

## Python API

```python
from sentiment_analyzer_pro import SentimentAnalyzerPro

analyzer = SentimentAnalyzerPro()
result = analyzer.analyze("The battery life is excellent but the camera is disappointing.")

print(result.sentiment)      # "positive"
print(result.score)          # +0.21  (compound, -1 to +1)
print(result.aspects)        # {'battery_life': 0.71, 'camera': -0.52}
print(result.primary_emotion) # "anticipation" or "joy"
print(result.summary())
# [POSITIVE] score=+0.210 | emotion=joy | aspects=[battery_life=↑0.71, camera=↓0.52]

# JSON-serializable
print(result.to_json())
```

### Individual Components

```python
from sentiment_analyzer_pro import LexiconSentiment, AspectExtractor, EmotionClassifier

# Lexicon scoring
scorer = LexiconSentiment()
r = scorer.score("This is absolutely amazing!")
print(r.score, r.label, r.positive, r.negative)

# Aspect extraction
extractor = AspectExtractor()
r = extractor.extract("Battery life is great but camera is poor")
print(r.to_dict())  # {'battery_life': 0.7, 'camera': -0.65}

# Emotion classification
clf = EmotionClassifier()
r = clf.classify("I'm terrified and anxious about the results")
print(r.primary)   # 'fear'
print(r.top(3))    # top 3 emotions by score
```

## Architecture

```
src/sentiment_analyzer_pro/
├── __init__.py       # Public API
├── lexicon.py        # LexiconSentiment — VADER-style scorer (~200 word lexicon)
├── aspects.py        # AspectExtractor — proximity-based aspect sentiment
├── emotions.py       # EmotionClassifier — Plutchik 8-emotion keyword classifier
├── analyzer.py       # SentimentAnalyzerPro — orchestrator + SentimentResult
└── cli.py            # Command-line interface
```

### LexiconSentiment

Bundled ~200-word sentiment lexicon with scores from -1.0 (very negative) to +1.0 (very positive). Handles:
- **Negation** (`not great` → dampened negative)
- **Intensifiers** (`extremely good` → boosted score)
- **Diminishers** (`somewhat good` → reduced score)
- **Capitalization** (`AMAZING` → slight boost)
- **Punctuation** (`!!!` → slight boost)

### AspectExtractor

Scans for 100+ aspect terms (product, software, service domains) and scores each using the lexicon applied to surrounding tokens. Contrastive connectors (`but`, `however`) split the text into sentiment regions so `"great battery but terrible camera"` correctly assigns opposite polarities.

### EmotionClassifier

Keyword-pattern classifier covering all 8 Plutchik basic emotions. Returns scores (0–1), primary/secondary emotions, valence, and matched keywords.

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
# or run individual test files
~/anaconda3/bin/python3 tests/test_lexicon.py
~/anaconda3/bin/python3 tests/test_aspects.py
~/anaconda3/bin/python3 tests/test_emotions.py
~/anaconda3/bin/python3 tests/test_analyzer.py
```

43 tests, all passing.

## Requirements

- Python 3.9+
- No external dependencies (stdlib only)
- numpy optional (not currently used, reserved for future vectorized scoring)
