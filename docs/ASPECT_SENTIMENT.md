# Aspect-Based Sentiment Analysis

The aspect sentiment component extracts noun phrases from a review and assigns the overall predicted sentiment to each aspect. This provides a lightweight way to see which parts of a product are viewed positively or negatively.

Example usage:

```python
from src.models import build_model
from src.aspect_sentiment import predict_aspect_sentiment

model = build_model()
model.fit(["great camera"], ["positive"])
print(predict_aspect_sentiment("The camera quality is great", model))
```

This approach relies on simple noun extraction using NLTK and therefore may miss complex aspects. For best results with larger datasets, consider replacing it with a more sophisticated parser.

