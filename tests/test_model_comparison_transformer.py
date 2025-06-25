import types
import numpy as np

import src.model_comparison as mc


class DummyClfModel:
    def fit(self, X, y, **kwargs):
        self._labels = list(y)

    def predict(self, X):
        return self._labels[: len(X)]


class DummyKerasModel:
    def fit(self, *args, **kwargs):
        pass

    def predict(self, texts):
        return np.ones((len(texts), 1))


class DummyTokenizer:
    def __init__(self, num_words=10000):
        pass

    def fit_on_texts(self, texts):
        pass

    def texts_to_sequences(self, texts):
        return [[0] * len(t) for t in texts]


def _setup_deps(monkeypatch, transformer=True):
    data = {"text": ["good", "bad"], "label": np.array(["positive", "negative"])}

    class DummySeries(list):
        def apply(self, func):
            return [func(x) for x in self]

    class DummyDF:
        def __init__(self, data):
            self._data = {
                "text": DummySeries(data["text"]),
                "label": data["label"],
            }

        def __getitem__(self, key):
            return self._data[key]

    monkeypatch.setattr(mc, "pd", types.SimpleNamespace(read_csv=lambda path: DummyDF(data)))
    monkeypatch.setattr(
        mc,
        "train_test_split",
        lambda texts, labels, test_size=0.2, random_state=0: (texts, texts, labels, labels),
    )
    monkeypatch.setattr(mc, "accuracy_score", lambda y_true, y_pred: 1.0)

    dummy_keras = types.SimpleNamespace(
        preprocessing=types.SimpleNamespace(
            text=types.SimpleNamespace(Tokenizer=DummyTokenizer),
            sequence=types.SimpleNamespace(pad_sequences=lambda seqs, maxlen=100: seqs),
        )
    )
    monkeypatch.setattr(mc, "keras", dummy_keras)
    monkeypatch.setattr(mc, "build_model", lambda: DummyClfModel())
    monkeypatch.setattr(mc, "build_lstm_model", lambda: DummyKerasModel())

    if transformer:
        monkeypatch.setattr(mc, "build_transformer_model", lambda: DummyKerasModel(), raising=False)
    else:
        monkeypatch.setattr(mc, "build_transformer_model", None, raising=False)


def test_compare_models_includes_transformer(monkeypatch):
    _setup_deps(monkeypatch, transformer=True)
    results = mc.compare_models("dummy.csv")
    assert any(r["model"] == "Transformer" for r in results)


def test_compare_models_skips_transformer(monkeypatch):
    _setup_deps(monkeypatch, transformer=False)
    results = mc.compare_models("dummy.csv")
    assert not any(r["model"] == "Transformer" for r in results)
