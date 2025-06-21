from src.models import build_model, build_lstm_model, build_transformer_model


def test_model_fit_predict():
    texts = ["good", "bad"]
    labels = ["positive", "negative"]
    model = build_model()
    model.fit(texts, labels)
    preds = model.predict(["good", "bad"])
    assert list(preds) == ["positive", "negative"]


def test_build_lstm_model():
    model = build_lstm_model()
    assert isinstance(model.layers[-1].activation.__name__, str)


def test_build_transformer_model():
    model = build_transformer_model()
    assert model.config.num_labels == 2
