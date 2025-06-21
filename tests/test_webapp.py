import joblib
import pytest

from src.models import build_model
from src import webapp


def test_webapp_predict_endpoint(tmp_path):
    pytest.importorskip('flask')
    pytest.importorskip('sklearn')
    model = build_model()
    model.fit(["good", "bad"], ["positive", "negative"])
    model_file = tmp_path / 'model.joblib'
    joblib.dump(model, model_file)

    webapp._model_cache.clear()
    webapp.MODEL_PATH = str(model_file)

    with webapp.app.test_client() as client:
        resp = client.post('/predict', json={'text': 'good'})
        assert resp.status_code == 200
        assert resp.get_json()['prediction'] == 'positive'
