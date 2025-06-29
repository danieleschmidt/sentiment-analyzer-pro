import joblib
import pytest

from src.models import build_model


@pytest.fixture(autouse=True)
def _reset_counters():
    """Reset metrics counters before each test."""
    from src import webapp

    webapp.REQUEST_COUNT = 0
    webapp.PREDICTION_COUNT = 0
    yield
    webapp.REQUEST_COUNT = 0
    webapp.PREDICTION_COUNT = 0


def test_webapp_predict_endpoint(tmp_path):
    pytest.importorskip('flask')
    pytest.importorskip('sklearn')
    from src import webapp

    model = build_model()
    model.fit(["good", "bad"], ["positive", "negative"])
    model_file = tmp_path / 'model.joblib'
    joblib.dump(model, model_file)

    webapp.load_model.cache_clear()
    webapp.MODEL_PATH = str(model_file)

    with webapp.app.test_client() as client:
        resp = client.post('/predict', json={'text': 'good'})
        assert resp.status_code == 200
        assert resp.get_json()['prediction'] == 'positive'


def test_webapp_root_endpoint(tmp_path):
    pytest.importorskip('flask')
    from src import webapp

    model_file = tmp_path / 'model.joblib'
    joblib.dump(build_model().fit(["good", "bad"], ["positive", "negative"]), model_file)
    webapp.load_model.cache_clear()
    webapp.MODEL_PATH = str(model_file)

    with webapp.app.test_client() as client:
        resp = client.get('/')
        assert resp.status_code == 200
        assert resp.get_json()['status'] == 'ok'


def test_webapp_version_endpoint(monkeypatch):
    pytest.importorskip('flask')
    from src import webapp

    monkeypatch.setattr(webapp, 'APP_VERSION', 'test-version')

    with webapp.app.test_client() as client:
        resp = client.get('/version')
        assert resp.status_code == 200
        assert resp.get_json()['version'] == 'test-version'


def test_webapp_metrics_endpoint(tmp_path):
    pytest.importorskip('flask')
    pytest.importorskip('sklearn')
    from src import webapp

    model = build_model()
    model.fit(["good", "bad"], ["positive", "negative"])
    model_file = tmp_path / 'model.joblib'
    joblib.dump(model, model_file)

    webapp.load_model.cache_clear()
    webapp.MODEL_PATH = str(model_file)

    with webapp.app.test_client() as client:
        client.post('/predict', json={'text': 'good'})
        resp = client.get('/metrics')
        data = resp.get_json()
        assert resp.status_code == 200
        assert data['predictions'] == 1
        assert data['requests'] >= 2  # at least predict + metrics
