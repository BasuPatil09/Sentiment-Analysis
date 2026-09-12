import pytest

import App


@pytest.fixture
def client():
    App.app.config["TESTING"] = True
    return App.app.test_client()


def test_health_reports_a_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model_source"] in ("direct", "mlflow_uri", "legacy_joblib")


def test_predict_returns_a_well_formed_response(client):
    response = client.post("/predict", json={"text": "A decent enough movie."})
    assert response.status_code == 200
    body = response.get_json()
    assert body["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_direction_on_a_clearly_positive_review(client):
    response = client.post("/predict", json={
        "text": "An absolutely brilliant, wonderful, masterfully acted film. I loved every second."
    })
    assert response.get_json()["label"] == "POSITIVE"


def test_predict_direction_on_a_clearly_negative_review(client):
    response = client.post("/predict", json={
        "text": "A terrible, boring, awful movie. Complete waste of time, I hated it."
    })
    assert response.get_json()["label"] == "NEGATIVE"


def test_compare_returns_all_five_models(client):
    response = client.get("/compare")
    assert response.status_code == 200
    body = response.get_json()
    assert len(body) == 5
    for metrics in body.values():
        assert {"accuracy", "f1_macro", "roc_auc"} <= metrics.keys()
