import json
import threading

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


def test_metrics_reflects_predictions_made(client):
    before = client.get("/metrics").get_json()["total_predictions"]

    client.post("/predict", json={"text": "A wonderful, brilliantly acted film."})
    client.post("/predict", json={"text": "A terrible, boring waste of time."})

    after = client.get("/metrics").get_json()
    assert after["total_predictions"] == before + 2
    assert after["label_counts"]["POSITIVE"] + after["label_counts"]["NEGATIVE"] >= 2
    assert 0.0 <= after["average_confidence"] <= 1.0


def test_metrics_counts_errors_separately_from_predictions(client, monkeypatch):
    def _boom(text):
        raise RuntimeError("No model is loaded.")

    monkeypatch.setattr(App, "predict_text", _boom)

    before = client.get("/metrics").get_json()["total_errors"]
    response = client.post("/predict", json={"text": "anything"})
    after = client.get("/metrics").get_json()["total_errors"]

    assert response.status_code == 503
    assert after == before + 1


def test_predict_emits_a_structured_json_log_line(client, caplog):
    with caplog.at_level("INFO", logger="sentiment_app"):
        client.post("/predict", json={"text": "A genuinely enjoyable movie."})

    assert len(caplog.records) >= 1
    payload = json.loads(caplog.records[-1].message)
    assert payload["event"] == "prediction"
    assert payload["label"] in ("POSITIVE", "NEGATIVE")
    assert payload["input_length"] == len("A genuinely enjoyable movie.")
    assert payload["latency_ms"] >= 0


def test_metrics_survive_concurrent_requests(client):
    N = 30
    results = []

    def _fire():
        r = App.app.test_client().post("/predict", json={"text": "A pretty good film overall."})
        results.append(r.status_code)

    before = client.get("/metrics").get_json()["total_predictions"]

    threads = [threading.Thread(target=_fire) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    after = client.get("/metrics").get_json()["total_predictions"]

    assert all(code == 200 for code in results)
    assert after == before + N
