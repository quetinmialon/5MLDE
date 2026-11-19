from uuid import UUID, uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bank_marketing import database
from bank_marketing.api import app as api_module
from bank_marketing.preprocessing import FEATURE_COLUMNS

CLIENT = {
    "age": 35, "job": "management", "marital": "married", "education": "tertiary",
    "default": "no", "balance": 1200, "housing": "yes", "loan": "no", "contact": "cellular",
    "day": 15, "month": "may", "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown",
}


class FakeModel:
    def __init__(self, probability=0.7):
        self.probability = probability
        self.frames = []

    def predict_proba(self, frame):
        self.frames.append(frame)
        return np.array([[1 - self.probability, self.probability]])


@pytest.fixture
def api(monkeypatch, fake_database):
    serving = {"model": FakeModel(), "version": "3"}

    def load_champion(config_path):
        if serving["model"] is None:
            raise ValueError("Aucun modèle champion défini.")
        return serving["model"], serving["version"]

    monkeypatch.setattr(api_module, "load_champion", load_champion)
    monkeypatch.setattr(api_module, "load_training_config", lambda path: {"training": {"threshold": 0.5}})

    def start(**options):
        return TestClient(api_module.app, **options)
    return start, serving, fake_database


def test_prediction_returns_exact_version_and_is_persisted(api):
    start, serving, stored = api
    with start() as client:
        response = client.post("/predict", json=CLIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "3"
    assert body["probability"] == pytest.approx(0.7)
    assert body["predicted_label"] == "yes" and body["threshold"] == 0.5
    saved = stored["predictions"][UUID(body["prediction_id"])]
    assert saved["model_version"] == "3" and saved["features"] == CLIENT
    assert list(serving["model"].frames[0].columns) == FEATURE_COLUMNS


def test_probability_below_threshold_predicts_no(api):
    start, serving, _ = api
    serving["model"] = FakeModel(probability=0.2)
    with start() as client:
        assert client.post("/predict", json=CLIENT).json()["predicted_label"] == "no"


@pytest.mark.parametrize("change", [
    {"day": 32}, {"job": "pilot"}, {"age": "35"}, {"age": 35.5}, {"pdays": -2},
    {"duration": 120}, {"y": "yes"},
])
def test_invalid_inputs_are_rejected_before_the_model_and_traced(api, change):
    start, serving, stored = api
    with start() as client:
        response = client.post("/predict", json={**CLIENT, **change})
    assert response.status_code == 422
    assert serving["model"].frames == []
    assert stored["errors"][-1]["route"] == "/predict" and stored["errors"][-1]["status_code"] == 422


def test_missing_field_is_rejected(api):
    start, _, _ = api
    payload = {key: value for key, value in CLIENT.items() if key != "balance"}
    with start() as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_without_champion_api_starts_but_refuses_predictions(api):
    start, serving, stored = api
    serving["model"] = None
    with start() as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        response = client.post("/predict", json=CLIENT)
    assert ready.status_code == 503 and ready.json()["model_version"] is None
    assert response.status_code == 503
    assert stored["errors"][-1]["status_code"] == 503


def test_persistence_failure_is_visible_and_returns_no_prediction(api, monkeypatch):
    start, _, stored = api

    def unavailable(*args):
        raise database.PersistenceError("Base de données indisponible.")

    monkeypatch.setattr(database, "save_prediction", unavailable)
    with start() as client:
        response = client.post("/predict", json=CLIENT)
    assert response.status_code == 503
    assert "prediction_id" not in response.json()
    assert "non enregistrée" in stored["errors"][-1]["error"]


def test_ready_reports_unavailable_database(api, monkeypatch):
    start, _, _ = api

    def unavailable():
        raise database.PersistenceError("Base de données indisponible.")

    monkeypatch.setattr(database, "check_database", unavailable)
    with start() as client:
        response = client.get("/ready")
    assert response.status_code == 503 and response.json()["database"] == "unavailable"


def test_feedback_is_linked_to_an_existing_prediction_once(api):
    start, _, stored = api
    with start() as client:
        prediction_id = client.post("/predict", json=CLIENT).json()["prediction_id"]
        first = client.post("/feedback", json={"prediction_id": prediction_id, "actual_label": "no"})
        duplicate = client.post("/feedback", json={"prediction_id": prediction_id, "actual_label": "yes"})
        unknown = client.post("/feedback", json={"prediction_id": str(uuid4()), "actual_label": "no"})
        invalid = client.post("/feedback", json={"prediction_id": prediction_id, "actual_label": "maybe"})
    assert first.status_code == 201
    assert duplicate.status_code == 409 and unknown.status_code == 404 and invalid.status_code == 422
    assert stored["feedbacks"] == {UUID(prediction_id): "no"}
    assert [error["status_code"] for error in stored["errors"]] == [409, 404, 422]


def test_refresh_serves_new_champion_and_failure_keeps_current_one(api):
    start, serving, _ = api
    with start() as client:
        serving["version"] = "4"
        assert client.post("/predict", json=CLIENT).json()["model_version"] == "3"
        assert client.post("/model/refresh").json() == {"previous_version": "3", "model_version": "4"}
        serving["model"] = None
        failed = client.post("/model/refresh")
        after_failure = client.post("/predict", json=CLIENT)
    assert failed.status_code == 503
    assert after_failure.json()["model_version"] == "4"


def test_unexpected_model_error_is_traced_as_500(api):
    start, serving, stored = api

    class BrokenModel(FakeModel):
        def predict_proba(self, frame):
            raise RuntimeError("panne")

    serving["model"] = BrokenModel()
    with start(raise_server_exceptions=False) as client:
        response = client.post("/predict", json=CLIENT)
    assert response.status_code == 500
    assert stored["errors"][-1] == {
        "route": "/predict", "status_code": 500, "error": "Erreur interne : RuntimeError", "model_version": "3",
    }


def metric_value(client, line_start):
    lines = [line for line in client.get("/metrics").text.splitlines() if line.startswith(line_start)]
    return float(lines[0].rsplit(" ", 1)[1]) if lines else 0.0


def test_metrics_count_requests_by_declared_route_and_status(api):
    start, _, _ = api
    with start() as client:
        before = metric_value(client, 'api_requests_total{route="/predict",status="200"}')
        client.post("/predict", json=CLIENT)
        client.post("/predict", json={**CLIENT, "day": 0})
        client.get("/does-not-exist")
        assert metric_value(client, 'api_requests_total{route="/predict",status="200"}') == before + 1
        assert metric_value(client, 'api_requests_total{route="/predict",status="422"}') >= 1
        assert metric_value(client, 'api_requests_total{route="unmatched",status="404"}') >= 1
        assert metric_value(client, 'api_request_duration_seconds_count{route="/predict"}') >= 2
        assert 'route="/metrics"' not in client.get("/metrics").text


def test_metrics_expose_served_model_version(api):
    start, serving, _ = api
    with start() as client:
        assert metric_value(client, "api_model_version") == 3
        serving["version"] = "4"
        client.post("/model/refresh")
        assert metric_value(client, "api_model_version") == 4
