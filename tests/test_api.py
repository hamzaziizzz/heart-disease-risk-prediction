"""
tests/test_api.py
===================
Pytest unit/integration tests for api/main.py (Assignment Task 5 unit
tests, and evidence for Task 6's "Container must build and run locally with
sample input" requirement -- these same assertions are what you'd run
manually against the Dockerized container with curl/Postman).

These tests use FastAPI's TestClient, which runs the ASGI app in-process
(no real network socket / Docker container needed), so they are fast and
fully deterministic for CI. They DO require a trained model artifact to
exist at models/heart_disease_pipeline.joblib -- run `python src/train.py`
first (the CI workflow does this as an explicit prior step; see
.github/workflows/ci.yml).
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_disease_pipeline.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model artifact not found -- run `python src/train.py` first.",
)

VALID_PAYLOAD = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


@pytest.fixture
def client():
    """Yield a TestClient with the FastAPI lifespan (model loading) active."""
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_root_endpoint_returns_service_info(client):
    """GET / must return 200 with basic service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert "predict_endpoint" in body


def test_health_endpoint_reports_model_loaded(client):
    """GET /health must return 200 and confirm the model loaded successfully."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_valid_prediction_schema(client):
    """
    POST /predict with a valid patient record must return 200 and a body
    matching the PredictionResponse schema: prediction in {0,1}, confidence
    and probability_disease as floats in [0, 1].
    """
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] in (0, 1)
    assert body["prediction_label"] in ("No Heart Disease", "Heart Disease Present")
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0


def test_predict_rejects_out_of_range_input(client):
    """
    POST /predict with an invalid 'sex' value (must be 0 or 1) must be
    rejected with a 422 Unprocessable Entity BEFORE it reaches the model --
    proves Pydantic validation is correctly wired.
    """
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["sex"] = 5  # out of the allowed {0, 1} range
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    """POST /predict with a required field missing must return 422."""
    incomplete_payload = dict(VALID_PAYLOAD)
    del incomplete_payload["age"]
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(client):
    """
    GET /metrics must return 200 with Prometheus text-exposition-format
    content (Task 8: monitoring). We check for a couple of well-known
    metric name prefixes that prometheus-fastapi-instrumentator always
    emits, rather than asserting exact output (which is version-sensitive).
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text or "http_request" in response.text
