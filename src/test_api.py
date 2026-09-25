"""
Basic smoke tests for the FastAPI service. Requires a model already
registered in MLflow (run train.py first).
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_response():
    payload = {
        "season": 1,
        "holiday": 0,
        "workingday": 1,
        "weather": 1,
        "temp": 20.5,
        "humidity": 60.0,
        "windspeed": 10.0,
        "hour": 8,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_count" in body
    assert isinstance(body["predicted_count"], float)
