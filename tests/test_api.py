import pytest
from fastapi.testclient import TestClient
from deployment.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_endpoint():
    payload = {
        "north_queue": 12,
        "south_queue": 8,
        "east_queue": 25,
        "west_queue": 20,
        "current_green_time": 10,
        "queue_growth": 3.0,
        "waiting_time": 15.0,
        "traffic_density": 0.65,
        "current_phase": 0.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "latency_ms" in data
    assert "prediction_trace" in data
