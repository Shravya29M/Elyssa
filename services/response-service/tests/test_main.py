import os

os.environ["MOCK_MODELS"] = "true"

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready():
    resp = client.get("/ready")
    assert resp.status_code == 200


def test_generate_response_mock():
    resp = client.post(
        "/generate-response",
        json={
            "user_text": "I feel overwhelmed",
            "facial_emotion": "anxious",
            "text_emotion": "anxious",
            "is_conflicting": False,
            "request_id": "test-001",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "test-001"
    assert len(data["response_text"]) > 0
    assert data["latency_ms"] >= 0


def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200
