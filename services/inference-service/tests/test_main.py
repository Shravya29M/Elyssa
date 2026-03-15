import base64
import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

os.environ["MOCK_MODELS"] = "true"

from app.main import app  # noqa: E402

client = TestClient(app)


def _make_b64_image() -> str:
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_ready():
    resp = client.get("/ready")
    assert resp.status_code == 200


def test_detect_emotion_mock():
    resp = client.post(
        "/detect-emotion",
        json={"image_b64": _make_b64_image(), "request_id": "test-001"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["emotion"] == "neutral"
    assert data["request_id"] == "test-001"
    assert data["latency_ms"] >= 0


def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"elyssa_inference" in resp.content
