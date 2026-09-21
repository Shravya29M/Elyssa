import base64
import io
import os

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


def test_ready_is_503_before_the_model_loads(monkeypatch):
    from app import main as inference_main

    monkeypatch.setattr(inference_main, "is_model_loaded", lambda: False)
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert "not yet loaded" in resp.json()["detail"]


def test_health_stays_200_while_the_model_is_still_loading(monkeypatch):
    """Liveness must not fail during warm-up, or the orchestrator kills the pod."""
    from app import main as inference_main

    monkeypatch.setattr(inference_main, "is_model_loaded", lambda: False)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


def test_detect_emotion_rejects_a_non_base64_payload():
    resp = client.post("/detect-emotion", json={"image_b64": "!!!", "request_id": "r1"})
    assert resp.status_code == 500


def test_detect_emotion_rejects_bytes_that_are_not_an_image():
    import base64

    junk = base64.b64encode(b"this is not a jpeg").decode()
    resp = client.post("/detect-emotion", json={"image_b64": junk, "request_id": "r1"})
    assert resp.status_code == 500


def test_detect_emotion_requires_its_fields():
    assert client.post("/detect-emotion", json={"request_id": "r1"}).status_code == 422


def test_lifespan_loads_the_model_on_startup():
    from fastapi.testclient import TestClient as TC

    from app.main import app as inference_app

    with TC(inference_app) as c:
        assert c.get("/ready").status_code == 200


def test_detect_emotion_converts_a_grayscale_image_to_rgb():
    import base64
    import io

    img = Image.new("L", (32, 32), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = client.post("/detect-emotion", json={"image_b64": b64, "request_id": "r1"})
    assert resp.status_code == 200
    assert resp.json()["emotion"] in {"neutral"}
