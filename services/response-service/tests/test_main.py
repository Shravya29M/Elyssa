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


def test_ready_is_503_before_the_model_loads(monkeypatch):
    from app import main as response_main

    monkeypatch.setattr(response_main, "is_model_loaded", lambda: False)
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert "not yet loaded" in resp.json()["detail"]


def test_health_stays_200_while_the_model_is_still_loading(monkeypatch):
    from app import main as response_main

    monkeypatch.setattr(response_main, "is_model_loaded", lambda: False)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


def test_lifespan_loads_the_model_on_startup():
    from fastapi.testclient import TestClient as TC

    from app.main import app as response_app

    with TC(response_app) as c:
        assert c.get("/ready").status_code == 200


def _body(**overrides):
    body = {
        "user_text": "I had a rough week.",
        "facial_emotion": "sad",
        "text_emotion": "sad",
        "is_conflicting": False,
        "request_id": "r1",
    }
    body.update(overrides)
    return body


def test_generate_returns_the_prompt_it_used():
    """The prompt is echoed back so a bad reply can be traced to its input."""
    body = client.post("/generate-response", json=_body()).json()
    assert "### Instruction:" in body["prompt_used"]
    assert "I had a rough week." in body["prompt_used"]


def test_generate_prompt_reflects_a_detected_conflict():
    body = client.post(
        "/generate-response",
        json=_body(facial_emotion="happy", text_emotion="sad", is_conflicting=True),
    ).json()
    assert "conflict" in body["prompt_used"].lower()


def test_generate_reports_latency_and_echoes_the_request_id():
    body = client.post("/generate-response", json=_body(request_id="trace-me")).json()
    assert body["request_id"] == "trace-me"
    assert body["latency_ms"] >= 0


def test_generator_failure_becomes_a_500(monkeypatch):
    from app import main as response_main

    async def boom(*_args, **_kwargs):
        raise RuntimeError("cuda oom")

    monkeypatch.setattr(response_main, "generate", boom)
    resp = client.post("/generate-response", json=_body())
    assert resp.status_code == 500
    assert "cuda oom" in resp.json()["detail"]


def test_generate_requires_its_fields():
    body = _body()
    del body["user_text"]
    assert client.post("/generate-response", json=body).status_code == 422


def test_max_new_tokens_defaults_when_omitted():
    resp = client.post("/generate-response", json=_body())
    assert resp.status_code == 200
