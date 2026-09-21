"""End-to-end tests for the gateway HTTP surface, with the three downstream
services stubbed at the orchestrator boundary."""

import pytest
from fastapi.testclient import TestClient

from app import main as gateway_main
from app.circuit_breaker import CircuitOpenError
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def downstream(monkeypatch):
    """Stub the three orchestrator calls; each returns the recorded payload."""
    state = {
        "inference": {"emotion": "neutral", "raw_response": "", "latency_ms": 5},
        "sentiment": {
            "is_conflicting": False,
            "facial_emotion": "neutral",
            "text_emotion": "neutral",
            "facial_category": "neutral",
            "text_category": "neutral",
            "latency_ms": 2,
        },
        "response": {"response_text": "I hear you.", "latency_ms": 40},
        "seen": [],
    }

    async def fake_inference(image_b64, request_id):
        state["seen"].append(("inference", request_id))
        return _resolve(state["inference"])

    async def fake_sentiment(facial_emotion, text, request_id):
        state["seen"].append(("sentiment", request_id))
        return _resolve(state["sentiment"])

    async def fake_response(user_text, facial, text_emotion, conflicting, request_id):
        state["seen"].append(("response", request_id))
        return _resolve(state["response"])

    def _resolve(value):
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(gateway_main, "call_inference", fake_inference)
    monkeypatch.setattr(gateway_main, "call_sentiment", fake_sentiment)
    monkeypatch.setattr(gateway_main, "call_response", fake_response)
    return state


def payload(**overrides):
    body = {"user_text": "I had a rough week.", "image_b64": "aGk="}
    body.update(overrides)
    return body


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "gateway"}


def test_ready(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "service": "gateway"}


def test_metrics_exposes_prometheus_text(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "gateway_requests_total" in resp.text or "# HELP" in resp.text


# --------------------------------------------------------------------------
# /chat happy path
# --------------------------------------------------------------------------


def test_chat_returns_the_generated_reply(client, downstream):
    resp = client.post("/chat", json=payload())
    assert resp.status_code == 200
    assert resp.json()["response_text"] == "I hear you."


def test_chat_calls_the_three_services_in_order(client, downstream):
    client.post("/chat", json=payload())
    assert [name for name, _ in downstream["seen"]] == [
        "inference",
        "sentiment",
        "response",
    ]


def test_chat_threads_one_request_id_through_every_hop(client, downstream):
    client.post("/chat", json=payload(request_id="trace-me"))
    assert {rid for _, rid in downstream["seen"]} == {"trace-me"}


def test_chat_generates_a_request_id_when_the_caller_omits_one(client, downstream):
    body = client.post("/chat", json=payload()).json()
    assert body["request_id"]
    assert len(body["request_id"]) == 36  # uuid4


def test_chat_reports_per_service_latencies(client, downstream):
    latencies = client.post("/chat", json=payload()).json()["service_latencies"]
    assert set(latencies) == {"inference_ms", "sentiment_ms", "response_ms"}
    assert all(v >= 0 for v in latencies.values())


def test_chat_total_latency_covers_the_whole_request(client, downstream):
    body = client.post("/chat", json=payload()).json()
    assert body["total_latency_ms"] >= sum(body["service_latencies"].values())


# --------------------------------------------------------------------------
# conflict handling — the product's whole reason for existing
# --------------------------------------------------------------------------


def test_non_conflicting_emotion_display_is_just_the_facial_emotion(client, downstream):
    downstream["inference"] = {"emotion": "sad", "raw_response": "", "latency_ms": 1}
    downstream["sentiment"] = {
        "is_conflicting": False,
        "facial_emotion": "sad",
        "text_emotion": "sad",
        "facial_category": "negative",
        "text_category": "negative",
        "latency_ms": 1,
    }
    body = client.post("/chat", json=payload()).json()
    assert body["emotion_display"] == "sad"
    assert body["is_conflicting"] is False


def test_conflicting_emotion_display_names_both_signals_and_asks(client, downstream):
    downstream["inference"] = {"emotion": "happy", "raw_response": "", "latency_ms": 1}
    downstream["sentiment"] = {
        "is_conflicting": True,
        "facial_emotion": "happy",
        "text_emotion": "sad",
        "facial_category": "positive",
        "text_category": "negative",
        "latency_ms": 1,
    }
    body = client.post("/chat", json=payload()).json()
    assert body["is_conflicting"] is True
    assert "happy (face)" in body["emotion_display"]
    assert "sad (text)" in body["emotion_display"]
    assert "?" in body["emotion_display"], "a conflict should end in a question"


def test_chat_surfaces_both_emotions_as_separate_fields(client, downstream):
    downstream["inference"] = {"emotion": "angry", "raw_response": "", "latency_ms": 1}
    downstream["sentiment"] = {
        "is_conflicting": False,
        "facial_emotion": "angry",
        "text_emotion": "anxious",
        "facial_category": "negative",
        "text_category": "negative",
        "latency_ms": 1,
    }
    body = client.post("/chat", json=payload()).json()
    assert body["facial_emotion"] == "angry"
    assert body["text_emotion"] == "anxious"


# --------------------------------------------------------------------------
# degraded and failing downstreams
# --------------------------------------------------------------------------


def test_chat_defaults_to_neutral_when_inference_returns_nothing(client, downstream):
    downstream["inference"] = {}
    body = client.post("/chat", json=payload()).json()
    assert body["facial_emotion"] == "neutral"


def test_chat_defaults_to_neutral_when_sentiment_returns_nothing(client, downstream):
    downstream["sentiment"] = {}
    body = client.post("/chat", json=payload()).json()
    assert body["text_emotion"] == "neutral"
    assert body["is_conflicting"] is False


def test_chat_returns_empty_text_when_the_generator_returns_nothing(client, downstream):
    downstream["response"] = {}
    assert client.post("/chat", json=payload()).json()["response_text"] == ""


def test_open_response_circuit_returns_503_not_500(client, downstream):
    """A tripped breaker is a retryable outage, so it must not look like a bug."""
    downstream["response"] = CircuitOpenError("response circuit open")
    resp = client.post("/chat", json=payload())
    assert resp.status_code == 503
    assert "temporarily unavailable" in resp.json()["detail"]


def test_unexpected_downstream_error_becomes_a_500(client, downstream):
    downstream["response"] = RuntimeError("model exploded")
    resp = client.post("/chat", json=payload())
    assert resp.status_code == 500
    assert "model exploded" in resp.json()["detail"]


def test_inference_failure_still_produces_a_reply(client, downstream):
    """Inference degrades rather than raising, so the chat should complete."""
    downstream["inference"] = {"emotion": "neutral", "raw_response": "", "latency_ms": 0}
    resp = client.post("/chat", json=payload())
    assert resp.status_code == 200
    assert resp.json()["response_text"]


# --------------------------------------------------------------------------
# request validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["user_text", "image_b64"])
def test_chat_rejects_a_request_missing_a_required_field(client, downstream, missing):
    body = payload()
    del body[missing]
    assert client.post("/chat", json=body).status_code == 422


def test_chat_accepts_an_optional_session_id(client, downstream):
    resp = client.post("/chat", json=payload(session_id="sess-9"))
    assert resp.status_code == 200
