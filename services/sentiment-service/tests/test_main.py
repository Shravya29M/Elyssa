import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok", "service": "sentiment-service"}


def test_ready():
    assert client.get("/ready").json()["status"] == "ready"


def test_metrics_exposes_prometheus_text():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "# HELP" in resp.text


# --------------------------------------------------------------------------
# /analyze-sentiment
# --------------------------------------------------------------------------


def test_analyze_sentiment_returns_emotion_and_score():
    resp = client.post(
        "/analyze-sentiment", json={"text": "I feel wonderful today!", "request_id": "r1"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"] == "r1"
    assert body["text_emotion"] in {"happy", "excited"}
    assert body["compound_score"] > 0


def test_analyze_sentiment_reports_latency():
    body = client.post(
        "/analyze-sentiment", json={"text": "hello", "request_id": "r1"}
    ).json()
    assert body["latency_ms"] >= 0


def test_analyze_sentiment_handles_empty_text():
    resp = client.post("/analyze-sentiment", json={"text": "", "request_id": "r1"})
    assert resp.status_code == 200
    assert resp.json()["text_emotion"] == "neutral"


@pytest.mark.parametrize("missing", ["text", "request_id"])
def test_analyze_sentiment_requires_its_fields(missing):
    body = {"text": "hi", "request_id": "r1"}
    del body[missing]
    assert client.post("/analyze-sentiment", json=body).status_code == 422


# --------------------------------------------------------------------------
# /detect-conflict
# --------------------------------------------------------------------------


def test_detect_conflict_flags_a_smiling_face_over_painful_text():
    resp = client.post(
        "/detect-conflict",
        json={
            "facial_emotion": "happy",
            "text": "I feel hopeless and completely worthless.",
            "request_id": "r1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_conflicting"] is True
    assert body["facial_category"] == "positive"
    assert body["text_category"] == "negative"


def test_detect_conflict_returns_false_when_signals_agree():
    body = client.post(
        "/detect-conflict",
        json={
            "facial_emotion": "sad",
            "text": "I feel hopeless and drained.",
            "request_id": "r1",
        },
    ).json()
    assert body["is_conflicting"] is False


def test_detect_conflict_echoes_the_facial_emotion_it_was_given():
    body = client.post(
        "/detect-conflict",
        json={"facial_emotion": "surprised", "text": "ok", "request_id": "r1"},
    ).json()
    assert body["facial_emotion"] == "surprised"


def test_detect_conflict_returns_every_documented_field():
    body = client.post(
        "/detect-conflict",
        json={"facial_emotion": "happy", "text": "I am fine", "request_id": "r1"},
    ).json()
    assert set(body) == {
        "request_id",
        "is_conflicting",
        "facial_emotion",
        "text_emotion",
        "facial_category",
        "text_category",
        "latency_ms",
    }


def test_detect_conflict_treats_an_unknown_facial_emotion_as_neutral():
    body = client.post(
        "/detect-conflict",
        json={"facial_emotion": "bewildered", "text": "I am hopeless", "request_id": "r1"},
    ).json()
    assert body["facial_category"] == "neutral"
    assert body["is_conflicting"] is False


@pytest.mark.parametrize("missing", ["facial_emotion", "text", "request_id"])
def test_detect_conflict_requires_its_fields(missing):
    body = {"facial_emotion": "happy", "text": "hi", "request_id": "r1"}
    del body[missing]
    assert client.post("/detect-conflict", json=body).status_code == 422


def test_endpoint_errors_are_surfaced_not_swallowed(monkeypatch):
    from app import main as sentiment_main

    def boom(*_args, **_kwargs):
        raise RuntimeError("analyzer down")

    monkeypatch.setattr(sentiment_main, "detect_conflict", boom)
    with pytest.raises(RuntimeError, match="analyzer down"):
        client.post(
            "/detect-conflict",
            json={"facial_emotion": "happy", "text": "hi", "request_id": "r1"},
        )


def test_sentiment_endpoint_errors_are_surfaced_not_swallowed(monkeypatch):
    from app import main as sentiment_main

    def boom(*_args, **_kwargs):
        raise RuntimeError("vader down")

    monkeypatch.setattr(sentiment_main, "analyze_text_sentiment", boom)
    with pytest.raises(RuntimeError, match="vader down"):
        client.post("/analyze-sentiment", json={"text": "hi", "request_id": "r1"})
