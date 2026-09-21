"""The orchestrator is the gateway's resilience layer: it retries transient
failures, trips circuit breakers, and degrades to neutral defaults so a dead
downstream service never takes the whole chat request with it. None of that
was covered before.
"""

import httpx
import pytest

from app import orchestrator
from app.circuit_breaker import CircuitState


class StubClient:
    """Minimal stand-in for httpx.AsyncClient.

    `responses` is a list of either httpx.Response objects to return or
    exceptions to raise, consumed one per POST.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "http://svc"))


def status(code: int) -> httpx.Response:
    return httpx.Response(code, json={}, request=httpx.Request("POST", "http://svc"))


NEUTRAL_INFERENCE = {"emotion": "neutral", "raw_response": "", "latency_ms": 0}


# --------------------------------------------------------------------------
# client lifecycle
# --------------------------------------------------------------------------


async def test_get_client_raises_before_lifespan_starts(monkeypatch):
    monkeypatch.setattr(orchestrator, "_client", None)
    with pytest.raises(RuntimeError, match="not initialised"):
        orchestrator._get_client()


async def test_lifespan_client_opens_and_closes_the_client():
    async with orchestrator.lifespan_client():
        client = orchestrator._get_client()
        assert isinstance(client, httpx.AsyncClient)
        assert not client.is_closed
    assert client.is_closed


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


async def test_call_inference_returns_the_service_payload(fake_client):
    fake_client(StubClient([ok({"emotion": "happy", "raw_response": "happy", "latency_ms": 12})]))
    result = await orchestrator.call_inference("aGk=", "req-1")
    assert result["emotion"] == "happy"


async def test_call_inference_posts_the_image_and_request_id(fake_client):
    client = fake_client(StubClient([ok({"emotion": "sad"})]))
    await orchestrator.call_inference("aGk=", "req-1")
    url, kwargs = client.calls[0]
    assert url.endswith("/detect-emotion")
    assert kwargs["json"] == {"image_b64": "aGk=", "request_id": "req-1"}


async def test_call_inference_retries_once_on_timeout(fake_client):
    client = fake_client(
        StubClient([httpx.TimeoutException("slow"), ok({"emotion": "angry"})])
    )
    result = await orchestrator.call_inference("aGk=", "req-1")
    assert result["emotion"] == "angry"
    assert len(client.calls) == 2


async def test_call_inference_degrades_to_neutral_when_the_service_is_down(fake_client):
    fake_client(StubClient([httpx.ConnectError("refused"), httpx.ConnectError("refused")]))
    assert await orchestrator.call_inference("aGk=", "req-1") == NEUTRAL_INFERENCE


async def test_call_inference_degrades_to_neutral_on_a_5xx(fake_client):
    fake_client(StubClient([status(500)]))
    assert await orchestrator.call_inference("aGk=", "req-1") == NEUTRAL_INFERENCE


async def test_repeated_inference_failures_trip_the_breaker(fake_client):
    threshold = orchestrator.inference_breaker.failure_threshold
    fake_client(StubClient([status(500)] * threshold))
    for _ in range(threshold):
        await orchestrator.call_inference("aGk=", "req-1")
    assert orchestrator.inference_breaker.state == CircuitState.OPEN


async def test_open_inference_breaker_serves_neutral_without_a_network_call(fake_client):
    threshold = orchestrator.inference_breaker.failure_threshold
    client = fake_client(StubClient([status(500)] * threshold))
    for _ in range(threshold):
        await orchestrator.call_inference("aGk=", "req-1")

    calls_before = len(client.calls)
    assert await orchestrator.call_inference("aGk=", "req-2") == NEUTRAL_INFERENCE
    assert len(client.calls) == calls_before, "open breaker must fail fast"


# --------------------------------------------------------------------------
# sentiment
# --------------------------------------------------------------------------


async def test_call_sentiment_returns_the_service_payload(fake_client):
    fake_client(
        StubClient([ok({"is_conflicting": True, "text_emotion": "sad", "latency_ms": 3})])
    )
    result = await orchestrator.call_sentiment("happy", "I feel awful", "req-1")
    assert result["is_conflicting"] is True
    assert result["text_emotion"] == "sad"


async def test_call_sentiment_retries_twice_before_giving_up(fake_client):
    client = fake_client(
        StubClient(
            [httpx.TimeoutException("slow"), httpx.TimeoutException("slow"), ok({"is_conflicting": False})]
        )
    )
    await orchestrator.call_sentiment("happy", "hi", "req-1")
    assert len(client.calls) == 3


async def test_call_sentiment_fallback_preserves_the_facial_emotion(fake_client):
    """The face was already detected, so a sentiment outage must not discard it."""
    fake_client(StubClient([status(503)]))
    result = await orchestrator.call_sentiment("angry", "whatever", "req-1")
    assert result["facial_emotion"] == "angry"
    assert result["is_conflicting"] is False
    assert result["text_emotion"] == "neutral"


async def test_open_sentiment_breaker_falls_back_without_a_call(fake_client):
    threshold = orchestrator.sentiment_breaker.failure_threshold
    client = fake_client(StubClient([status(500)] * threshold))
    for _ in range(threshold):
        await orchestrator.call_sentiment("sad", "hi", "req-1")

    calls_before = len(client.calls)
    result = await orchestrator.call_sentiment("sad", "hi", "req-2")
    assert result["facial_emotion"] == "sad"
    assert len(client.calls) == calls_before


# --------------------------------------------------------------------------
# response
# --------------------------------------------------------------------------


async def test_call_response_returns_the_generated_text(fake_client):
    fake_client(StubClient([ok({"response_text": "I hear you.", "latency_ms": 900})]))
    result = await orchestrator.call_response("hi", "sad", "sad", False, "req-1")
    assert result["response_text"] == "I hear you."


async def test_call_response_forwards_the_full_context(fake_client):
    client = fake_client(StubClient([ok({"response_text": "ok"})]))
    await orchestrator.call_response("hi", "happy", "sad", True, "req-1")
    _, kwargs = client.calls[0]
    assert kwargs["json"] == {
        "user_text": "hi",
        "facial_emotion": "happy",
        "text_emotion": "sad",
        "is_conflicting": True,
        "request_id": "req-1",
    }


async def test_call_response_propagates_failures_rather_than_degrading(fake_client):
    """There is no useful neutral fallback for a counselling reply, so the
    gateway must surface the failure instead of inventing one."""
    fake_client(StubClient([status(500)]))
    with pytest.raises(httpx.HTTPStatusError):
        await orchestrator.call_response("hi", "sad", "sad", False, "req-1")


async def test_response_breaker_opens_after_three_failures(fake_client):
    fake_client(StubClient([status(500)] * 3))
    for _ in range(3):
        with pytest.raises(httpx.HTTPStatusError):
            await orchestrator.call_response("hi", "sad", "sad", False, "req-1")
    assert orchestrator.response_breaker.state == CircuitState.OPEN
