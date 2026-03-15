import os
from contextlib import asynccontextmanager

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .circuit_breaker import CircuitBreaker, CircuitOpenError

log = structlog.get_logger()

INFERENCE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://inference-service:8001")
SENTIMENT_URL = os.getenv("SENTIMENT_SERVICE_URL", "http://sentiment-service:8002")
RESPONSE_URL = os.getenv("RESPONSE_SERVICE_URL", "http://response-service:8003")

_client: httpx.AsyncClient | None = None

inference_breaker = CircuitBreaker("inference", failure_threshold=5, recovery_timeout=30)
sentiment_breaker = CircuitBreaker("sentiment", failure_threshold=5, recovery_timeout=10)
response_breaker = CircuitBreaker("response", failure_threshold=3, recovery_timeout=60)


@asynccontextmanager
async def lifespan_client():
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=5.0))
    yield
    await _client.aclose()


def _get_client() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("HTTP client not initialised — call lifespan_client first")
    return _client


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=0.5, max=4),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True,
)
async def _call_inference_raw(image_b64: str, request_id: str) -> dict:
    resp = await _get_client().post(
        f"{INFERENCE_URL}/detect-emotion",
        json={"image_b64": image_b64, "request_id": request_id},
        timeout=httpx.Timeout(30.0, connect=5.0),
    )
    resp.raise_for_status()
    return resp.json()


async def call_inference(image_b64: str, request_id: str) -> dict:
    try:
        return await inference_breaker.call(
            _call_inference_raw(image_b64, request_id)
        )
    except CircuitOpenError:
        log.warning("inference_circuit_open", request_id=request_id)
        return {"emotion": "neutral", "raw_response": "", "latency_ms": 0}
    except Exception as e:
        log.error("inference_failed", request_id=request_id, error=str(e))
        return {"emotion": "neutral", "raw_response": "", "latency_ms": 0}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.3, max=3),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True,
)
async def _call_sentiment_raw(facial_emotion: str, text: str, request_id: str) -> dict:
    resp = await _get_client().post(
        f"{SENTIMENT_URL}/detect-conflict",
        json={
            "facial_emotion": facial_emotion,
            "text": text,
            "request_id": request_id,
        },
        timeout=httpx.Timeout(5.0, connect=5.0),
    )
    resp.raise_for_status()
    return resp.json()


async def call_sentiment(facial_emotion: str, text: str, request_id: str) -> dict:
    try:
        return await sentiment_breaker.call(
            _call_sentiment_raw(facial_emotion, text, request_id)
        )
    except CircuitOpenError:
        log.warning("sentiment_circuit_open", request_id=request_id)
        return {
            "is_conflicting": False,
            "facial_emotion": facial_emotion,
            "text_emotion": "neutral",
            "facial_category": "neutral",
            "text_category": "neutral",
            "latency_ms": 0,
        }
    except Exception as e:
        log.error("sentiment_failed", request_id=request_id, error=str(e))
        return {
            "is_conflicting": False,
            "facial_emotion": facial_emotion,
            "text_emotion": "neutral",
            "facial_category": "neutral",
            "text_category": "neutral",
            "latency_ms": 0,
        }


@retry(
    stop=stop_after_attempt(1),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True,
)
async def _call_response_raw(
    user_text: str,
    facial_emotion: str,
    text_emotion: str,
    is_conflicting: bool,
    request_id: str,
) -> dict:
    resp = await _get_client().post(
        f"{RESPONSE_URL}/generate-response",
        json={
            "user_text": user_text,
            "facial_emotion": facial_emotion,
            "text_emotion": text_emotion,
            "is_conflicting": is_conflicting,
            "request_id": request_id,
        },
        timeout=httpx.Timeout(60.0, connect=5.0),
    )
    resp.raise_for_status()
    return resp.json()


async def call_response(
    user_text: str,
    facial_emotion: str,
    text_emotion: str,
    is_conflicting: bool,
    request_id: str,
) -> dict:
    return await response_breaker.call(
        _call_response_raw(
            user_text, facial_emotion, text_emotion, is_conflicting, request_id
        )
    )
