import logging
import time

import structlog
from fastapi import FastAPI
from prometheus_client import generate_latest
from starlette.responses import Response

from .analyzer import analyze_text_sentiment, detect_conflict
from .metrics import REQUEST_COUNT, REQUEST_LATENCY
from .schemas import (
    ConflictRequest,
    ConflictResponse,
    HealthResponse,
    SentimentRequest,
    SentimentResponse,
)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

app = FastAPI(title="Elyssa Sentiment Service", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", service="sentiment-service")


@app.get("/ready", response_model=HealthResponse)
async def ready():
    return HealthResponse(status="ready", service="sentiment-service")


@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(req: SentimentRequest):
    start = time.time()
    try:
        emotion, compound = analyze_text_sentiment(req.text)
        latency_ms = (time.time() - start) * 1000
        REQUEST_COUNT.labels(endpoint="analyze-sentiment", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="analyze-sentiment").observe(latency_ms / 1000)
        log.info(
            "sentiment_analyzed",
            request_id=req.request_id,
            emotion=emotion,
            compound=compound,
            latency_ms=latency_ms,
        )
        return SentimentResponse(
            request_id=req.request_id,
            text_emotion=emotion,
            compound_score=compound,
            latency_ms=latency_ms,
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="analyze-sentiment", status="error").inc()
        log.error("sentiment_error", request_id=req.request_id, error=str(e))
        raise


@app.post("/detect-conflict", response_model=ConflictResponse)
async def detect_conflict_endpoint(req: ConflictRequest):
    start = time.time()
    try:
        result = detect_conflict(req.facial_emotion, req.text)
        latency_ms = (time.time() - start) * 1000
        REQUEST_COUNT.labels(endpoint="detect-conflict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="detect-conflict").observe(latency_ms / 1000)
        log.info(
            "conflict_detected",
            request_id=req.request_id,
            is_conflicting=result["is_conflicting"],
            latency_ms=latency_ms,
        )
        return ConflictResponse(
            request_id=req.request_id,
            latency_ms=latency_ms,
            **result,
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="detect-conflict", status="error").inc()
        log.error("conflict_error", request_id=req.request_id, error=str(e))
        raise


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
