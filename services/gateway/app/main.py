import logging
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from prometheus_client import generate_latest
from starlette.responses import Response

from .circuit_breaker import CircuitOpenError
from .metrics import REQUEST_COUNT, REQUEST_LATENCY
from .orchestrator import call_inference, call_response, call_sentiment, lifespan_client
from .schemas import ChatRequest, ChatResponse, HealthResponse

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with lifespan_client():
        yield


app = FastAPI(title="Elyssa Gateway", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", service="gateway")


@app.get("/ready", response_model=HealthResponse)
async def ready():
    return HealthResponse(status="ready", service="gateway")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    request_id = req.request_id
    log.info("chat_request_received", request_id=request_id)

    try:
        # Step 1: Emotion detection from image
        t0 = time.time()
        inference_result = await call_inference(req.image_b64, request_id)
        inference_latency = (time.time() - t0) * 1000
        facial_emotion = inference_result.get("emotion", "neutral")

        # Step 2: Conflict detection (includes text sentiment)
        t1 = time.time()
        sentiment_result = await call_sentiment(facial_emotion, req.user_text, request_id)
        sentiment_latency = (time.time() - t1) * 1000
        is_conflicting = sentiment_result.get("is_conflicting", False)
        text_emotion = sentiment_result.get("text_emotion", "neutral")

        # Step 3: Response generation
        t2 = time.time()
        try:
            response_result = await call_response(
                req.user_text,
                facial_emotion,
                text_emotion,
                is_conflicting,
                request_id,
            )
            response_latency = (time.time() - t2) * 1000
            response_text = response_result.get("response_text", "")
        except CircuitOpenError:
            raise HTTPException(
                status_code=503,
                detail="Response generation service is temporarily unavailable. Please try again shortly.",
            )

        # Build emotion display string
        if is_conflicting:
            emotion_display = (
                f"{facial_emotion} (face) vs {text_emotion} (text) — conflicting. "
                "I notice there seems to be a difference between your facial expression and "
                "what you've shared. Would you feel comfortable sharing which emotion is "
                "closer to your experience right now?"
            )
        else:
            emotion_display = facial_emotion

        total_latency_ms = (time.time() - start) * 1000
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(total_latency_ms / 1000)

        log.info(
            "chat_request_complete",
            request_id=request_id,
            facial_emotion=facial_emotion,
            text_emotion=text_emotion,
            is_conflicting=is_conflicting,
            total_latency_ms=total_latency_ms,
        )

        return ChatResponse(
            request_id=request_id,
            response_text=response_text,
            facial_emotion=facial_emotion,
            text_emotion=text_emotion,
            is_conflicting=is_conflicting,
            emotion_display=emotion_display,
            total_latency_ms=total_latency_ms,
            service_latencies={
                "inference_ms": round(inference_latency, 2),
                "sentiment_ms": round(sentiment_latency, 2),
                "response_ms": round(response_latency, 2),
            },
        )

    except HTTPException:
        REQUEST_COUNT.labels(status="error").inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        log.error("chat_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
