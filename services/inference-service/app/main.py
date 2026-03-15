import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager

import structlog
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from prometheus_client import generate_latest
from starlette.responses import Response

from .metrics import REQUEST_COUNT, REQUEST_LATENCY
from .model import detect_emotion, is_model_loaded, load_vision_model
from .schemas import EmotionDetectRequest, EmotionDetectResponse, HealthResponse

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
    load_vision_model()
    yield


app = FastAPI(title="Elyssa Inference Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        service="inference-service",
        model_loaded=is_model_loaded(),
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/ready", response_model=HealthResponse)
async def ready():
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return HealthResponse(
        status="ready",
        service="inference-service",
        model_loaded=True,
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/detect-emotion", response_model=EmotionDetectResponse)
async def detect_emotion_endpoint(req: EmotionDetectRequest):
    start = time.time()
    try:
        image_bytes = base64.b64decode(req.image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        emotion, raw_response = await detect_emotion(image)
        latency_ms = (time.time() - start) * 1000
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(latency_ms / 1000)
        log.info(
            "emotion_detected",
            request_id=req.request_id,
            emotion=emotion,
            latency_ms=latency_ms,
        )
        return EmotionDetectResponse(
            request_id=req.request_id,
            emotion=emotion,
            raw_response=raw_response,
            latency_ms=latency_ms,
        )
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        log.error("inference_error", request_id=req.request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
