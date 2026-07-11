import logging
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from prometheus_client import generate_latest
from starlette.responses import Response

from .metrics import REQUEST_COUNT, REQUEST_LATENCY
from .model import generate, gpu_available, is_model_loaded, load_language_model
from .prompt import create_prompt, parse_response
from .schemas import GenerateRequest, GenerateResponse, HealthResponse

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
    load_language_model()
    yield


app = FastAPI(title="Elyssa Response Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        service="response-service",
        model_loaded=is_model_loaded(),
        gpu_available=gpu_available(),
    )


@app.get("/ready", response_model=HealthResponse)
async def ready():
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return HealthResponse(
        status="ready",
        service="response-service",
        model_loaded=True,
        gpu_available=gpu_available(),
    )


@app.post("/generate-response", response_model=GenerateResponse)
async def generate_response_endpoint(req: GenerateRequest):
    start = time.time()
    try:
        prompt = create_prompt(
            req.user_text,
            req.facial_emotion,
            req.is_conflicting,
            req.text_emotion,
        )
        full_decoded = await generate(prompt, req.max_new_tokens)
        response_text = parse_response(full_decoded)
        latency_ms = (time.time() - start) * 1000
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(latency_ms / 1000)
        log.info(
            "response_generated",
            request_id=req.request_id,
            latency_ms=latency_ms,
        )
        return GenerateResponse(
            request_id=req.request_id,
            response_text=response_text,
            prompt_used=prompt,
            latency_ms=latency_ms,
        )
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        log.error("response_error", request_id=req.request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
