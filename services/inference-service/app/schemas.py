from pydantic import BaseModel


class EmotionDetectRequest(BaseModel):
    image_b64: str
    request_id: str


class EmotionDetectResponse(BaseModel):
    request_id: str
    emotion: str
    raw_response: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    gpu_available: bool
