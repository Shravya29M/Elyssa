from pydantic import BaseModel


class GenerateRequest(BaseModel):
    user_text: str
    facial_emotion: str
    text_emotion: str
    is_conflicting: bool
    request_id: str
    max_new_tokens: int = 512


class GenerateResponse(BaseModel):
    request_id: str
    response_text: str
    prompt_used: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    gpu_available: bool
