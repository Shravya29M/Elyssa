import uuid
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_text: str
    image_b64: str
    session_id: Optional[str] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ChatResponse(BaseModel):
    request_id: str
    response_text: str
    facial_emotion: str
    text_emotion: str
    is_conflicting: bool
    emotion_display: str
    total_latency_ms: float
    service_latencies: dict


class HealthResponse(BaseModel):
    status: str
    service: str
