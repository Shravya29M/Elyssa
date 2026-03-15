from pydantic import BaseModel


class SentimentRequest(BaseModel):
    text: str
    request_id: str


class SentimentResponse(BaseModel):
    request_id: str
    text_emotion: str
    compound_score: float
    latency_ms: float


class ConflictRequest(BaseModel):
    facial_emotion: str
    text: str
    request_id: str


class ConflictResponse(BaseModel):
    request_id: str
    is_conflicting: bool
    facial_emotion: str
    text_emotion: str
    facial_category: str
    text_category: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
