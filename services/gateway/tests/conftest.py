import pytest

from app import orchestrator
from app.circuit_breaker import CircuitState


@pytest.fixture(autouse=True)
def reset_breakers():
    """The breakers are module-level singletons, so one test's failures would
    otherwise leak into the next."""
    yield
    for breaker in (
        orchestrator.inference_breaker,
        orchestrator.sentiment_breaker,
        orchestrator.response_breaker,
    ):
        breaker.failure_count = 0
        breaker.last_failure_time = 0.0
        breaker.state = CircuitState.CLOSED


@pytest.fixture
def fake_client(monkeypatch):
    """Install a stand-in for the module-level httpx.AsyncClient."""

    def _install(client):
        monkeypatch.setattr(orchestrator, "_client", client)
        return client

    return _install
