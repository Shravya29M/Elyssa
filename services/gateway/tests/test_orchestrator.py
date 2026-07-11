import pytest

from app.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


async def failing_coro():
    raise RuntimeError("fail")


async def success_coro():
    return "ok"


@pytest.mark.asyncio
async def test_circuit_starts_closed():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=30)

    for _ in range(2):
        try:
            await cb.call(failing_coro)
        except RuntimeError:
            pass

    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_resets_on_success():
    cb = CircuitBreaker("test", failure_threshold=5, recovery_timeout=30)

    result = await cb.call(success_coro)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


@pytest.mark.asyncio
async def test_open_circuit_raises_immediately():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=9999)

    try:
        await cb.call(failing_coro)
    except RuntimeError:
        pass

    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitOpenError):
        await cb.call(failing_coro)


@pytest.mark.asyncio
async def test_circuit_recovers_after_timeout():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0)

    try:
        await cb.call(failing_coro)
    except RuntimeError:
        pass

    assert cb.state == CircuitState.OPEN

    # recovery_timeout=0 → next call goes half-open and succeeds
    result = await cb.call(success_coro)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED
