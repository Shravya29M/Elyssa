import asyncio

import pytest

from app.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


async def failing_coro():
    raise RuntimeError("fail")


async def success_coro():
    return "ok"


async def test_circuit_starts_closed():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


async def test_successful_call_returns_result():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    assert await cb.call(success_coro) == "ok"
    assert cb.state == CircuitState.CLOSED


async def test_passes_through_args_and_kwargs():
    cb = CircuitBreaker("test")

    async def echo(a, b, *, c):
        return (a, b, c)

    assert await cb.call(echo, 1, 2, c=3) == (1, 2, 3)


async def test_failures_below_threshold_keep_circuit_closed():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(failing_coro)
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 2


async def test_circuit_opens_after_threshold():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(failing_coro)
    assert cb.state == CircuitState.OPEN


async def test_open_circuit_raises_immediately_without_calling_through():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=30)
    with pytest.raises(RuntimeError):
        await cb.call(failing_coro)

    calls = 0

    async def tracked():
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(CircuitOpenError):
        await cb.call(tracked)
    assert calls == 0, "an open circuit must not invoke the wrapped call"


async def test_circuit_resets_on_success():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(failing_coro)
    await cb.call(success_coro)
    assert cb.failure_count == 0
    assert cb.state == CircuitState.CLOSED


async def test_circuit_half_opens_after_recovery_timeout():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
    with pytest.raises(RuntimeError):
        await cb.call(failing_coro)
    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.02)
    assert await cb.call(success_coro) == "ok"
    assert cb.state == CircuitState.CLOSED


async def test_half_open_probe_failure_reopens_the_circuit():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
    with pytest.raises(RuntimeError):
        await cb.call(failing_coro)

    await asyncio.sleep(0.02)
    with pytest.raises(RuntimeError):
        await cb.call(failing_coro)
    assert cb.state == CircuitState.OPEN


async def test_circuit_open_error_names_the_breaker():
    cb = CircuitBreaker("inference", failure_threshold=1, recovery_timeout=30)
    with pytest.raises(RuntimeError):
        await cb.call(failing_coro)
    with pytest.raises(CircuitOpenError, match="inference"):
        await cb.call(failing_coro)


async def test_concurrent_failures_are_counted_exactly_once_each():
    cb = CircuitBreaker("test", failure_threshold=10, recovery_timeout=30)
    results = await asyncio.gather(
        *(cb.call(failing_coro) for _ in range(5)), return_exceptions=True
    )
    assert all(isinstance(r, RuntimeError) for r in results)
    assert cb.failure_count == 5
