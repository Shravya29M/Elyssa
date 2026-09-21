# Elyssa — working context

Multi-modal mental-health assistant: facial emotion detection plus text sentiment, combined to
produce context-aware counselling responses. Four Python microservices behind a gateway,
deployed on Kubernetes. Indian Patent No. 202541093582A.

The product idea that drives the architecture: when the face and the text **disagree**, that
conflict is the signal. The gateway detects it and asks the user which is closer to the truth
rather than guessing.

## Layout

```
services/
  gateway/            orchestrator.py (retries, breakers, fallbacks), main.py (/chat)
                      circuit_breaker.py
  inference-service/  vision model, /detect-emotion; model.py has extract_emotion
  sentiment-service/  VADER + mental-health lexicon, /analyze-sentiment /detect-conflict
  response-service/   counselling LLM, /generate-response; prompt.py builds Alpaca prompts
frontend/             Gradio app
demo/                 free-tier build (dropdown stands in for the GPU vision model)
hf-space/             single-process GPU build
k8s/                  deployments, services, hpa, ingress, configmaps, storage
eval/                 LLM-judge + ROUGE quality evals (no timing data)
```

## Commands

```bash
cd services/<service>
MOCK_MODELS=true pytest          # coverage on by default, per-service floor in pytest.ini
ruff check . --select E,F,W,I --ignore E501

docker compose -f docker-compose.mock.yml up --build   # full stack, no GPU, ~2 min
```

Everything runs with `MOCK_MODELS=true`. Nothing in the test suite needs a GPU.

## Verified metrics

Measured 2026-09-21. Re-derive with the command above; do not estimate.

| Service | Tests | Branch + line coverage | CI floor |
|---|---:|---:|---:|
| gateway | 54 | 99.47% | 95% |
| inference-service | 41 | 94.74% | 90% |
| response-service | 36 | 95.33% | 90% |
| sentiment-service | 26 | 100.00% | 100% |
| **total** | **157** | **~97%** | per service |

Floors live in each service's `pytest.ini` under `addopts`.

### Kubernetes HPA

| Service | min | max | Target metric | File |
|---|---:|---:|---|---|
| gateway | 2 | 8 | CPU 60% **+ memory 75%** | `k8s/hpa/gateway-hpa.yaml:11-25` |
| sentiment | 2 | 10 | CPU 60% | `k8s/hpa/sentiment-hpa.yaml:11-19` |
| inference | 1 | 3 | CPU 70% | `k8s/hpa/inference-hpa.yaml:11-19` |
| response | 1 | 3 | CPU 70% | `k8s/hpa/response-hpa.yaml:11-19` |
| frontend | 1 | 4 | CPU 70% | `k8s/hpa/frontend-hpa.yaml:11-19` |

Gateway is the only service scaling on two metrics.

## Known UNKNOWNs

- **No load test or performance benchmark exists.** No concurrent-user count, no requests/sec,
  no latency under failure. `git grep -in "requests_per_second\|rps\|concurrent users\|throughput\|p95"`
  returns nothing; no locust/k6/wrk/vegeta tooling is tracked. To get it: script k6 or locust
  against the gateway `/chat` endpoint with `MOCK_MODELS=true`, then repeat with a downstream
  service stopped to measure behavior under failure.
- **The latency numbers in `README.md` (`total_latency_ms: 4821.3`, `latency_ms: 3201.4`) are
  illustrative example payloads**, not measurements — note the placeholder `request_id: "abc-123"`
  and the 2025-03-15 timestamp. Do not cite them.
- `eval/` holds quality evals only (`generations.json`, `judge_results.json`, `rouge_results.json`).

## Resilience rules the tests pin down

These live in `services/gateway/` and are the most load-bearing logic in the system:

- **Retries differ per service**: inference 2 attempts, sentiment 3, response 1. Only
  `TimeoutException` and `ConnectError` are retried.
- **Breaker thresholds differ**: inference and sentiment take `CIRCUIT_BREAKER_THRESHOLD`
  (default 5); response trips at **3** with a 60s recovery.
- **Inference and sentiment degrade to neutral defaults** so a chat request still completes.
  A sentiment outage preserves the already-detected facial emotion rather than discarding it.
- **Response failures propagate** — there is no safe neutral default for a counselling reply.
- **An open response breaker returns 503, not 500.** A tripped breaker is a retryable outage,
  not a bug. This is asserted explicitly.
- An open breaker must not invoke the wrapped call at all (fail fast, no network).

## Gotchas

- Breakers are **module-level singletons** in `orchestrator.py`. Tests reset them via the
  autouse `reset_breakers` fixture in `services/gateway/tests/conftest.py`; without it one
  test's failures leak into the next.
- GPU-only code is split into its own functions — `_load_real_vision_model`,
  `_run_real_inference` (inference), `_load_real_language_model`, `_run_real_generation`
  (response) — each marked `# pragma: no cover`. This is deliberate: it keeps the mock branches
  that CI actually executes honestly measured instead of hiding a whole function behind a
  blanket pragma. Put new GPU code in those functions, not in the dispatchers.
- `extract_emotion` (`inference-service/app/model.py`) was pulled out of the CUDA path so label
  parsing is testable without a GPU. It only searches text **after** the assistant turn, because
  the prompt itself lists every emotion word. Matches are word-boundary anchored, so "unhappy"
  does not match "happy". Unknown wording falls back to `neutral`.
- `MOCK_MODELS` is read at import time. Tests that need it flipped must `importlib.reload`.
- `is_model_loaded()` returns `True` in mock mode even though `_model is None`.
- Readiness vs liveness: `/ready` returns 503 before the model loads, `/health` stays 200 —
  otherwise Kubernetes kills the pod during warm-up.

## Dependencies

Renovate (`renovate.json`): the shared FastAPI + observability baseline (fastapi, uvicorn, httpx,
pydantic, structlog, prometheus-client, tenacity) is grouped so the four services stay in
lockstep. The model stack (unsloth, torch, transformers, nltk, Pillow) is held back from
automerge — the checkpoints were trained against specific versions. Requires the Renovate
GitHub App on the repo.
