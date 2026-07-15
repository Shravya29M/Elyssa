# Elyssa – AI Mental Health Assistant

> Indian Patent No. 202541093582A

Elyssa is a production-grade, multi-modal AI mental health assistant that combines real-time facial emotion detection with text sentiment analysis to deliver empathetic, context-aware counseling responses. The system is architected as independently scalable microservices deployed on Kubernetes.

---

## Try It

- **🌐 Live demo** — [elyssa-ijm1.onrender.com](https://elyssa-ijm1.onrender.com) — free-tier build with the real sentiment + conflict-detection pipeline (facial emotion is a dropdown selector standing in for the GPU vision model; source in [`demo/`](demo/)). One-click redeploy: [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Shravya29M/Elyssa)
- **🤗 GPU demo** — full-model single-process build ready to deploy from [`hf-space/`](hf-space/) (needs an HF PRO / GPU Space)
- **Run locally without a GPU** (full 5-service stack, mocked LLMs, ~2 min):

  ```bash
  git clone https://github.com/Shravya29M/Elyssa.git && cd Elyssa
  docker compose -f docker-compose.mock.yml up --build
  # open http://localhost:7860
  ```

---

## Architecture

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                   Kubernetes Cluster                     │
                    │                                                          │
  User Browser      │  ┌──────────────┐      ┌─────────────────────────────┐  │
  ─────────────     │  │    NGINX     │      │   Frontend Service           │  │
  Webcam + Chat ───►│  │   Ingress   │─────►│   Gradio UI  · Port 7860     │  │
                    │  └──────────────┘      │   HPA: 1–4 replicas         │  │
                    │                        └──────────────┬───────────────┘  │
                    │                                       │                  │
                    │                           POST /chat  │ (image_b64 +    │
                    │                                       │  user_text)      │
                    │                                       ▼                  │
                    │                        ┌──────────────────────────────┐  │
                    │                        │   Gateway Service             │  │
                    │                        │   Orchestrator · Port 8000    │  │
                    │                        │   Circuit Breaker + Retries   │  │
                    │                        │   HPA: 2–8 replicas           │  │
                    │                        └──────┬───────────────┬────────┘  │
                    │                               │               │           │
                    │              ┌────────────────┘               └─────┐    │
                    │              ▼                                       ▼    │
                    │  ┌──────────────────────┐         ┌──────────────────────┐ │
                    │  │  Inference Service   │         │  Response Service    │ │
                    │  │  Llama 3.2-11B Vision│         │  Llama-3.2-3B       │ │
                    │  │  Emotion Detection   │         │  Mental Health LLM   │ │
                    │  │  Port 8001 · GPU     │         │  Port 8003 · GPU     │ │
                    │  │  HPA: 1–3 replicas   │         │  HPA: 1–3 replicas   │ │
                    │  └──────────────────────┘         └──────────────────────┘ │
                    │                               │                          │
                    │                               ▼                          │
                    │                  ┌─────────────────────────┐             │
                    │                  │  Sentiment Service       │             │
                    │                  │  NLTK VADER Analysis     │             │
                    │                  │  Conflict Detection      │             │
                    │                  │  Port 8002 · CPU only    │             │
                    │                  │  HPA: 2–10 replicas      │             │
                    │                  └─────────────────────────┘             │
                    └──────────────────────────────────────────────────────────┘
```

### Request Flow

Every chat message follows this pipeline:

```
POST /chat (image_b64 + user_text)
  │
  ├─ 1. Inference Service  → facial_emotion       (30s timeout, 2 attempts)
  │
  ├─ 2. Sentiment Service  → conflict detection   (5s timeout, 3 attempts)
  │       ├─ NLTK VADER text sentiment
  │       └─ Conflict: face vs text emotion mismatch
  │
  └─ 3. Response Service   → counseling response  (60s timeout, 1 attempt)
          ├─ Alpaca-format prompt with emotion context
          └─ Fine-tuned Llama-3.2-3B generation (up to 512 tokens)
```

**Fault tolerance:**
- **Inference circuit open** → fall back to `facial_emotion="neutral"`, continue
- **Sentiment circuit open** → fall back to `is_conflicting=False`, continue
- **Response circuit open** → return HTTP 503 with user-friendly message

---

## Services

| Service | Port | Tech | GPU | HPA |
|---|---|---|---|---|
| **inference-service** | 8001 | FastAPI + Llama-3.2-11B Vision | Yes (12Gi VRAM) | 1–3 pods |
| **sentiment-service** | 8002 | FastAPI + NLTK VADER | No | 2–10 pods |
| **response-service** | 8003 | FastAPI + Llama-3.2-3B | Yes (8Gi VRAM) | 1–3 pods |
| **gateway** | 8000 | FastAPI orchestrator | No | 2–8 pods |
| **frontend** | 7860 | Gradio | No | 1–4 pods |

---

## Models

| Model | HuggingFace ID | Base | Fine-tuned on |
|---|---|---|---|
| Emotion Detection | [seasalt29/imageModelBig](https://huggingface.co/seasalt29/imageModelBig) | Llama-3.2-11B-Vision-Instruct | Tukey Human Emotion Dataset |
| Mental Health LLM | [seasalt29/model3](https://huggingface.co/seasalt29/model3) | Llama-3.2-3B | MentalChat16K |

### Evaluation

The counseling model was evaluated against its base model on 100 prompts from
[Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
(real counselor answers, unseen during training), identical alpaca prompt
template, greedy decoding, fixed seed. Reproducible script and full results:
[`eval/rouge_eval.py`](eval/rouge_eval.py), [`eval/rouge_results.json`](eval/rouge_results.json).

| ROUGE F1 | Base Llama-3.2-3B | Fine-tuned (model3) | Change |
|---|---|---|---|
| ROUGE-1 | 0.3008 | 0.3150 | **+4.7%** |
| ROUGE-2 | 0.0472 | 0.0446 | -5.4% |
| ROUGE-L | 0.1528 | 0.1479 | -3.2% |

Fine-tuning moved word choice measurably closer to real counselor responses
(ROUGE-1); phrase-level metrics were flat to slightly lower, which is expected
when a model stops parroting prompt phrasing and answers in its own words.
Training notebooks: [`Model3.ipynb`](Model3.ipynb) (counseling LLM) and
[`FER_MODEL_LATEST.ipynb`](FER_MODEL_LATEST.ipynb) (facial emotion recognition).

---

## API Reference

### Gateway — `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_text": "I feel overwhelmed with work lately.",
    "image_b64": "<base64-encoded-JPEG>"
  }'
```

Response:
```json
{
  "request_id": "abc-123",
  "response_text": "I hear that you're feeling overwhelmed...",
  "facial_emotion": "anxious",
  "text_emotion": "anxious",
  "is_conflicting": false,
  "emotion_display": "anxious",
  "total_latency_ms": 4821.3,
  "service_latencies": {
    "inference_ms": 3200.1,
    "sentiment_ms": 2.4,
    "response_ms": 1618.8
  }
}
```

### Inference Service — `POST /detect-emotion`

```bash
curl -X POST http://localhost:8001/detect-emotion \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "<base64-JPEG>", "request_id": "req-001"}'
```

### Sentiment Service — `POST /detect-conflict`

```bash
curl -X POST http://localhost:8002/detect-conflict \
  -H "Content-Type: application/json" \
  -d '{"facial_emotion": "happy", "text": "I feel terrible today.", "request_id": "req-001"}'
```

### Response Service — `POST /generate-response`

```bash
curl -X POST http://localhost:8003/generate-response \
  -H "Content-Type: application/json" \
  -d '{
    "user_text": "I feel overwhelmed",
    "facial_emotion": "anxious",
    "text_emotion": "anxious",
    "is_conflicting": false,
    "request_id": "req-001"
  }'
```

### Health / Readiness (all services)

```bash
curl http://localhost:8001/health   # {"status": "ok", "model_loaded": true, ...}
curl http://localhost:8001/ready    # 200 OK when ready, 503 while loading
curl http://localhost:8001/metrics  # Prometheus text format
```

---

## Local Development

### Prerequisites

- Docker + Docker Compose
- NVIDIA GPU with CUDA 12.1+ (for inference and response services)
- `nvidia-container-toolkit` installed

### Run all services

```bash
git clone https://github.com/Shravya29M/Elyssa.git
cd Elyssa
docker compose up --build
```

Services available at:
- Frontend: http://localhost:7860
- Gateway API: http://localhost:8000
- Inference: http://localhost:8001
- Sentiment: http://localhost:8002
- Response: http://localhost:8003

> **Note:** First startup downloads ~22GB of model weights from HuggingFace. Subsequent starts use the cached `model-cache` Docker volume.

### Run without GPU (mock mode)

Runs the full 5-service stack on any machine — the two LLM services return
canned responses instead of loading model weights (no GPU, no 22GB download):

```bash
docker compose -f docker-compose.mock.yml up --build
```

### Run tests

```bash
cd services/sentiment-service && pytest tests/ -v
cd services/inference-service && MOCK_MODELS=true pytest tests/ -v
cd services/response-service  && MOCK_MODELS=true pytest tests/ -v
cd services/gateway           && pytest tests/ -v
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster with NVIDIA GPU nodes labeled `accelerator=nvidia-gpu`
- `kubectl` configured
- NGINX Ingress Controller installed

### Deploy

```bash
# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Create model cache storage
kubectl apply -f k8s/storage/ -n elyssa

# 3. Apply ConfigMaps
kubectl apply -f k8s/configmaps/ -n elyssa

# 4. Deploy services
kubectl apply -f k8s/deployments/ -n elyssa

# 5. Create internal services
kubectl apply -f k8s/services/ -n elyssa

# 6. Configure autoscaling
kubectl apply -f k8s/hpa/ -n elyssa

# 7. Configure ingress
kubectl apply -f k8s/ingress/ -n elyssa
```

### Verify

```bash
kubectl get pods -n elyssa
kubectl get hpa -n elyssa
kubectl logs -n elyssa deployment/gateway --tail=50
kubectl port-forward -n elyssa svc/frontend 7860:7860
```

---

## Observability

### Prometheus Metrics

All services expose `/metrics`. Key metrics:

| Metric | Service |
|---|---|
| `elyssa_inference_requests_total` | inference |
| `elyssa_inference_request_duration_seconds` | inference |
| `elyssa_sentiment_requests_total` | sentiment |
| `elyssa_response_requests_total` | response |
| `elyssa_gateway_requests_total` | gateway |
| `elyssa_gateway_request_duration_seconds` | gateway |

### Structured Logging

All services emit JSON logs with `request_id` for distributed tracing:

```json
{
  "event": "emotion_detected",
  "request_id": "abc-123",
  "emotion": "anxious",
  "latency_ms": 3201.4,
  "level": "info",
  "timestamp": "2025-03-15T10:22:01.123Z"
}
```

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):

1. **Lint & Test** — ruff + pytest (with `MOCK_MODELS=true`) for all services in parallel
2. **Build & Push** — Docker images pushed to GHCR on merge to `main`
3. **Validate K8s** — `kubectl apply --dry-run=client` on all manifests

---

## Repository Structure

```
elyssa/
├── services/
│   ├── inference-service/     # Port 8001 — Vision LLM emotion detection
│   ├── sentiment-service/     # Port 8002 — NLTK VADER + conflict detection
│   ├── response-service/      # Port 8003 — Counseling LLM generation
│   └── gateway/               # Port 8000 — Async orchestrator + circuit breaker
├── frontend/                  # Port 7860 — Gradio webcam + chat UI
├── hf-space/                  # Hugging Face Space demo build (single process)
├── k8s/
│   ├── namespace.yaml
│   ├── configmaps/            # Per-service environment configuration
│   ├── deployments/           # Kubernetes Deployments (GPU-aware)
│   ├── services/              # ClusterIP Services for internal discovery
│   ├── hpa/                   # HorizontalPodAutoscalers
│   ├── ingress/               # NGINX Ingress
│   └── storage/               # PVC for HuggingFace model cache
├── docker-compose.yml         # Full local development stack
├── .github/workflows/ci.yml   # CI/CD pipeline
└── app.py                     # Original monolith (preserved for reference)
```

---

## Patent

This system is protected under **Indian Patent No. 202541093582A**.
