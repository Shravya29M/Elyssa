FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# REQUIREMENTS=requirements-ci.txt builds a CPU-only image for mock/demo mode
ARG REQUIREMENTS=requirements.txt
COPY requirements.txt requirements-ci.txt ./
RUN pip3 install --no-cache-dir -r ${REQUIREMENTS}

COPY app/ ./app/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HOME=/model-cache

EXPOSE 8003

# 1 worker: GPU tensors are not safe to fork across processes
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
