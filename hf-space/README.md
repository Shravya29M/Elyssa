---
title: Elyssa – AI Mental Health Assistant
emoji: 💬
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "5.20.1"
app_file: app.py
pinned: true
license: apache-2.0
short_description: Multimodal mental health assistant — facial emotion + text
---

# Elyssa – AI Mental Health Assistant

Multimodal mental health assistant that combines **facial emotion detection**
(fine-tuned Llama-3.2-11B Vision) with **text sentiment analysis** (NLTK VADER)
to generate empathetic counseling responses (fine-tuned Llama-3.2-3B on
MentalChat16K).

Indian Patent No. 202541093582A.

This Space is the single-process demo build. The production architecture runs
as five independently scalable FastAPI microservices on Kubernetes with
autoscaling, circuit breakers, and Prometheus observability — source at
[github.com/Shravya29M/Elyssa](https://github.com/Shravya29M/Elyssa).

## Deploying this Space

```bash
# from the repo root
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli repo create elyssa --repo-type space --space_sdk gradio
huggingface-cli upload <your-username>/elyssa hf-space/ . --repo-type space
```

Hardware: needs a GPU Space (the 11B vision model in 4-bit + the 3B model need
roughly 14GB VRAM). ZeroGPU (free with HF Pro) or an L40S/A10G paid Space both
work; on ZeroGPU the models load lazily on the first message.
