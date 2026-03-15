import asyncio
import os
import re

import torch
from PIL import Image

_model = None
_tokenizer = None
_lock = asyncio.Lock()

MOCK_MODELS = os.getenv("MOCK_MODELS", "false").lower() == "true"

EMOTION_WORDS = [
    "happy", "sad", "angry", "surprised", "fearful",
    "disgusted", "neutral", "confused", "excited", "anxious", "tired",
]


def load_vision_model():
    global _model, _tokenizer
    if MOCK_MODELS:
        print("[MOCK] Vision model not loaded (MOCK_MODELS=true)")
        return
    print("Loading emotion detection model (seasalt29/imageModelBig)...")
    from unsloth import FastVisionModel  # noqa: PLC0415
    model, tokenizer = FastVisionModel.from_pretrained(
        "seasalt29/imageModelBig",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    _model = model
    _tokenizer = tokenizer
    print("Vision model loaded successfully.")


def _sync_detect(image: Image.Image) -> tuple[str, str]:
    """Blocking inference — call via run_in_executor."""
    if MOCK_MODELS:
        return "neutral", "[mock response]"

    from transformers import TextStreamer  # noqa: PLC0415

    instruction = (
        "Analyze this facial expression and identify the emotion. "
        "Respond with EXACTLY ONE WORD from this list: "
        "happy, sad, angry, surprised, fearful, disgusted, neutral, "
        "confused, excited, anxious, tired. "
        "Your entire response should be just that one word."
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    input_text = _tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(_tokenizer, skip_prompt=True)
    output = _model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "128")),
        use_cache=True,
        temperature=float(os.getenv("INFERENCE_TEMPERATURE", "0.1")),
        min_p=0.05,
    )

    decoded = _tokenizer.decode(output[0])
    cleaned = re.sub(r"<\|.*?\|>", "", decoded).strip().lower()

    assistant_response = ""
    if "assistant" in cleaned:
        parts = cleaned.split("assistant")
        if len(parts) > 1:
            assistant_response = parts[1].strip()

    text_to_search = assistant_response if assistant_response else cleaned

    for word in EMOTION_WORDS:
        if re.search(r"\b" + word + r"\b", text_to_search):
            return word, cleaned

    return "neutral", cleaned


async def detect_emotion(image: Image.Image) -> tuple[str, str]:
    async with _lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_detect, image)


def is_model_loaded() -> bool:
    if MOCK_MODELS:
        return True
    return _model is not None
