import asyncio
import os
import re

from PIL import Image

_model = None
_tokenizer = None
_lock = asyncio.Lock()

MOCK_MODELS = os.getenv("MOCK_MODELS", "false").lower() == "true"

EMOTION_WORDS = [
    "happy", "sad", "angry", "surprised", "fearful",
    "disgusted", "neutral", "confused", "excited", "anxious", "tired",
]


def _load_real_vision_model() -> None:  # pragma: no cover - needs a GPU host
    """Pull the fine-tuned vision model onto the GPU.

    Split out of `load_vision_model` because it needs CUDA and the unsloth
    stack, neither of which exists in CI; keeping it separate lets the
    mock-mode branch be measured honestly.
    """
    global _model, _tokenizer
    model_id = os.getenv("MODEL_ID", "seasalt29/imageModelBig")
    load_in_4bit = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
    print(f"Loading emotion detection model ({model_id})...")
    from unsloth import FastVisionModel  # noqa: PLC0415

    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    _model = model
    _tokenizer = tokenizer
    print("Vision model loaded successfully.")


def load_vision_model():
    if MOCK_MODELS:
        print("[MOCK] Vision model not loaded (MOCK_MODELS=true)")
        return
    _load_real_vision_model()


def extract_emotion(decoded: str) -> tuple[str, str]:
    """Pull a single emotion word out of a raw model decode.

    Split out of `_sync_detect` so it can be tested without a GPU: this is
    where the model's free-form text gets forced back onto the fixed label
    set, and it is the part most likely to drift.

    Returns (emotion, cleaned_decode); falls back to "neutral" when no known
    emotion word appears.
    """
    cleaned = re.sub(r"<\|.*?\|>", "", decoded).strip().lower()

    # Chat templates echo the prompt, which itself lists every emotion word.
    # Only the text after the assistant turn reflects the model's answer.
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


def _run_real_inference(image: Image.Image) -> tuple[str, str]:  # pragma: no cover - needs a GPU host
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

    return extract_emotion(_tokenizer.decode(output[0]))


def _sync_detect(image: Image.Image) -> tuple[str, str]:
    """Blocking inference — call via run_in_executor."""
    if MOCK_MODELS:
        return "neutral", "[mock response]"
    return _run_real_inference(image)


async def detect_emotion(image: Image.Image) -> tuple[str, str]:
    async with _lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_detect, image)


def is_model_loaded() -> bool:
    if MOCK_MODELS:
        return True
    return _model is not None


def gpu_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.is_available()
    except ImportError:
        return False
