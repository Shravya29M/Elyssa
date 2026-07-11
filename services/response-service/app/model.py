import asyncio
import os

_model = None
_tokenizer = None
_lock = asyncio.Lock()

MOCK_MODELS = os.getenv("MOCK_MODELS", "false").lower() == "true"


def load_language_model():
    global _model, _tokenizer
    if MOCK_MODELS:
        print("[MOCK] Language model not loaded (MOCK_MODELS=true)")
        return
    model_id = os.getenv("MODEL_ID", "seasalt29/model3")
    print(f"Loading counseling model ({model_id})...")
    from unsloth import FastLanguageModel  # noqa: PLC0415

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "512")),
        dtype=None,
        load_in_4bit=False,
    )
    model = FastLanguageModel.for_inference(model)
    _model = model
    _tokenizer = tokenizer
    print("Language model loaded successfully.")


def _sync_generate(prompt: str, max_new_tokens: int) -> str:
    if MOCK_MODELS:
        return "### Response:\nI hear you and I'm here to help. [mock response]"

    inputs = _tokenizer([prompt], return_tensors="pt").to("cuda")
    output = _model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    return _tokenizer.decode(output[0])


async def generate(prompt: str, max_new_tokens: int = 512) -> str:
    async with _lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_generate, prompt, max_new_tokens)


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
