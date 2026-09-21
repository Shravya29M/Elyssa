import asyncio
import os

_model = None
_tokenizer = None
_lock = asyncio.Lock()

MOCK_MODELS = os.getenv("MOCK_MODELS", "false").lower() == "true"


def _load_real_language_model() -> None:  # pragma: no cover - needs a GPU host
    """Pull the fine-tuned counselling model onto the GPU.

    Split out of `load_language_model` because it needs CUDA and the unsloth
    stack, neither of which exists in CI; keeping it separate lets the
    mock-mode branch be measured honestly.
    """
    global _model, _tokenizer
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


def load_language_model():
    if MOCK_MODELS:
        print("[MOCK] Language model not loaded (MOCK_MODELS=true)")
        return
    _load_real_language_model()


def _run_real_generation(prompt: str, max_new_tokens: int) -> str:  # pragma: no cover - needs a GPU host
    inputs = _tokenizer([prompt], return_tensors="pt").to("cuda")
    output = _model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    return _tokenizer.decode(output[0])


def _sync_generate(prompt: str, max_new_tokens: int) -> str:
    if MOCK_MODELS:
        return "### Response:\nI hear you and I'm here to help. [mock response]"
    return _run_real_generation(prompt, max_new_tokens)


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
