import os

os.environ["MOCK_MODELS"] = "true"

import asyncio  # noqa: E402

from app import model as model_mod  # noqa: E402
from app.prompt import parse_response  # noqa: E402


def test_load_language_model_is_a_no_op_in_mock_mode():
    model_mod.load_language_model()  # must not import unsloth or touch CUDA
    assert model_mod._model is None


def test_model_reports_itself_loaded_in_mock_mode():
    assert model_mod.is_model_loaded() is True


def test_sync_generate_returns_a_parseable_mock():
    raw = model_mod._sync_generate("prompt", 16)
    assert "### Response:" in raw
    # The mock has to survive the same parser the real decode goes through.
    assert parse_response(raw)


async def test_generate_runs_off_the_event_loop():
    raw = await model_mod.generate("prompt", 16)
    assert "### Response:" in raw


async def test_generate_serialises_concurrent_callers():
    """A single GPU can only run one generation at a time; the lock enforces
    that, so concurrent requests must still all complete."""
    results = await asyncio.gather(*(model_mod.generate("p", 8) for _ in range(5)))
    assert len(results) == 5
    assert all("### Response:" in r for r in results)


async def test_generate_defaults_to_512_new_tokens():
    captured = {}

    def spy(prompt, max_new_tokens):
        captured["max_new_tokens"] = max_new_tokens
        return "### Response:\nok"

    original = model_mod._sync_generate
    model_mod._sync_generate = spy
    try:
        await model_mod.generate("p")
    finally:
        model_mod._sync_generate = original
    assert captured["max_new_tokens"] == 512


def test_is_model_loaded_is_false_before_a_real_load(monkeypatch):
    monkeypatch.setattr(model_mod, "MOCK_MODELS", False)
    monkeypatch.setattr(model_mod, "_model", None)
    assert model_mod.is_model_loaded() is False


def test_is_model_loaded_is_true_once_a_real_model_is_set(monkeypatch):
    monkeypatch.setattr(model_mod, "MOCK_MODELS", False)
    monkeypatch.setattr(model_mod, "_model", object())
    assert model_mod.is_model_loaded() is True


def test_gpu_available_is_false_without_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert model_mod.gpu_available() is False
