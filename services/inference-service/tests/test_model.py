import os

os.environ["MOCK_MODELS"] = "true"

import importlib  # noqa: E402

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from app import model as model_mod  # noqa: E402
from app.model import EMOTION_WORDS, extract_emotion  # noqa: E402

# --------------------------------------------------------------------------
# extract_emotion — the model's free text forced onto a fixed label set
# --------------------------------------------------------------------------


@pytest.mark.parametrize("word", EMOTION_WORDS)
def test_every_supported_emotion_round_trips(word):
    emotion, _ = extract_emotion(f"assistant {word}")
    assert emotion == word


def test_special_tokens_are_stripped():
    emotion, cleaned = extract_emotion("<|begin_of_text|>assistant sad<|end_of_text|>")
    assert emotion == "sad"
    assert "<|" not in cleaned


def test_decode_is_lowercased_and_trimmed():
    _, cleaned = extract_emotion("  ASSISTANT Happy  ")
    assert cleaned == "assistant happy"


def test_only_the_assistant_turn_is_searched():
    """The prompt lists every emotion word, so searching the whole decode
    would match the instruction rather than the answer."""
    decoded = (
        "user: respond with exactly one word from: happy, sad, angry, surprised. "
        "assistant: tired"
    )
    emotion, _ = extract_emotion(decoded)
    assert emotion == "tired"


def test_falls_back_to_the_whole_decode_when_there_is_no_assistant_turn():
    emotion, _ = extract_emotion("the person looks angry")
    assert emotion == "angry"


def test_unknown_wording_falls_back_to_neutral():
    emotion, cleaned = extract_emotion("assistant: i am not sure what they feel")
    assert emotion == "neutral"
    assert cleaned  # the raw decode is still returned for debugging


def test_empty_decode_falls_back_to_neutral():
    assert extract_emotion("")[0] == "neutral"


def test_substrings_do_not_count_as_matches():
    """'unhappy' contains 'happy'; a word-boundary match must reject it."""
    emotion, _ = extract_emotion("assistant: unhappiness")
    assert emotion == "neutral"


def test_first_listed_emotion_wins_when_several_appear():
    emotion, _ = extract_emotion("assistant: happy, maybe sad")
    assert emotion == EMOTION_WORDS[0]


def test_punctuation_around_the_word_is_tolerated():
    assert extract_emotion("assistant: 'fearful.'")[0] == "fearful"


def test_emotion_word_list_has_no_duplicates():
    assert len(set(EMOTION_WORDS)) == len(EMOTION_WORDS)


def test_neutral_is_a_supported_label():
    """The fallback value must itself be a valid label."""
    assert "neutral" in EMOTION_WORDS


# --------------------------------------------------------------------------
# mock-mode behaviour, which is what CI and the free-tier demo run
# --------------------------------------------------------------------------


def test_load_vision_model_is_a_no_op_in_mock_mode():
    model_mod.load_vision_model()  # must not raise or import unsloth
    assert model_mod._model is None


def test_model_reports_itself_loaded_in_mock_mode():
    assert model_mod.is_model_loaded() is True


def test_sync_detect_returns_a_neutral_mock():
    emotion, raw = model_mod._sync_detect(Image.new("RGB", (8, 8)))
    assert emotion == "neutral"
    assert "mock" in raw


async def test_detect_emotion_runs_off_the_event_loop():
    emotion, _ = await model_mod.detect_emotion(Image.new("RGB", (8, 8)))
    assert emotion == "neutral"


def test_is_model_loaded_is_false_before_a_real_load(monkeypatch):
    monkeypatch.setattr(model_mod, "MOCK_MODELS", False)
    monkeypatch.setattr(model_mod, "_model", None)
    assert model_mod.is_model_loaded() is False


def test_is_model_loaded_is_true_once_a_real_model_is_set(monkeypatch):
    monkeypatch.setattr(model_mod, "MOCK_MODELS", False)
    monkeypatch.setattr(model_mod, "_model", object())
    assert model_mod.is_model_loaded() is True


def test_gpu_available_is_false_without_torch(monkeypatch):
    """CPU-only deploys have no torch at all; that must report False, not crash."""
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert model_mod.gpu_available() is False


def test_mock_models_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("MOCK_MODELS", "false")
    reloaded = importlib.reload(model_mod)
    assert reloaded.MOCK_MODELS is False
    monkeypatch.setenv("MOCK_MODELS", "true")
    importlib.reload(model_mod)
