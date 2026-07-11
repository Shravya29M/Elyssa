import os

os.environ["MOCK_MODELS"] = "true"

from app.prompt import create_prompt, parse_response


def test_create_prompt_non_conflicting():
    prompt = create_prompt("I feel anxious", "anxious", is_conflicting=False)
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:" in prompt
    assert "anxious" in prompt
    assert "I feel anxious" in prompt


def test_create_prompt_conflicting():
    prompt = create_prompt("I am great", "happy", is_conflicting=True, text_emotion="sad")
    assert "conflict" in prompt.lower()
    assert "happy" in prompt
    assert "sad" in prompt


def test_parse_response_extracts_after_marker():
    full = "### Instruction:\nblah\n### Input:\nfoo\n### Response:\nHello, I am here to help."
    result = parse_response(full)
    assert result == "Hello, I am here to help."


def test_parse_response_no_marker_returns_fallback():
    result = parse_response("No marker here at all.")
    assert "apologize" in result.lower()


def test_parse_response_strips_end_token():
    full = "### Response:\nHello world<|end_of_text|>"
    result = parse_response(full)
    assert "<|end_of_text|>" not in result
    assert "Hello world" in result
