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


def test_conflicting_prompt_names_both_emotions_and_asks_which_is_true():
    prompt = create_prompt("I'm fine", "happy", is_conflicting=True, text_emotion="sad")
    assert "happy" in prompt
    assert "sad" in prompt
    assert "?" in prompt


def test_non_conflicting_prompt_does_not_mention_a_conflict():
    prompt = create_prompt("I feel low", "sad", is_conflicting=False)
    assert "conflict" not in prompt.lower()


def test_prompt_leaves_the_response_slot_empty_for_the_model():
    prompt = create_prompt("hello", "neutral")
    assert prompt.rstrip().endswith("### Response:")


def test_prompt_defaults_to_non_conflicting():
    assert "conflict" not in create_prompt("hello", "neutral").lower()


def test_user_text_lands_in_the_input_section():
    prompt = create_prompt("my dog died", "sad")
    input_block = prompt.split("### Input:")[1].split("### Response:")[0]
    assert "my dog died" in input_block


def test_parse_response_keeps_multiline_replies_intact():
    parsed = parse_response("### Response:\nline one\nline two")
    assert parsed == "line one\nline two"


def test_parse_response_only_reads_the_first_response_marker():
    parsed = parse_response("### Instruction:\nfoo\n### Response:\nthe reply")
    assert parsed.startswith("the reply")


def test_parse_response_on_an_empty_decode_returns_the_fallback():
    assert "apologize" in parse_response("")


def test_parse_response_trims_surrounding_whitespace():
    assert parse_response("### Response:\n\n   hello   \n\n") == "hello"
