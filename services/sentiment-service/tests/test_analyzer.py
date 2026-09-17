from app.analyzer import analyze_text_sentiment, detect_conflict


def test_positive_sentiment():
    emotion, score = analyze_text_sentiment("I feel so happy and wonderful today!")
    assert emotion in ("happy", "excited")
    assert score > 0.1


def test_negative_sentiment():
    emotion, score = analyze_text_sentiment("I am devastated and feel terrible.")
    assert emotion in ("sad", "anxious")
    assert score < -0.1


def test_neutral_sentiment():
    emotion, score = analyze_text_sentiment("I went to the store.")
    assert emotion == "neutral"


def test_conflict_positive_face_negative_text():
    result = detect_conflict("happy", "I feel terrible and hopeless today.")
    assert result["is_conflicting"] is True
    assert result["facial_category"] == "positive"
    assert result["text_category"] == "negative"


def test_no_conflict_matching_emotions():
    result = detect_conflict("happy", "I am so excited and joyful!")
    assert result["is_conflicting"] is False


def test_no_conflict_neutral():
    result = detect_conflict("neutral", "The weather is okay.")
    assert result["is_conflicting"] is False


def test_mental_health_phrases_score_negative():
    # Base VADER scores the first two mildly positive ("down" is not in its
    # lexicon, "overwhelmed" is +0.2); the lexicon overlay makes all three negative
    for text in (
        "I've been feeling really down lately.",
        "I'm feeling overwhelmed with my workload.",
        "I'm so stressed and anxious about everything.",
    ):
        emotion, score = analyze_text_sentiment(text)
        assert score < -0.1, text
        assert emotion in ("sad", "anxious"), text


def test_conflict_happy_face_feeling_down():
    result = detect_conflict("happy", "I've been feeling really down lately.")
    assert result["is_conflicting"] is True
