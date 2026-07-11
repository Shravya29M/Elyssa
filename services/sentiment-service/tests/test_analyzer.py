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
