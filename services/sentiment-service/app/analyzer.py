import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

_sia = SentimentIntensityAnalyzer()

POSITIVE_EMOTIONS = {"happy", "excited"}
NEGATIVE_EMOTIONS = {"sad", "angry", "fearful", "disgusted", "anxious", "tired"}
NEUTRAL_EMOTIONS = {"neutral", "surprised", "confused"}


def analyze_text_sentiment(text: str) -> tuple[str, float]:
    """Returns (emotion_category, compound_score)."""
    scores = _sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.5:
        emotion = "happy"
    elif compound >= 0.1:
        emotion = "excited"
    elif compound <= -0.5:
        emotion = "sad"
    elif compound <= -0.1:
        emotion = "anxious"
    else:
        emotion = "neutral"

    return emotion, compound


def _categorize(emotion: str) -> str:
    if emotion in POSITIVE_EMOTIONS:
        return "positive"
    if emotion in NEGATIVE_EMOTIONS:
        return "negative"
    return "neutral"


def detect_conflict(facial_emotion: str, text: str) -> dict:
    """Detect conflict between facial emotion and text sentiment."""
    text_emotion, compound = analyze_text_sentiment(text)
    facial_category = _categorize(facial_emotion)
    text_category = _categorize(text_emotion)

    is_conflicting = (
        (facial_category == "positive" and text_category == "negative")
        or (facial_category == "negative" and text_category == "positive")
    )

    return {
        "is_conflicting": is_conflicting,
        "facial_emotion": facial_emotion,
        "text_emotion": text_emotion,
        "facial_category": facial_category,
        "text_category": text_category,
        "compound_score": compound,
    }
