"""Elyssa – free-tier demo build.

Runs the real Elyssa pipeline logic (VADER text sentiment, emotion-conflict
detection, prompt construction) on CPU-only hardware. The two GPU models are
simulated: facial emotion comes from a selector instead of the fine-tuned
Llama-3.2-11B Vision model, and counseling replies are curated templates
instead of live Llama-3.2-3B generations.

Full system (5 FastAPI microservices on Kubernetes, real models):
https://github.com/Shravya29M/Elyssa
"""

import os

import gradio as gr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# Same mental-health lexicon overlay as the production sentiment service:
# base VADER scores "feeling down" / "overwhelmed" as mildly positive.
sia.lexicon.update({
    "down": -1.2,
    "overwhelmed": -1.8,
    "overwhelming": -1.5,
    "stressed": -1.8,
    "anxious": -1.9,
    "anxiety": -1.6,
    "lonely": -2.0,
    "loneliness": -1.8,
    "hopeless": -2.5,
    "worthless": -2.5,
    "numb": -1.5,
    "drained": -1.5,
    "exhausted": -1.6,
    "burnout": -1.8,
    "insomnia": -1.4,
    "panic": -2.0,
})

POSITIVE = {"happy", "excited"}
NEGATIVE = {"sad", "angry", "fearful", "disgusted", "anxious", "tired"}

EMOTION_CHOICES = [
    "neutral", "happy", "sad", "angry", "surprised", "fearful",
    "disgusted", "confused", "excited", "anxious", "tired",
]


def analyze_text_sentiment(text: str) -> str:
    compound = sia.polarity_scores(text)["compound"]
    if compound >= 0.5:
        return "happy"
    if compound >= 0.1:
        return "excited"
    if compound <= -0.5:
        return "sad"
    if compound <= -0.1:
        return "anxious"
    return "neutral"


def _categorize(emotion: str) -> str:
    if emotion in POSITIVE:
        return "positive"
    if emotion in NEGATIVE:
        return "negative"
    return "neutral"


def detect_conflict(facial_emotion: str, text: str):
    text_emotion = analyze_text_sentiment(text)
    fc, tc = _categorize(facial_emotion), _categorize(text_emotion)
    is_conflicting = (fc == "positive" and tc == "negative") or (
        fc == "negative" and tc == "positive"
    )
    return is_conflicting, text_emotion


RESPONSES = {
    "positive": (
        "It sounds like things are going well for you right now, and I'm genuinely glad "
        "to hear it. Moments like these are worth savoring — they can also be a good time "
        "to reflect on what's contributing to this feeling so you can return to it during "
        "harder stretches. Is there anything on your mind you'd like to explore further?"
    ),
    "negative": (
        "Thank you for sharing that with me — it takes courage to put difficult feelings "
        "into words. What you're experiencing sounds genuinely hard, and it's completely "
        "understandable to feel this way. You don't have to carry it alone. Could you tell "
        "me a little more about when these feelings tend to be strongest? Sometimes noticing "
        "the pattern is the first step toward easing it."
    ),
    "neutral": (
        "I hear you. Sometimes feelings aren't clearly one thing or another, and that's "
        "okay — you don't need to have it all figured out before talking about it. I'm here "
        "to listen. What's been occupying your thoughts most lately?"
    ),
}


def generate_reply(user_text: str, facial_emotion: str):
    is_conflicting, text_emotion = detect_conflict(facial_emotion, user_text)

    if is_conflicting:
        reply = (
            f"I notice that your facial expression appears {facial_emotion}, but your "
            f"message suggests you might be feeling {text_emotion}. Could you tell me "
            f"which better reflects how you're actually feeling right now? "
            "Whichever it is, I want to make sure my support meets you where you truly are."
        )
        emotion_display = f"{facial_emotion} (face) vs {text_emotion} (text) — conflicting"
    else:
        category = _categorize(text_emotion if text_emotion != "neutral" else facial_emotion)
        reply = RESPONSES[category]
        emotion_display = facial_emotion

    return reply, emotion_display


def add_text(history, text):
    if not text.strip():
        return history, ""
    return history + [(text, None)], ""


def bot(history, facial_emotion):
    user_message = history[-1][0]
    reply, emotion_display = generate_reply(user_message, facial_emotion)
    history[-1] = (
        user_message,
        f"{reply}\n\n[Detected emotion: {emotion_display}]",
    )
    return history


with gr.Blocks(theme=gr.themes.Soft(), title="Elyssa Demo") as demo:
    gr.Markdown(
        """
        # Elyssa – AI Mental Health Assistant · Live Demo

        Elyssa combines **facial emotion detection** with **text sentiment analysis** to
        deliver counseling responses that address what you *show* as well as what you *say* —
        including gently flagging when the two disagree.

        > ⚡ **Demo mode.** This free-tier instance runs the real routing, sentiment, and
        > conflict-detection pipeline, but the two GPU models are simulated: pick a facial
        > emotion below instead of the fine-tuned Llama-3.2-11B Vision model, and replies are
        > curated instead of live Llama-3.2-3B generations. The full system — five FastAPI
        > microservices on Kubernetes with autoscaling — is on
        > [GitHub](https://github.com/Shravya29M/Elyssa).
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            facial = gr.Dropdown(
                EMOTION_CHOICES,
                value="neutral",
                label="Simulated facial emotion",
                info="Stands in for the webcam + vision model. Try 'happy' with a sad message to trigger conflict detection.",
            )
            gr.Markdown(
                """
                **Try the conflict detector:**
                1. Set facial emotion to *happy*
                2. Send: *"I've been feeling really down lately."*

                Elyssa will notice the mismatch and ask which is true.
                """
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=460,
                show_label=False,
                show_copy_button=True,
                avatar_images=("👤", "🤖"),
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message and press Enter...",
                    container=False,
                    scale=6,
                    show_label=False,
                )
                send_btn = gr.Button("Send", scale=1)
                clear = gr.Button("Clear", scale=1)

    msg.submit(add_text, [chatbot, msg], [chatbot, msg]).then(
        bot, [chatbot, facial], chatbot
    )
    send_btn.click(add_text, [chatbot, msg], [chatbot, msg]).then(
        bot, [chatbot, facial], chatbot
    )
    clear.click(lambda: [], outputs=[chatbot])

    gr.Examples(
        [
            "I passed all my exams right now. I feel so happy and content.",
            "I'm feeling overwhelmed with my workload lately.",
            "I had a fight with my friend and I'm not sure what to do.",
            "I've been feeling down for the past few weeks.",
        ],
        inputs=msg,
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
