"""Elyssa – AI Mental Health Assistant (Hugging Face Space edition).

Single-process build of the Elyssa microservice system for hosting on a
Hugging Face GPU Space. Models are loaded lazily on the first request so
the app also starts on ZeroGPU hardware, where CUDA is only available
inside @spaces.GPU-decorated functions.

Full microservice deployment (FastAPI + Kubernetes):
https://github.com/Shravya29M/Elyssa
"""

import re
import threading

import gradio as gr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from PIL import Image

try:
    import spaces

    GPU = spaces.GPU
except ImportError:  # running outside HF Spaces

    def GPU(fn=None, **kwargs):
        if fn is None:
            return lambda f: f
        return fn


try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

VISION_MODEL_ID = "seasalt29/imageModelBig"
LANGUAGE_MODEL_ID = "seasalt29/model3"

EMOTION_WORDS = [
    "happy", "sad", "angry", "surprised", "fearful",
    "disgusted", "neutral", "confused", "excited", "anxious", "tired",
]

_models = {}
_load_lock = threading.Lock()


def _ensure_models():
    """Load both models once, on first request (GPU is attached by then)."""
    with _load_lock:
        if "vision" in _models:
            return
        from unsloth import FastLanguageModel, FastVisionModel

        print(f"Loading emotion detection model ({VISION_MODEL_ID})...")
        vision_model, vision_tokenizer = FastVisionModel.from_pretrained(
            VISION_MODEL_ID,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(vision_model)

        print(f"Loading counseling model ({LANGUAGE_MODEL_ID})...")
        language_model, language_tokenizer = FastLanguageModel.from_pretrained(
            model_name=LANGUAGE_MODEL_ID,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=False,
        )
        language_model = FastLanguageModel.for_inference(language_model)

        _models.update(
            vision=vision_model,
            vision_tok=vision_tokenizer,
            language=language_model,
            language_tok=language_tokenizer,
        )
        print("Both models loaded successfully.")


def detect_emotion(image) -> str:
    if image is None:
        return "neutral"
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    tokenizer = _models["vision_tok"]
    model = _models["vision"]

    instruction = (
        "Analyze this facial expression and identify the emotion. "
        "Respond with EXACTLY ONE WORD from this list: "
        "happy, sad, angry, surprised, fearful, disgusted, neutral, "
        "confused, excited, anxious, tired. "
        "Your entire response should be just that one word."
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image, input_text, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")

    output = model.generate(
        **inputs, max_new_tokens=128, use_cache=True, temperature=0.1, min_p=0.05
    )
    cleaned = re.sub(r"<\|.*?\|>", "", tokenizer.decode(output[0])).strip().lower()

    assistant_response = ""
    if "assistant" in cleaned:
        parts = cleaned.split("assistant")
        if len(parts) > 1:
            assistant_response = parts[1].strip()
    text_to_search = assistant_response if assistant_response else cleaned

    for word in EMOTION_WORDS:
        if re.search(r"\b" + word + r"\b", text_to_search):
            return word
    return "neutral"


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


POSITIVE = {"happy", "excited"}
NEGATIVE = {"sad", "angry", "fearful", "disgusted", "anxious", "tired"}


def _categorize(emotion: str) -> str:
    if emotion in POSITIVE:
        return "positive"
    if emotion in NEGATIVE:
        return "negative"
    return "neutral"


def detect_conflicting_emotions(facial_emotion: str, text: str):
    text_emotion = analyze_text_sentiment(text)
    fc, tc = _categorize(facial_emotion), _categorize(text_emotion)
    is_conflicting = (fc == "positive" and tc == "negative") or (
        fc == "negative" and tc == "positive"
    )
    return is_conflicting, facial_emotion, text_emotion


ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""


def create_prompt(user_text, facial_emotion, is_conflicting=False, text_emotion=None):
    if is_conflicting:
        instruction = (
            f"Detected facial emotion: {facial_emotion}, but text sentiment suggests: {text_emotion}. "
            f"There seems to be a conflict between facial expression and text content.\n"
            f"You are a helpful mental health counselling assistant. You've noticed a conflict between "
            f"the user's facial expression and their text content. Your first priority is to kindly "
            f"acknowledge this conflict and ask them to confirm their true emotional state.\n"
            f'Begin your response with: "I notice that your facial expression appears {facial_emotion}, '
            f"but your message suggests you might be feeling {text_emotion}. Could you tell me which "
            f'better reflects how you\'re actually feeling right now?"\n'
            f"After that introduction, briefly address their message with empathy, but keep the focus "
            f"on clarifying their emotional state before proceeding with more detailed support."
        )
    else:
        instruction = (
            f"Detected facial emotion: {facial_emotion}.\n\n"
            f"You are a helpful mental health counselling assistant that also considers the detected "
            f"facial emotion of the user, please answer the mental health questions based on the "
            f"patient's description. The assistant gives helpful, comprehensive, and appropriate "
            f"answers to the user's questions."
        )
    return ALPACA_TEMPLATE.format(instruction, user_text, "")


@GPU(duration=120)
def generate_response(user_text, user_image):
    _ensure_models()

    facial_emotion = detect_emotion(user_image)
    is_conflicting, facial_emotion, text_emotion = detect_conflicting_emotions(
        facial_emotion, user_text
    )

    prompt = create_prompt(user_text, facial_emotion, is_conflicting, text_emotion)
    tokenizer = _models["language_tok"]
    model = _models["language"]

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    full_response = tokenizer.decode(output[0])

    match = re.search(r"### Response:\s*(.*)", full_response, re.DOTALL)
    if match:
        response = re.sub(r"<\|end_of_text\|>$", "", match.group(1).strip()).strip()
    else:
        response = "I apologize, but I couldn't generate a proper response. Please try again."

    emotion_display = facial_emotion
    if is_conflicting:
        emotion_display = (
            f"{facial_emotion} (face) vs {text_emotion} (text) — conflicting. "
            "I notice there seems to be a difference between your facial expression and "
            "what you've shared. Would you feel comfortable sharing which emotion is "
            "closer to your experience right now?"
        )
    return response, emotion_display


def update_frame(frame, last_frame):
    if frame is not None:
        return frame.copy()
    return last_frame


def process_image(img):
    if img is None:
        return None
    processed = img.copy()
    h, w = processed.shape[0], processed.shape[1]
    t = 10
    processed[0:t, :] = [255, 255, 255]
    processed[h - t : h, :] = [255, 255, 255]
    processed[:, 0:t] = [255, 255, 255]
    processed[:, w - t : w] = [255, 255, 255]
    return processed


def add_text(history, text, last_frame):
    if not text.strip():
        return history, "", None, "Whenever you're ready, just type a message — we're here to listen."
    if last_frame is None:
        return history, text, None, "No webcam image detected. Please make sure your camera is working."
    processed_image = process_image(last_frame)
    history = history + [(text, None)]
    return history, "", processed_image, "Photo automatically captured for emotion analysis"


def bot(history, last_frame):
    user_message = history[-1][0]
    try:
        response, emotion = generate_response(user_message, last_frame)
        history[-1] = (user_message, f"{response}\n\n[Detected emotion: {emotion}]")
    except Exception as e:
        history[-1] = (user_message, f"Something went wrong while generating a response: {e}")
    return history


with gr.Blocks(theme=gr.themes.Soft(), title="Elyssa") as demo:
    gr.Markdown(
        """
        # Elyssa – Bringing Therapy Closer to You

        Elyssa sees beyond your words. As you share your thoughts, our caring system quietly observes
        the emotions written on your face – those subtle signals we often don't express in text.
        Just be yourself as you type, and Elyssa will blend what you say with what your expressions
        reveal, creating a space where support meets you exactly where you are.

        *This Space is the demo build. The production system runs as five FastAPI microservices on
        Kubernetes — see the [GitHub repository](https://github.com/Shravya29M/Elyssa).*
        """
    )

    status = gr.Textbox(
        value="Your webcam is streaming! No rush, take a deep breath, and begin when you're ready — we're always here for you.",
        label="Status",
        interactive=False,
    )

    with gr.Row():
        with gr.Column(scale=1):
            webcam = gr.Image(
                label="Live Camera Feed",
                type="numpy",
                sources=["webcam"],
                streaming=True,
                height=200,
            )
            captured = gr.Image(label="Last Captured Image", type="numpy", height=200)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                bubble_full_width=False,
                show_copy_button=True,
                avatar_images=("👤", "🤖"),
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here and press Enter or click Send...",
                    container=False,
                    scale=6,
                    show_label=False,
                )
                send_btn = gr.Button("Send", scale=1)
                clear = gr.Button("Clear Chat", scale=1)

    frame_state = gr.State(None)

    webcam.stream(
        update_frame, inputs=[webcam, frame_state], outputs=frame_state, show_progress=False
    )
    msg.submit(add_text, [chatbot, msg, frame_state], [chatbot, msg, captured, status]).then(
        bot, [chatbot, frame_state], chatbot
    )
    send_btn.click(add_text, [chatbot, msg, frame_state], [chatbot, msg, captured, status]).then(
        bot, [chatbot, frame_state], chatbot
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

    gr.Markdown(
        """
        ### Privacy Notice

        This application automatically captures your webcam images when you send a message to detect emotions.
        Images are processed in real time and are not stored. By using this application, you consent to facial
        emotion analysis for mental health support purposes.

        **Note:** the first message takes a few minutes while the models load.
        """
    )


if __name__ == "__main__":
    demo.queue().launch()
